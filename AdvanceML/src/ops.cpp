#include "advanceml/ops.hpp"

#include <dnnl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace advanceml {

namespace {

dnnl::engine& cpu_engine() {
    static dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    return engine;
}

dnnl::stream& cpu_stream() {
    static dnnl::stream stream(cpu_engine());
    return stream;
}

dnnl::memory reorder_to(dnnl::stream& stream, dnnl::engine& engine, dnnl::memory src,
                         const dnnl::memory::desc& dst_desc) {
    if (src.get_desc() == dst_desc) {
        return src;
    }
    dnnl::memory dst(dst_desc, engine);
    dnnl::reorder(src, dst).execute(stream, src, dst);
    stream.wait();
    return dst;
}

void reorder_into(dnnl::stream& stream, dnnl::memory src, dnnl::memory dst) {
    dnnl::reorder(src, dst).execute(stream, src, dst);
    stream.wait();
}

// dnnl::primitive_desc construction JIT-selects and compiles a kernel, which is far more
// expensive than executing it. Training loops call the same op shape thousands of times
// (once per batch per layer per epoch), so primitives are cached per op+exact-shape rather
// than rebuilt on every forward/backward call. Keyed on real dims (not an assumed constant
// batch size) because the dataset's last batch is typically smaller than kBatchSize.
size_t hash_combine(size_t seed, size_t value) {
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

struct GemmKey {
    size_t m, k, n;
    bool operator==(const GemmKey& other) const { return m == other.m && k == other.k && n == other.n; }
};

struct GemmKeyHash {
    size_t operator()(const GemmKey& key) const {
        size_t h = std::hash<size_t>{}(key.m);
        h = hash_combine(h, std::hash<size_t>{}(key.k));
        h = hash_combine(h, std::hash<size_t>{}(key.n));
        return h;
    }
};

struct GemmCacheEntry {
    dnnl::matmul::primitive_desc pd;
    dnnl::matmul prim;
};

std::unordered_map<GemmKey, GemmCacheEntry, GemmKeyHash>& gemm_cache() {
    static std::unordered_map<GemmKey, GemmCacheEntry, GemmKeyHash> cache;
    return cache;
}

std::vector<float> gemm(const float* a, const float* b, size_t m, size_t k, size_t n) {
    using dnnl::memory;
    dnnl::engine& engine = cpu_engine();

    memory::dims a_dims = {static_cast<dnnl::memory::dim>(m), static_cast<dnnl::memory::dim>(k)};
    memory::dims b_dims = {static_cast<dnnl::memory::dim>(k), static_cast<dnnl::memory::dim>(n)};
    memory::dims c_dims = {static_cast<dnnl::memory::dim>(m), static_cast<dnnl::memory::dim>(n)};

    auto a_md = memory::desc(a_dims, memory::data_type::f32, memory::format_tag::ab);
    auto b_md = memory::desc(b_dims, memory::data_type::f32, memory::format_tag::ab);
    auto c_md = memory::desc(c_dims, memory::data_type::f32, memory::format_tag::ab);

    std::vector<float> c(m * n);
    memory a_mem(a_md, engine, const_cast<float*>(a));
    memory b_mem(b_md, engine, const_cast<float*>(b));
    memory c_mem(c_md, engine, c.data());

    GemmKey key{m, k, n};
    auto& cache = gemm_cache();
    auto it = cache.find(key);
    if (it == cache.end()) {
        auto matmul_pd = dnnl::matmul::primitive_desc(engine, a_md, b_md, c_md);
        dnnl::matmul prim(matmul_pd);
        it = cache.emplace(key, GemmCacheEntry{std::move(matmul_pd), std::move(prim)}).first;
    }
    it->second.prim.execute(cpu_stream(), {
        {DNNL_ARG_SRC, a_mem},
        {DNNL_ARG_WEIGHTS, b_mem},
        {DNNL_ARG_DST, c_mem},
    });
    cpu_stream().wait();
    return c;
}

std::vector<float> transpose(const std::vector<float>& data, size_t rows, size_t cols) {
    std::vector<float> out(data.size());
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    return out;
}

struct PoolKey {
    size_t N, C, H, W, kernel_size, stride;
    dnnl::algorithm algorithm;
    bool operator==(const PoolKey& other) const {
        return N == other.N && C == other.C && H == other.H && W == other.W &&
               kernel_size == other.kernel_size && stride == other.stride && algorithm == other.algorithm;
    }
};

struct PoolKeyHash {
    size_t operator()(const PoolKey& key) const {
        size_t h = std::hash<size_t>{}(key.N);
        h = hash_combine(h, std::hash<size_t>{}(key.C));
        h = hash_combine(h, std::hash<size_t>{}(key.H));
        h = hash_combine(h, std::hash<size_t>{}(key.W));
        h = hash_combine(h, std::hash<size_t>{}(key.kernel_size));
        h = hash_combine(h, std::hash<size_t>{}(key.stride));
        h = hash_combine(h, std::hash<size_t>{}(static_cast<size_t>(key.algorithm)));
        return h;
    }
};

struct PoolCacheEntry {
    dnnl::pooling_forward::primitive_desc fwd_pd;
    dnnl::pooling_forward fwd_prim;
    std::optional<dnnl::pooling_backward::primitive_desc> bwd_pd;
    std::optional<dnnl::pooling_backward> bwd_prim;
};

std::unordered_map<PoolKey, PoolCacheEntry, PoolKeyHash>& pool_cache() {
    static std::unordered_map<PoolKey, PoolCacheEntry, PoolKeyHash> cache;
    return cache;
}

Tensor pool2d(const Tensor& x, size_t kernel_size, size_t stride, dnnl::algorithm algorithm, const char* op_name) {
    if (x.shape().size() != 4) {
        throw std::runtime_error(std::string(op_name) + ": expected a 4D (N, C, H, W) tensor");
    }
    const size_t N = x.shape()[0];
    const size_t C = x.shape()[1];
    const size_t H = x.shape()[2];
    const size_t W = x.shape()[3];
    if (kernel_size > H || kernel_size > W) {
        throw std::runtime_error(std::string(op_name) + ": kernel_size exceeds H or W");
    }
    const size_t h_out = (H - kernel_size) / stride + 1;
    const size_t w_out = (W - kernel_size) / stride + 1;

    using dnnl::memory;
    dnnl::engine& engine = cpu_engine();

    memory::dims src_dims = {static_cast<memory::dim>(N), static_cast<memory::dim>(C), static_cast<memory::dim>(H),
                              static_cast<memory::dim>(W)};
    memory::dims dst_dims = {static_cast<memory::dim>(N), static_cast<memory::dim>(C),
                              static_cast<memory::dim>(h_out), static_cast<memory::dim>(w_out)};
    memory::dims strides_dims = {static_cast<memory::dim>(stride), static_cast<memory::dim>(stride)};
    memory::dims kernel_dims = {static_cast<memory::dim>(kernel_size), static_cast<memory::dim>(kernel_size)};
    memory::dims dilation_dims = {0, 0};
    memory::dims padding_dims = {0, 0};

    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);

    PoolKey key{N, C, H, W, kernel_size, stride, algorithm};
    auto& cache = pool_cache();
    auto cache_it = cache.find(key);
    if (cache_it == cache.end()) {
        auto fwd_pd = dnnl::pooling_forward::primitive_desc(engine, dnnl::prop_kind::forward_training, algorithm,
                                                             src_md, dst_md, strides_dims, kernel_dims,
                                                             dilation_dims, padding_dims, padding_dims);
        dnnl::pooling_forward fwd_prim(fwd_pd);
        cache_it = cache.emplace(key, PoolCacheEntry{std::move(fwd_pd), std::move(fwd_prim), std::nullopt, std::nullopt}).first;
    }
    PoolCacheEntry& entry = cache_it->second;
    const dnnl::pooling_forward::primitive_desc& fwd_pd = entry.fwd_pd;

    memory src_mem(fwd_pd.src_desc(), engine, const_cast<float*>(x.data().data()));
    std::vector<float> out_data(N * C * h_out * w_out);
    memory dst_mem(fwd_pd.dst_desc(), engine, out_data.data());

    const bool needs_workspace = algorithm == dnnl::algorithm::pooling_max;
    auto workspace_data = std::make_shared<std::vector<uint8_t>>(needs_workspace ? fwd_pd.workspace_desc().get_size() : 0);
    std::unordered_map<int, memory> fwd_args = {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}};
    if (needs_workspace) {
        fwd_args[DNNL_ARG_WORKSPACE] = memory(fwd_pd.workspace_desc(), engine, workspace_data->data());
    }
    entry.fwd_prim.execute(cpu_stream(), fwd_args);
    cpu_stream().wait();

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = std::move(out_data);
    out_impl->shape = {N, C, h_out, w_out};
    out_impl->requires_grad = x.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {x.impl()};
        std::vector<size_t> x_shape = x.shape();
        node->backward_fn = [key, algorithm, strides_dims, kernel_dims, dilation_dims, padding_dims, x_shape,
                              workspace_data, needs_workspace](const Tensor& grad_output) -> std::vector<Tensor> {
            dnnl::engine& engine = cpu_engine();
            PoolCacheEntry& entry = pool_cache().at(key);
            if (!entry.bwd_pd.has_value()) {
                entry.bwd_pd = dnnl::pooling_backward::primitive_desc(
                    engine, algorithm, entry.fwd_pd.src_desc(), entry.fwd_pd.dst_desc(), strides_dims, kernel_dims,
                    dilation_dims, padding_dims, padding_dims, entry.fwd_pd);
                entry.bwd_prim = dnnl::pooling_backward(*entry.bwd_pd);
            }
            const dnnl::pooling_backward::primitive_desc& bwd_pd = *entry.bwd_pd;

            memory diff_dst_mem(bwd_pd.diff_dst_desc(), engine, const_cast<float*>(grad_output.data().data()));
            std::vector<float> grad_x_data(x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3]);
            memory diff_src_mem(bwd_pd.diff_src_desc(), engine, grad_x_data.data());

            std::unordered_map<int, memory> bwd_args = {{DNNL_ARG_DIFF_DST, diff_dst_mem},
                                                          {DNNL_ARG_DIFF_SRC, diff_src_mem}};
            if (needs_workspace) {
                bwd_args[DNNL_ARG_WORKSPACE] = memory(bwd_pd.workspace_desc(), engine, workspace_data->data());
            }
            entry.bwd_prim->execute(cpu_stream(), bwd_args);
            cpu_stream().wait();

            return {Tensor(std::move(grad_x_data), x_shape)};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

struct ConvKey {
    size_t N, c_in, H, W, c_out, kh, kw, stride, padding;
    bool operator==(const ConvKey& other) const {
        return N == other.N && c_in == other.c_in && H == other.H && W == other.W && c_out == other.c_out &&
               kh == other.kh && kw == other.kw && stride == other.stride && padding == other.padding;
    }
};

struct ConvKeyHash {
    size_t operator()(const ConvKey& key) const {
        size_t h = std::hash<size_t>{}(key.N);
        h = hash_combine(h, std::hash<size_t>{}(key.c_in));
        h = hash_combine(h, std::hash<size_t>{}(key.H));
        h = hash_combine(h, std::hash<size_t>{}(key.W));
        h = hash_combine(h, std::hash<size_t>{}(key.c_out));
        h = hash_combine(h, std::hash<size_t>{}(key.kh));
        h = hash_combine(h, std::hash<size_t>{}(key.kw));
        h = hash_combine(h, std::hash<size_t>{}(key.stride));
        h = hash_combine(h, std::hash<size_t>{}(key.padding));
        return h;
    }
};

struct ConvCacheEntry {
    dnnl::convolution_forward::primitive_desc fwd_pd;
    dnnl::convolution_forward fwd_prim;
    std::optional<dnnl::convolution_backward_data::primitive_desc> bwd_data_pd;
    std::optional<dnnl::convolution_backward_data> bwd_data_prim;
    std::optional<dnnl::convolution_backward_weights::primitive_desc> bwd_weights_pd;
    std::optional<dnnl::convolution_backward_weights> bwd_weights_prim;
};

std::unordered_map<ConvKey, ConvCacheEntry, ConvKeyHash>& conv_cache() {
    static std::unordered_map<ConvKey, ConvCacheEntry, ConvKeyHash> cache;
    return cache;
}

}  // namespace

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape().size() != 2 || b.shape().size() != 2 || a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error("matmul: expected 2D tensors with compatible inner dimensions");
    }
    const size_t m = a.shape()[0];
    const size_t k = a.shape()[1];
    const size_t n = b.shape()[1];

    std::vector<float> out_data = gemm(a.data().data(), b.data().data(), m, k, n);

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = std::move(out_data);
    out_impl->shape = {m, n};
    out_impl->requires_grad = a.requires_grad() || b.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {a.impl(), b.impl()};
        Tensor a_copy = a;
        Tensor b_copy = b;
        node->backward_fn = [a_copy, b_copy, m, k, n](const Tensor& grad_output) -> std::vector<Tensor> {
            std::vector<float> b_t = transpose(b_copy.data(), k, n);
            std::vector<float> grad_a_data = gemm(grad_output.data().data(), b_t.data(), m, n, k);

            std::vector<float> a_t = transpose(a_copy.data(), m, k);
            std::vector<float> grad_b_data = gemm(a_t.data(), grad_output.data().data(), k, m, n);

            return {Tensor(std::move(grad_a_data), {m, k}), Tensor(std::move(grad_b_data), {k, n})};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor relu(const Tensor& x) {
    std::vector<float> out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        out_data[i] = x.data()[i] > 0.0f ? x.data()[i] : 0.0f;
    }

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = std::move(out_data);
    out_impl->shape = x.shape();
    out_impl->requires_grad = x.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {x.impl()};
        Tensor x_copy = x;
        node->backward_fn = [x_copy](const Tensor& grad_output) -> std::vector<Tensor> {
            std::vector<float> grad_x(x_copy.numel());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                grad_x[i] = x_copy.data()[i] > 0.0f ? grad_output.data()[i] : 0.0f;
            }
            return {Tensor(std::move(grad_x), x_copy.shape())};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor mse_loss(const Tensor& pred, const Tensor& target) {
    if (pred.shape() != target.shape()) {
        throw std::runtime_error("mse_loss: pred and target shapes must match");
    }
    const size_t n = pred.numel();

    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float diff = pred.data()[i] - target.data()[i];
        sum_sq += diff * diff;
    }
    const float loss_value = sum_sq / static_cast<float>(n);

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = {loss_value};
    out_impl->shape = {1};
    out_impl->requires_grad = pred.requires_grad() || target.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {pred.impl(), target.impl()};
        Tensor pred_copy = pred;
        Tensor target_copy = target;
        node->backward_fn = [pred_copy, target_copy, n](const Tensor& grad_output) -> std::vector<Tensor> {
            const float upstream = grad_output.data()[0];
            std::vector<float> grad_pred(n);
            std::vector<float> grad_target(n);
            for (size_t i = 0; i < n; ++i) {
                const float diff = pred_copy.data()[i] - target_copy.data()[i];
                const float g = upstream * 2.0f * diff / static_cast<float>(n);
                grad_pred[i] = g;
                grad_target[i] = -g;
            }
            return {Tensor(std::move(grad_pred), pred_copy.shape()), Tensor(std::move(grad_target), target_copy.shape())};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor abs_loss(const Tensor& pred, const Tensor& target) {
    if (pred.shape() != target.shape()) {
        throw std::runtime_error("abs_loss: pred and target shapes must match");
    }
    const size_t n = pred.numel();

    float sum_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum_abs += std::abs(pred.data()[i] - target.data()[i]);
    }
    const float loss_value = sum_abs / static_cast<float>(n);

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = {loss_value};
    out_impl->shape = {1};
    out_impl->requires_grad = pred.requires_grad() || target.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {pred.impl(), target.impl()};
        Tensor pred_copy = pred;
        Tensor target_copy = target;
        node->backward_fn = [pred_copy, target_copy, n](const Tensor& grad_output) -> std::vector<Tensor> {
            const float upstream = grad_output.data()[0];
            std::vector<float> grad_pred(n);
            std::vector<float> grad_target(n);
            for (size_t i = 0; i < n; ++i) {
                const float diff = pred_copy.data()[i] - target_copy.data()[i];
                const float sign = diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f);
                const float g = upstream * sign / static_cast<float>(n);
                grad_pred[i] = g;
                grad_target[i] = -g;
            }
            return {Tensor(std::move(grad_pred), pred_copy.shape()), Tensor(std::move(grad_target), target_copy.shape())};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor binary_cross_entropy(const Tensor& pred, const Tensor& target) {
    if (pred.shape() != target.shape()) {
        throw std::runtime_error("binary_cross_entropy: pred and target shapes must match");
    }
    if (pred.shape().size() != 1 && pred.shape().size() != 2) {
        throw std::runtime_error("binary_cross_entropy: expected a 1D or 2D tensor");
    }
    constexpr float kEpsilon = 1e-15f;
    const size_t n = pred.shape().size() == 2 ? pred.shape()[0] : 1;
    const size_t numel = pred.numel();

    float sum = 0.0f;
    for (size_t i = 0; i < numel; ++i) {
        const float clipped = std::clamp(pred.data()[i], kEpsilon, 1.0f - kEpsilon);
        sum += target.data()[i] * std::log(clipped) + (1.0f - target.data()[i]) * std::log(1.0f - clipped);
    }
    const float loss_value = -sum / static_cast<float>(n);

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = {loss_value};
    out_impl->shape = {1};
    out_impl->requires_grad = pred.requires_grad() || target.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {pred.impl(), target.impl()};
        Tensor pred_copy = pred;
        Tensor target_copy = target;
        node->backward_fn = [pred_copy, target_copy, n, numel](const Tensor& grad_output) -> std::vector<Tensor> {
            constexpr float kEpsilon = 1e-15f;
            const float upstream = grad_output.data()[0];
            std::vector<float> grad_pred(numel);
            std::vector<float> grad_target(numel);
            for (size_t i = 0; i < numel; ++i) {
                const float clipped = std::clamp(pred_copy.data()[i], kEpsilon, 1.0f - kEpsilon);
                const float t = target_copy.data()[i];
                grad_pred[i] = -upstream * (t / clipped - (1.0f - t) / (1.0f - clipped)) / static_cast<float>(n);
                grad_target[i] = -upstream * (std::log(clipped) - std::log(1.0f - clipped)) / static_cast<float>(n);
            }
            return {Tensor(std::move(grad_pred), pred_copy.shape()), Tensor(std::move(grad_target), target_copy.shape())};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor sigmoid(const Tensor& x) {
    std::vector<float> out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        out_data[i] = 1.0f / (1.0f + std::exp(-x.data()[i]));
    }

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->shape = x.shape();
    out_impl->requires_grad = x.requires_grad();
    out_impl->data = out_data;

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {x.impl()};
        std::vector<float> y_copy = out_data;
        node->backward_fn = [y_copy](const Tensor& grad_output) -> std::vector<Tensor> {
            std::vector<float> grad_x(y_copy.size());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                grad_x[i] = grad_output.data()[i] * y_copy[i] * (1.0f - y_copy[i]);
            }
            return {Tensor(std::move(grad_x), grad_output.shape())};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor leaky_relu(const Tensor& x, float negative_slope) {
    std::vector<float> out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        const float v = x.data()[i];
        out_data[i] = v > 0.0f ? v : negative_slope * v;
    }

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = std::move(out_data);
    out_impl->shape = x.shape();
    out_impl->requires_grad = x.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {x.impl()};
        Tensor x_copy = x;
        node->backward_fn = [x_copy, negative_slope](const Tensor& grad_output) -> std::vector<Tensor> {
            std::vector<float> grad_x(x_copy.numel());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                grad_x[i] = x_copy.data()[i] > 0.0f ? grad_output.data()[i] : negative_slope * grad_output.data()[i];
            }
            return {Tensor(std::move(grad_x), x_copy.shape())};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor gelu(const Tensor& x) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    constexpr float kInvSqrt2Pi = 0.39894228040143267794f;

    std::vector<float> out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        const float v = x.data()[i];
        out_data[i] = v * 0.5f * (1.0f + std::erf(v * kInvSqrt2));
    }

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = std::move(out_data);
    out_impl->shape = x.shape();
    out_impl->requires_grad = x.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {x.impl()};
        Tensor x_copy = x;
        node->backward_fn = [x_copy, kInvSqrt2, kInvSqrt2Pi](const Tensor& grad_output) -> std::vector<Tensor> {
            std::vector<float> grad_x(x_copy.numel());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                const float v = x_copy.data()[i];
                const float cdf = 0.5f * (1.0f + std::erf(v * kInvSqrt2));
                const float pdf = kInvSqrt2Pi * std::exp(-0.5f * v * v);
                grad_x[i] = grad_output.data()[i] * (cdf + v * pdf);
            }
            return {Tensor(std::move(grad_x), x_copy.shape())};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor tanh(const Tensor& x) {
    std::vector<float> out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        out_data[i] = std::tanh(x.data()[i]);
    }

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->shape = x.shape();
    out_impl->requires_grad = x.requires_grad();
    out_impl->data = out_data;

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {x.impl()};
        std::vector<float> y_copy = out_data;
        node->backward_fn = [y_copy](const Tensor& grad_output) -> std::vector<Tensor> {
            std::vector<float> grad_x(y_copy.size());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                grad_x[i] = grad_output.data()[i] * (1.0f - y_copy[i] * y_copy[i]);
            }
            return {Tensor(std::move(grad_x), grad_output.shape())};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor identity(const Tensor& x) {
    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = x.data();
    out_impl->shape = x.shape();
    out_impl->requires_grad = x.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {x.impl()};
        node->backward_fn = [](const Tensor& grad_output) -> std::vector<Tensor> {
            return {Tensor(grad_output.data(), grad_output.shape())};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor prelu(const Tensor& x, const Tensor& a) {
    const size_t num_parameters = a.numel();
    if (num_parameters != 1 && (x.shape().empty() || x.shape().back() != num_parameters)) {
        throw std::runtime_error("prelu: a.numel() must be 1 or equal to x.shape().back()");
    }

    std::vector<float> out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        const size_t c = num_parameters == 1 ? 0 : i % num_parameters;
        const float v = x.data()[i];
        out_data[i] = v > 0.0f ? v : a.data()[c] * v;
    }

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = std::move(out_data);
    out_impl->shape = x.shape();
    out_impl->requires_grad = x.requires_grad() || a.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {x.impl(), a.impl()};
        Tensor x_copy = x;
        Tensor a_copy = a;
        node->backward_fn = [x_copy, a_copy, num_parameters](const Tensor& grad_output) -> std::vector<Tensor> {
            std::vector<float> grad_x(x_copy.numel());
            std::vector<float> grad_a(num_parameters, 0.0f);
            for (size_t i = 0; i < grad_x.size(); ++i) {
                const size_t c = num_parameters == 1 ? 0 : i % num_parameters;
                const float v = x_copy.data()[i];
                if (v > 0.0f) {
                    grad_x[i] = grad_output.data()[i];
                } else {
                    grad_x[i] = a_copy.data()[c] * grad_output.data()[i];
                    grad_a[c] += grad_output.data()[i] * v;
                }
            }
            return {Tensor(std::move(grad_x), x_copy.shape()), Tensor(std::move(grad_a), {num_parameters})};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor elu(const Tensor& x, float alpha) {
    std::vector<float> out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        const float v = x.data()[i];
        out_data[i] = v > 0.0f ? v : alpha * std::expm1(v);
    }

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = std::move(out_data);
    out_impl->shape = x.shape();
    out_impl->requires_grad = x.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {x.impl()};
        Tensor x_copy = x;
        node->backward_fn = [x_copy, alpha](const Tensor& grad_output) -> std::vector<Tensor> {
            std::vector<float> grad_x(x_copy.numel());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                const float v = x_copy.data()[i];
                const float local = v > 0.0f ? 1.0f : alpha * std::exp(v);
                grad_x[i] = grad_output.data()[i] * local;
            }
            return {Tensor(std::move(grad_x), x_copy.shape())};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor selu(const Tensor& x) {
    constexpr float kAlpha = 1.6732632423543772f;
    constexpr float kScale = 1.0507009873554805f;

    std::vector<float> out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        const float v = x.data()[i];
        out_data[i] = kScale * (v > 0.0f ? v : kAlpha * std::expm1(v));
    }

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = std::move(out_data);
    out_impl->shape = x.shape();
    out_impl->requires_grad = x.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {x.impl()};
        Tensor x_copy = x;
        node->backward_fn = [x_copy](const Tensor& grad_output) -> std::vector<Tensor> {
            std::vector<float> grad_x(x_copy.numel());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                const float v = x_copy.data()[i];
                const float local = kScale * (v > 0.0f ? 1.0f : kAlpha * std::exp(v));
                grad_x[i] = grad_output.data()[i] * local;
            }
            return {Tensor(std::move(grad_x), x_copy.shape())};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor softplus(const Tensor& x) {
    std::vector<float> out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        const float v = x.data()[i];
        out_data[i] = std::max(v, 0.0f) + std::log1p(std::exp(-std::fabs(v)));
    }

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = std::move(out_data);
    out_impl->shape = x.shape();
    out_impl->requires_grad = x.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {x.impl()};
        Tensor x_copy = x;
        node->backward_fn = [x_copy](const Tensor& grad_output) -> std::vector<Tensor> {
            std::vector<float> grad_x(x_copy.numel());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                const float sig = 1.0f / (1.0f + std::exp(-x_copy.data()[i]));
                grad_x[i] = grad_output.data()[i] * sig;
            }
            return {Tensor(std::move(grad_x), x_copy.shape())};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor swish(const Tensor& x, float beta) {
    std::vector<float> out_data(x.numel());
    auto sig = std::make_shared<std::vector<float>>(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        const float v = x.data()[i];
        const float s = 1.0f / (1.0f + std::exp(-beta * v));
        (*sig)[i] = s;
        out_data[i] = v * s;
    }

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = std::move(out_data);
    out_impl->shape = x.shape();
    out_impl->requires_grad = x.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {x.impl()};
        Tensor x_copy = x;
        node->backward_fn = [x_copy, sig, beta](const Tensor& grad_output) -> std::vector<Tensor> {
            std::vector<float> grad_x(x_copy.numel());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                const float s = (*sig)[i];
                const float local = s + beta * x_copy.data()[i] * s * (1.0f - s);
                grad_x[i] = grad_output.data()[i] * local;
            }
            return {Tensor(std::move(grad_x), x_copy.shape())};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor mish(const Tensor& x) {
    std::vector<float> out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        const float v = x.data()[i];
        const float sp = std::max(v, 0.0f) + std::log1p(std::exp(-std::fabs(v)));
        out_data[i] = v * std::tanh(sp);
    }

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = std::move(out_data);
    out_impl->shape = x.shape();
    out_impl->requires_grad = x.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {x.impl()};
        Tensor x_copy = x;
        node->backward_fn = [x_copy](const Tensor& grad_output) -> std::vector<Tensor> {
            std::vector<float> grad_x(x_copy.numel());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                const float v = x_copy.data()[i];
                const float sp = std::max(v, 0.0f) + std::log1p(std::exp(-std::fabs(v)));
                const float t = std::tanh(sp);
                const float sig = 1.0f / (1.0f + std::exp(-v));
                const float local = t + v * (1.0f - t * t) * sig;
                grad_x[i] = grad_output.data()[i] * local;
            }
            return {Tensor(std::move(grad_x), x_copy.shape())};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor hardtanh(const Tensor& x, float min_val, float max_val) {
    if (max_val <= min_val) {
        throw std::runtime_error("hardtanh: max_val must be greater than min_val");
    }

    std::vector<float> out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        out_data[i] = std::clamp(x.data()[i], min_val, max_val);
    }

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = std::move(out_data);
    out_impl->shape = x.shape();
    out_impl->requires_grad = x.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {x.impl()};
        Tensor x_copy = x;
        node->backward_fn = [x_copy, min_val, max_val](const Tensor& grad_output) -> std::vector<Tensor> {
            std::vector<float> grad_x(x_copy.numel());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                const float v = x_copy.data()[i];
                grad_x[i] = (v > min_val && v < max_val) ? grad_output.data()[i] : 0.0f;
            }
            return {Tensor(std::move(grad_x), x_copy.shape())};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor hardsigmoid(const Tensor& x) {
    std::vector<float> out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        out_data[i] = std::clamp(x.data()[i] / 6.0f + 0.5f, 0.0f, 1.0f);
    }

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = std::move(out_data);
    out_impl->shape = x.shape();
    out_impl->requires_grad = x.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {x.impl()};
        Tensor x_copy = x;
        node->backward_fn = [x_copy](const Tensor& grad_output) -> std::vector<Tensor> {
            std::vector<float> grad_x(x_copy.numel());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                const float v = x_copy.data()[i];
                grad_x[i] = (v > -3.0f && v < 3.0f) ? grad_output.data()[i] / 6.0f : 0.0f;
            }
            return {Tensor(std::move(grad_x), x_copy.shape())};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor softmax(const Tensor& x) {
    if (x.shape().size() != 1 && x.shape().size() != 2) {
        throw std::runtime_error("softmax: expected a 1D or 2D tensor");
    }
    const size_t rows = x.shape().size() == 2 ? x.shape()[0] : 1;
    const size_t cols = x.shape().size() == 2 ? x.shape()[1] : x.shape()[0];

    std::vector<float> out_data(x.numel());
    for (size_t r = 0; r < rows; ++r) {
        const size_t base = r * cols;
        float row_max = x.data()[base];
        for (size_t c = 1; c < cols; ++c) {
            row_max = std::max(row_max, x.data()[base + c]);
        }
        float sum_exp = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            const float e = std::exp(x.data()[base + c] - row_max);
            out_data[base + c] = e;
            sum_exp += e;
        }
        for (size_t c = 0; c < cols; ++c) {
            out_data[base + c] /= sum_exp;
        }
    }

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->shape = x.shape();
    out_impl->requires_grad = x.requires_grad();
    out_impl->data = out_data;

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {x.impl()};
        std::vector<float> y_copy = out_data;
        node->backward_fn = [y_copy, rows, cols](const Tensor& grad_output) -> std::vector<Tensor> {
            std::vector<float> grad_x(y_copy.size());
            for (size_t r = 0; r < rows; ++r) {
                const size_t base = r * cols;
                float dot = 0.0f;
                for (size_t c = 0; c < cols; ++c) {
                    dot += grad_output.data()[base + c] * y_copy[base + c];
                }
                for (size_t c = 0; c < cols; ++c) {
                    grad_x[base + c] = y_copy[base + c] * (grad_output.data()[base + c] - dot);
                }
            }
            return {Tensor(std::move(grad_x), grad_output.shape())};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor cross_entropy_loss(const Tensor& pred, const Tensor& target) {
    if (pred.shape() != target.shape()) {
        throw std::runtime_error("cross_entropy_loss: pred and target shapes must match");
    }
    if (pred.shape().size() != 1 && pred.shape().size() != 2) {
        throw std::runtime_error("cross_entropy_loss: expected a 1D or 2D tensor");
    }
    constexpr float kEpsilon = 1e-15f;
    const size_t n = pred.shape().size() == 2 ? pred.shape()[0] : 1;
    const size_t numel = pred.numel();

    float sum = 0.0f;
    for (size_t i = 0; i < numel; ++i) {
        const float clipped = std::clamp(pred.data()[i], kEpsilon, 1.0f - kEpsilon);
        sum += target.data()[i] * std::log(clipped);
    }
    const float loss_value = -sum / static_cast<float>(n);

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = {loss_value};
    out_impl->shape = {1};
    out_impl->requires_grad = pred.requires_grad() || target.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {pred.impl(), target.impl()};
        Tensor pred_copy = pred;
        Tensor target_copy = target;
        node->backward_fn = [pred_copy, target_copy, n, numel](const Tensor& grad_output) -> std::vector<Tensor> {
            constexpr float kEpsilon = 1e-15f;
            const float upstream = grad_output.data()[0];
            std::vector<float> grad_pred(numel);
            std::vector<float> grad_target(numel);
            for (size_t i = 0; i < numel; ++i) {
                const float clipped = std::clamp(pred_copy.data()[i], kEpsilon, 1.0f - kEpsilon);
                grad_pred[i] = -upstream * target_copy.data()[i] / (clipped * static_cast<float>(n));
                grad_target[i] = -upstream * std::log(clipped) / static_cast<float>(n);
            }
            return {Tensor(std::move(grad_pred), pred_copy.shape()), Tensor(std::move(grad_target), target_copy.shape())};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor conv2d(const Tensor& x, const Tensor& weight, const Tensor& bias, size_t stride, size_t padding) {
    if (x.shape().size() != 4) {
        throw std::runtime_error("conv2d: expected a 4D (N, C, H, W) input tensor");
    }
    if (weight.shape().size() != 4) {
        throw std::runtime_error("conv2d: expected a 4D (out_channels, in_channels, kh, kw) weight tensor");
    }
    if (bias.shape().size() != 1 || bias.shape()[0] != weight.shape()[0]) {
        throw std::runtime_error("conv2d: bias must be 1D with out_channels entries");
    }
    if (x.shape()[1] != weight.shape()[1]) {
        throw std::runtime_error("conv2d: input channel count must match weight's in_channels");
    }
    const size_t N = x.shape()[0];
    const size_t c_in = x.shape()[1];
    const size_t H = x.shape()[2];
    const size_t W = x.shape()[3];
    const size_t c_out = weight.shape()[0];
    const size_t kh = weight.shape()[2];
    const size_t kw = weight.shape()[3];
    if (H + 2 * padding < kh || W + 2 * padding < kw) {
        throw std::runtime_error("conv2d: kernel is larger than the padded input");
    }
    const size_t h_out = (H + 2 * padding - kh) / stride + 1;
    const size_t w_out = (W + 2 * padding - kw) / stride + 1;

    using dnnl::memory;
    dnnl::engine& engine = cpu_engine();

    memory::dims src_dims = {static_cast<memory::dim>(N), static_cast<memory::dim>(c_in),
                              static_cast<memory::dim>(H), static_cast<memory::dim>(W)};
    memory::dims weights_dims = {static_cast<memory::dim>(c_out), static_cast<memory::dim>(c_in),
                                  static_cast<memory::dim>(kh), static_cast<memory::dim>(kw)};
    memory::dims bias_dims = {static_cast<memory::dim>(c_out)};
    memory::dims dst_dims = {static_cast<memory::dim>(N), static_cast<memory::dim>(c_out),
                              static_cast<memory::dim>(h_out), static_cast<memory::dim>(w_out)};
    memory::dims strides_dims = {static_cast<memory::dim>(stride), static_cast<memory::dim>(stride)};
    memory::dims padding_dims = {static_cast<memory::dim>(padding), static_cast<memory::dim>(padding)};

    // format_tag::any (instead of a fixed nchw/oihw layout) lets oneDNN pick its optimized
    // blocked-layout AVX-512 VNNI kernel (`jit:avx512_core`) instead of silently falling back
    // to the unoptimized reference `jit_uni_ncsp_convolution:conv+ref:any` path that plain
    // NCHW/OIHW memory formats force it into -- confirmed via DNNL_VERBOSE, and the dominant
    // cost of a conv2d call, dwarfing primitive_desc construction. The rest of the codebase
    // (Tensor) only ever holds plain contiguous NCHW/OIHW data, so the plain<->optimal layout
    // conversion happens via an explicit oneDNN reorder at the boundary of each call.
    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::any);
    auto weights_md = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::any);
    auto bias_md = memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::x);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::any);

    ConvKey key{N, c_in, H, W, c_out, kh, kw, stride, padding};
    auto& cache = conv_cache();
    auto cache_it = cache.find(key);
    if (cache_it == cache.end()) {
        auto fwd_pd = dnnl::convolution_forward::primitive_desc(
            engine, dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_direct, src_md, weights_md,
            bias_md, dst_md, strides_dims, padding_dims, padding_dims);
        dnnl::convolution_forward fwd_prim(fwd_pd);
        cache_it = cache.emplace(key, ConvCacheEntry{std::move(fwd_pd), std::move(fwd_prim), std::nullopt,
                                                      std::nullopt, std::nullopt, std::nullopt}).first;
    }
    ConvCacheEntry& entry = cache_it->second;
    const dnnl::convolution_forward::primitive_desc& fwd_pd = entry.fwd_pd;

    auto plain_src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto plain_weights_md = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::oihw);
    auto plain_dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);

    memory src_plain(plain_src_md, engine, const_cast<float*>(x.data().data()));
    memory weights_plain(plain_weights_md, engine, const_cast<float*>(weight.data().data()));
    memory bias_mem(fwd_pd.bias_desc(), engine, const_cast<float*>(bias.data().data()));

    memory src_mem = reorder_to(cpu_stream(), engine, src_plain, fwd_pd.src_desc());
    memory weights_mem = reorder_to(cpu_stream(), engine, weights_plain, fwd_pd.weights_desc());
    memory dst_mem(fwd_pd.dst_desc(), engine);

    entry.fwd_prim.execute(cpu_stream(), {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_BIAS, bias_mem},
        {DNNL_ARG_DST, dst_mem},
    });
    cpu_stream().wait();

    std::vector<float> out_data(N * c_out * h_out * w_out);
    memory dst_plain(plain_dst_md, engine, out_data.data());
    reorder_into(cpu_stream(), dst_mem, dst_plain);

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = std::move(out_data);
    out_impl->shape = {N, c_out, h_out, w_out};
    out_impl->requires_grad = x.requires_grad() || weight.requires_grad() || bias.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {x.impl(), weight.impl(), bias.impl()};
        Tensor x_copy = x;
        Tensor weight_copy = weight;
        node->backward_fn = [key, x_copy, weight_copy, strides_dims, padding_dims,
                              bias_dims](const Tensor& grad_output) -> std::vector<Tensor> {
            dnnl::engine& engine = cpu_engine();
            dnnl::stream& stream = cpu_stream();
            ConvCacheEntry& entry = conv_cache().at(key);
            const dnnl::convolution_forward::primitive_desc& fwd_pd = entry.fwd_pd;

            memory::dims x_dims = {static_cast<memory::dim>(x_copy.shape()[0]), static_cast<memory::dim>(x_copy.shape()[1]),
                                    static_cast<memory::dim>(x_copy.shape()[2]), static_cast<memory::dim>(x_copy.shape()[3])};
            memory::dims w_dims = {static_cast<memory::dim>(weight_copy.shape()[0]), static_cast<memory::dim>(weight_copy.shape()[1]),
                                    static_cast<memory::dim>(weight_copy.shape()[2]), static_cast<memory::dim>(weight_copy.shape()[3])};
            memory::dims go_dims = {static_cast<memory::dim>(grad_output.shape()[0]), static_cast<memory::dim>(grad_output.shape()[1]),
                                     static_cast<memory::dim>(grad_output.shape()[2]), static_cast<memory::dim>(grad_output.shape()[3])};
            auto plain_src_md = memory::desc(x_dims, memory::data_type::f32, memory::format_tag::nchw);
            auto plain_weights_md = memory::desc(w_dims, memory::data_type::f32, memory::format_tag::oihw);
            auto plain_dst_md = memory::desc(go_dims, memory::data_type::f32, memory::format_tag::nchw);

            memory diff_dst_plain(plain_dst_md, engine, const_cast<float*>(grad_output.data().data()));
            memory weights_plain(plain_weights_md, engine, const_cast<float*>(weight_copy.data().data()));
            memory src_plain(plain_src_md, engine, const_cast<float*>(x_copy.data().data()));

            if (!entry.bwd_data_pd.has_value()) {
                auto diff_src_any = memory::desc(x_dims, memory::data_type::f32, memory::format_tag::any);
                auto weights_any = memory::desc(w_dims, memory::data_type::f32, memory::format_tag::any);
                auto diff_dst_any = memory::desc(go_dims, memory::data_type::f32, memory::format_tag::any);
                entry.bwd_data_pd = dnnl::convolution_backward_data::primitive_desc(
                    engine, dnnl::algorithm::convolution_direct, diff_src_any, weights_any, diff_dst_any,
                    strides_dims, padding_dims, padding_dims, fwd_pd);
                entry.bwd_data_prim = dnnl::convolution_backward_data(*entry.bwd_data_pd);
            }
            const dnnl::convolution_backward_data::primitive_desc& bwd_data_pd = *entry.bwd_data_pd;
            memory diff_dst_opt = reorder_to(stream, engine, diff_dst_plain, bwd_data_pd.diff_dst_desc());
            memory weights_opt = reorder_to(stream, engine, weights_plain, bwd_data_pd.weights_desc());
            memory diff_src_opt(bwd_data_pd.diff_src_desc(), engine);
            entry.bwd_data_prim->execute(stream, {
                {DNNL_ARG_DIFF_DST, diff_dst_opt},
                {DNNL_ARG_WEIGHTS, weights_opt},
                {DNNL_ARG_DIFF_SRC, diff_src_opt},
            });
            stream.wait();
            std::vector<float> grad_x_data(x_copy.numel());
            memory diff_src_plain(plain_src_md, engine, grad_x_data.data());
            reorder_into(stream, diff_src_opt, diff_src_plain);

            if (!entry.bwd_weights_pd.has_value()) {
                auto src_any = memory::desc(x_dims, memory::data_type::f32, memory::format_tag::any);
                auto diff_weights_any = memory::desc(w_dims, memory::data_type::f32, memory::format_tag::any);
                auto diff_dst_any = memory::desc(go_dims, memory::data_type::f32, memory::format_tag::any);
                entry.bwd_weights_pd = dnnl::convolution_backward_weights::primitive_desc(
                    engine, dnnl::algorithm::convolution_direct, src_any, diff_weights_any, fwd_pd.bias_desc(),
                    diff_dst_any, strides_dims, padding_dims, padding_dims, fwd_pd);
                entry.bwd_weights_prim = dnnl::convolution_backward_weights(*entry.bwd_weights_pd);
            }
            const dnnl::convolution_backward_weights::primitive_desc& bwd_weights_pd = *entry.bwd_weights_pd;
            memory src_opt = reorder_to(stream, engine, src_plain, bwd_weights_pd.src_desc());
            memory diff_dst_opt2 = reorder_to(stream, engine, diff_dst_plain, bwd_weights_pd.diff_dst_desc());
            memory diff_weights_opt(bwd_weights_pd.diff_weights_desc(), engine);
            std::vector<float> grad_b_data(static_cast<size_t>(bias_dims[0]));
            memory diff_bias_mem(bwd_weights_pd.diff_bias_desc(), engine, grad_b_data.data());
            entry.bwd_weights_prim->execute(stream, {
                {DNNL_ARG_SRC, src_opt},
                {DNNL_ARG_DIFF_DST, diff_dst_opt2},
                {DNNL_ARG_DIFF_WEIGHTS, diff_weights_opt},
                {DNNL_ARG_DIFF_BIAS, diff_bias_mem},
            });
            stream.wait();
            std::vector<float> grad_w_data(weight_copy.numel());
            memory diff_weights_plain(plain_weights_md, engine, grad_w_data.data());
            reorder_into(stream, diff_weights_opt, diff_weights_plain);

            return {
                Tensor(std::move(grad_x_data), x_copy.shape()),
                Tensor(std::move(grad_w_data), weight_copy.shape()),
                Tensor(std::move(grad_b_data), {static_cast<size_t>(bias_dims[0])}),
            };
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor max_pool2d(const Tensor& x, size_t kernel_size, size_t stride) {
    return pool2d(x, kernel_size, stride, dnnl::algorithm::pooling_max, "max_pool2d");
}

Tensor avg_pool2d(const Tensor& x, size_t kernel_size, size_t stride) {
    return pool2d(x, kernel_size, stride, dnnl::algorithm::pooling_avg_exclude_padding, "avg_pool2d");
}

Tensor flatten(const Tensor& x) {
    if (x.shape().empty()) {
        throw std::runtime_error("flatten: expected a tensor with at least 1 dimension");
    }
    const size_t n = x.shape()[0];
    const size_t rest = n == 0 ? 0 : x.numel() / n;

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = x.data();
    out_impl->shape = {n, rest};
    out_impl->requires_grad = x.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {x.impl()};
        std::vector<size_t> x_shape = x.shape();
        node->backward_fn = [x_shape](const Tensor& grad_output) -> std::vector<Tensor> {
            return {Tensor(grad_output.data(), x_shape)};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor batch_norm2d(const Tensor& x, const Tensor& gamma, const Tensor& beta, std::vector<float>& running_mean,
                     std::vector<float>& running_var, bool training, float momentum, float eps) {
    if (x.shape().size() != 4) {
        throw std::runtime_error("batch_norm2d: expected a 4D (N, C, H, W) input tensor");
    }
    const size_t N = x.shape()[0];
    const size_t C = x.shape()[1];
    const size_t H = x.shape()[2];
    const size_t W = x.shape()[3];
    if (gamma.shape() != std::vector<size_t>{C} || beta.shape() != std::vector<size_t>{C}) {
        throw std::runtime_error("batch_norm2d: gamma and beta must be 1D with C entries");
    }
    if (running_mean.size() != C || running_var.size() != C) {
        throw std::runtime_error("batch_norm2d: running_mean and running_var must have C entries");
    }
    const size_t plane = H * W;
    const size_t m = N * plane;

    std::vector<float> mean(C);
    std::vector<float> var(C);
    if (training) {
        for (size_t c = 0; c < C; ++c) {
            float sum = 0.0f;
            for (size_t n = 0; n < N; ++n) {
                const size_t base = (n * C + c) * plane;
                for (size_t p = 0; p < plane; ++p) {
                    sum += x.data()[base + p];
                }
            }
            mean[c] = sum / static_cast<float>(m);
        }
        for (size_t c = 0; c < C; ++c) {
            float sum_sq = 0.0f;
            for (size_t n = 0; n < N; ++n) {
                const size_t base = (n * C + c) * plane;
                for (size_t p = 0; p < plane; ++p) {
                    const float d = x.data()[base + p] - mean[c];
                    sum_sq += d * d;
                }
            }
            var[c] = sum_sq / static_cast<float>(m);
        }
        for (size_t c = 0; c < C; ++c) {
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean[c];
            running_var[c] = (1.0f - momentum) * running_var[c] + momentum * var[c];
        }
    } else {
        mean = running_mean;
        var = running_var;
    }

    std::vector<float> std_inv(C);
    for (size_t c = 0; c < C; ++c) {
        std_inv[c] = 1.0f / std::sqrt(var[c] + eps);
    }

    auto xhat = std::make_shared<std::vector<float>>(x.numel());
    std::vector<float> out_data(x.numel());
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const size_t base = (n * C + c) * plane;
            for (size_t p = 0; p < plane; ++p) {
                const float xh = (x.data()[base + p] - mean[c]) * std_inv[c];
                (*xhat)[base + p] = xh;
                out_data[base + p] = gamma.data()[c] * xh + beta.data()[c];
            }
        }
    }

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = std::move(out_data);
    out_impl->shape = x.shape();
    out_impl->requires_grad = x.requires_grad() || gamma.requires_grad() || beta.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {x.impl(), gamma.impl(), beta.impl()};
        Tensor gamma_copy = gamma;
        std::vector<size_t> x_shape = x.shape();
        node->backward_fn = [xhat, std_inv, gamma_copy, x_shape, N, C, plane, m,
                              training](const Tensor& grad_output) -> std::vector<Tensor> {
            std::vector<float> grad_gamma(C, 0.0f);
            std::vector<float> grad_beta(C, 0.0f);
            for (size_t n = 0; n < N; ++n) {
                for (size_t c = 0; c < C; ++c) {
                    const size_t base = (n * C + c) * plane;
                    for (size_t p = 0; p < plane; ++p) {
                        grad_gamma[c] += grad_output.data()[base + p] * (*xhat)[base + p];
                        grad_beta[c] += grad_output.data()[base + p];
                    }
                }
            }

            std::vector<float> grad_x(N * C * plane);
            if (training) {
                for (size_t c = 0; c < C; ++c) {
                    float sum_dout = 0.0f;
                    float sum_dout_xhat = 0.0f;
                    for (size_t n = 0; n < N; ++n) {
                        const size_t base = (n * C + c) * plane;
                        for (size_t p = 0; p < plane; ++p) {
                            sum_dout += grad_output.data()[base + p];
                            sum_dout_xhat += grad_output.data()[base + p] * (*xhat)[base + p];
                        }
                    }
                    const float coeff = gamma_copy.data()[c] * std_inv[c] / static_cast<float>(m);
                    for (size_t n = 0; n < N; ++n) {
                        const size_t base = (n * C + c) * plane;
                        for (size_t p = 0; p < plane; ++p) {
                            grad_x[base + p] = coeff * (static_cast<float>(m) * grad_output.data()[base + p] -
                                                         sum_dout - (*xhat)[base + p] * sum_dout_xhat);
                        }
                    }
                }
            } else {
                for (size_t c = 0; c < C; ++c) {
                    const float coeff = gamma_copy.data()[c] * std_inv[c];
                    for (size_t n = 0; n < N; ++n) {
                        const size_t base = (n * C + c) * plane;
                        for (size_t p = 0; p < plane; ++p) {
                            grad_x[base + p] = coeff * grad_output.data()[base + p];
                        }
                    }
                }
            }

            return {
                Tensor(std::move(grad_x), x_shape),
                Tensor(std::move(grad_gamma), {C}),
                Tensor(std::move(grad_beta), {C}),
            };
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor batch_norm1d(const Tensor& x, const Tensor& gamma, const Tensor& beta, std::vector<float>& running_mean,
                     std::vector<float>& running_var, bool training, float momentum, float eps) {
    if (x.shape().size() != 2) {
        throw std::runtime_error("batch_norm1d: expected a 2D (N, C) input tensor");
    }
    const size_t N = x.shape()[0];
    const size_t C = x.shape()[1];
    if (gamma.shape() != std::vector<size_t>{C} || beta.shape() != std::vector<size_t>{C}) {
        throw std::runtime_error("batch_norm1d: gamma and beta must be 1D with C entries");
    }
    if (running_mean.size() != C || running_var.size() != C) {
        throw std::runtime_error("batch_norm1d: running_mean and running_var must have C entries");
    }

    std::vector<float> mean(C);
    std::vector<float> var(C);
    if (training) {
        for (size_t c = 0; c < C; ++c) {
            float sum = 0.0f;
            for (size_t n = 0; n < N; ++n) {
                sum += x.data()[n * C + c];
            }
            mean[c] = sum / static_cast<float>(N);
        }
        for (size_t c = 0; c < C; ++c) {
            float sum_sq = 0.0f;
            for (size_t n = 0; n < N; ++n) {
                const float d = x.data()[n * C + c] - mean[c];
                sum_sq += d * d;
            }
            var[c] = sum_sq / static_cast<float>(N);
        }
        for (size_t c = 0; c < C; ++c) {
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean[c];
            running_var[c] = (1.0f - momentum) * running_var[c] + momentum * var[c];
        }
    } else {
        mean = running_mean;
        var = running_var;
    }

    std::vector<float> std_inv(C);
    for (size_t c = 0; c < C; ++c) {
        std_inv[c] = 1.0f / std::sqrt(var[c] + eps);
    }

    auto xhat = std::make_shared<std::vector<float>>(x.numel());
    std::vector<float> out_data(x.numel());
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const size_t idx = n * C + c;
            const float xh = (x.data()[idx] - mean[c]) * std_inv[c];
            (*xhat)[idx] = xh;
            out_data[idx] = gamma.data()[c] * xh + beta.data()[c];
        }
    }

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = std::move(out_data);
    out_impl->shape = x.shape();
    out_impl->requires_grad = x.requires_grad() || gamma.requires_grad() || beta.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {x.impl(), gamma.impl(), beta.impl()};
        Tensor gamma_copy = gamma;
        std::vector<size_t> x_shape = x.shape();
        node->backward_fn = [xhat, std_inv, gamma_copy, x_shape, N, C,
                              training](const Tensor& grad_output) -> std::vector<Tensor> {
            std::vector<float> grad_gamma(C, 0.0f);
            std::vector<float> grad_beta(C, 0.0f);
            for (size_t n = 0; n < N; ++n) {
                for (size_t c = 0; c < C; ++c) {
                    const size_t idx = n * C + c;
                    grad_gamma[c] += grad_output.data()[idx] * (*xhat)[idx];
                    grad_beta[c] += grad_output.data()[idx];
                }
            }

            std::vector<float> grad_x(N * C);
            if (training) {
                for (size_t c = 0; c < C; ++c) {
                    float sum_dout = 0.0f;
                    float sum_dout_xhat = 0.0f;
                    for (size_t n = 0; n < N; ++n) {
                        const size_t idx = n * C + c;
                        sum_dout += grad_output.data()[idx];
                        sum_dout_xhat += grad_output.data()[idx] * (*xhat)[idx];
                    }
                    const float coeff = gamma_copy.data()[c] * std_inv[c] / static_cast<float>(N);
                    for (size_t n = 0; n < N; ++n) {
                        const size_t idx = n * C + c;
                        grad_x[idx] =
                            coeff * (static_cast<float>(N) * grad_output.data()[idx] - sum_dout - (*xhat)[idx] * sum_dout_xhat);
                    }
                }
            } else {
                for (size_t c = 0; c < C; ++c) {
                    const float coeff = gamma_copy.data()[c] * std_inv[c];
                    for (size_t n = 0; n < N; ++n) {
                        const size_t idx = n * C + c;
                        grad_x[idx] = coeff * grad_output.data()[idx];
                    }
                }
            }

            return {
                Tensor(std::move(grad_x), x_shape),
                Tensor(std::move(grad_gamma), {C}),
                Tensor(std::move(grad_beta), {C}),
            };
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor global_avg_pool2d(const Tensor& x) {
    if (x.shape().size() != 4) {
        throw std::runtime_error("global_avg_pool2d: expected a 4D (N, C, H, W) tensor");
    }
    const size_t N = x.shape()[0];
    const size_t C = x.shape()[1];
    const size_t H = x.shape()[2];
    const size_t W = x.shape()[3];
    const size_t plane = H * W;

    std::vector<float> out_data(N * C);
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const size_t base = (n * C + c) * plane;
            float sum = 0.0f;
            for (size_t p = 0; p < plane; ++p) {
                sum += x.data()[base + p];
            }
            out_data[n * C + c] = sum / static_cast<float>(plane);
        }
    }

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = std::move(out_data);
    out_impl->shape = {N, C, 1, 1};
    out_impl->requires_grad = x.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {x.impl()};
        std::vector<size_t> x_shape = x.shape();
        node->backward_fn = [x_shape, N, C, plane](const Tensor& grad_output) -> std::vector<Tensor> {
            std::vector<float> grad_x(N * C * plane);
            for (size_t n = 0; n < N; ++n) {
                for (size_t c = 0; c < C; ++c) {
                    const float g = grad_output.data()[n * C + c] / static_cast<float>(plane);
                    const size_t base = (n * C + c) * plane;
                    for (size_t p = 0; p < plane; ++p) {
                        grad_x[base + p] = g;
                    }
                }
            }
            return {Tensor(std::move(grad_x), x_shape)};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor global_max_pool2d(const Tensor& x) {
    if (x.shape().size() != 4) {
        throw std::runtime_error("global_max_pool2d: expected a 4D (N, C, H, W) tensor");
    }
    const size_t N = x.shape()[0];
    const size_t C = x.shape()[1];
    const size_t H = x.shape()[2];
    const size_t W = x.shape()[3];
    const size_t plane = H * W;

    std::vector<float> out_data(N * C);
    auto argmax = std::make_shared<std::vector<size_t>>(N * C);
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const size_t base = (n * C + c) * plane;
            size_t best = 0;
            float best_value = x.data()[base];
            for (size_t p = 1; p < plane; ++p) {
                if (x.data()[base + p] > best_value) {
                    best_value = x.data()[base + p];
                    best = p;
                }
            }
            out_data[n * C + c] = best_value;
            (*argmax)[n * C + c] = best;
        }
    }

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = std::move(out_data);
    out_impl->shape = {N, C, 1, 1};
    out_impl->requires_grad = x.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {x.impl()};
        std::vector<size_t> x_shape = x.shape();
        node->backward_fn = [x_shape, argmax, N, C, plane](const Tensor& grad_output) -> std::vector<Tensor> {
            std::vector<float> grad_x(N * C * plane, 0.0f);
            for (size_t n = 0; n < N; ++n) {
                for (size_t c = 0; c < C; ++c) {
                    const size_t base = (n * C + c) * plane;
                    grad_x[base + (*argmax)[n * C + c]] = grad_output.data()[n * C + c];
                }
            }
            return {Tensor(std::move(grad_x), x_shape)};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor dropout(const Tensor& x, float p, bool training, std::mt19937& rng) {
    if (p < 0.0f || p >= 1.0f) {
        throw std::runtime_error("dropout: p must be in [0, 1)");
    }
    if (!training || p == 0.0f) {
        auto out_impl = std::make_shared<TensorImpl>();
        out_impl->data = x.data();
        out_impl->shape = x.shape();
        out_impl->requires_grad = x.requires_grad();
        if (out_impl->requires_grad) {
            auto node = std::make_shared<Node>();
            node->inputs = {x.impl()};
            node->backward_fn = [](const Tensor& grad_output) -> std::vector<Tensor> {
                return {Tensor(grad_output.data(), grad_output.shape())};
            };
            out_impl->grad_fn = std::move(node);
        }
        return Tensor::from_impl(std::move(out_impl));
    }

    const float keep_prob = 1.0f - p;
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    auto mask = std::make_shared<std::vector<float>>(x.numel());
    std::vector<float> out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        const float keep = uniform(rng) < keep_prob ? 1.0f / keep_prob : 0.0f;
        (*mask)[i] = keep;
        out_data[i] = x.data()[i] * keep;
    }

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = std::move(out_data);
    out_impl->shape = x.shape();
    out_impl->requires_grad = x.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {x.impl()};
        node->backward_fn = [mask](const Tensor& grad_output) -> std::vector<Tensor> {
            std::vector<float> grad_x(mask->size());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                grad_x[i] = grad_output.data()[i] * (*mask)[i];
            }
            return {Tensor(std::move(grad_x), grad_output.shape())};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

}  // namespace advanceml
