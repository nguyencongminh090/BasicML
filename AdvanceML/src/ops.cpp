#include "advanceml/ops.hpp"

#include "advanceml/detail/autograd.hpp"
#include "detail/dnnl_layout.hpp"

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

using dnnl::memory;
using detail::cpu_engine;
using detail::cpu_stream;
using detail::Layout;

// dnnl::primitive_desc construction JIT-selects and compiles a kernel, which is far more
// expensive than executing it. Training loops call the same op shape thousands of times
// (once per batch per layer per epoch), so primitives are cached per op+exact-shape rather
// than rebuilt on every forward/backward call. Keyed on real dims (not an assumed constant
// batch size) because the dataset's last batch is typically smaller than kBatchSize.
// The caches are plain function-local statics: unbounded (one entry per distinct shape and
// layout, which is a handful per model) and not thread-safe, like Tensor itself.
size_t hash_combine(size_t seed, size_t value) {
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

size_t hash_values(std::initializer_list<size_t> values) {
    size_t h = 0;
    for (size_t value : values) {
        h = hash_combine(h, std::hash<size_t>{}(value));
    }
    return h;
}

size_t hash_dims(size_t seed, const memory::dims& dims) {
    for (memory::dim d : dims) {
        seed = hash_combine(seed, std::hash<memory::dim>{}(d));
    }
    return seed;
}

template <typename Key, typename Entry, typename Hash>
std::unordered_map<Key, Entry, Hash>& cache_for() {
    static std::unordered_map<Key, Entry, Hash> cache;
    return cache;
}

memory::desc permuted_2d(size_t rows, size_t cols, bool transposed) {
    memory::desc desc = detail::plain_desc({static_cast<memory::dim>(rows), static_cast<memory::dim>(cols)});
    return transposed ? desc.permute_axes({1, 0}) : desc;
}

// Transposed operands are passed as strided views (memory::desc::permute_axes) of the same
// buffer, so backward never copies a matrix just to transpose it.
struct GemmKey {
    size_t m, k, n;
    bool a_transposed, b_transposed, has_bias;
    bool operator==(const GemmKey&) const = default;
};

struct GemmKeyHash {
    size_t operator()(const GemmKey& key) const {
        return hash_values({key.m, key.k, key.n, key.a_transposed, key.b_transposed, key.has_bias});
    }
};

struct GemmCacheEntry {
    dnnl::matmul::primitive_desc pd;
    dnnl::matmul prim;
};

/**
 * `(m, k) @ (k, n) [+ bias (n)]`. `a` is stored row-major as `(m, k)`, or as
 * `(k, m)` when `a_transposed`; likewise `b` as `(k, n)` or `(n, k)`.
 */
FloatBuffer gemm(const float* a, bool a_transposed, const float* b, bool b_transposed, const float* bias, size_t m,
                 size_t k, size_t n) {
    dnnl::engine& engine = cpu_engine();
    const memory::desc a_md = a_transposed ? permuted_2d(k, m, true) : permuted_2d(m, k, false);
    const memory::desc b_md = b_transposed ? permuted_2d(n, k, true) : permuted_2d(k, n, false);
    const memory::desc c_md = permuted_2d(m, n, false);

    const GemmKey key{m, k, n, a_transposed, b_transposed, bias != nullptr};
    auto& cache = cache_for<GemmKey, GemmCacheEntry, GemmKeyHash>();
    auto it = cache.find(key);
    if (it == cache.end()) {
        auto pd = bias != nullptr
                      ? dnnl::matmul::primitive_desc(engine, a_md, b_md, permuted_2d(1, n, false), c_md)
                      : dnnl::matmul::primitive_desc(engine, a_md, b_md, c_md);
        dnnl::matmul prim(pd);
        it = cache.emplace(key, GemmCacheEntry{std::move(pd), std::move(prim)}).first;
    }

    FloatBuffer c(m * n);
    std::unordered_map<int, memory> args = {
        {DNNL_ARG_SRC, memory(a_md, engine, const_cast<float*>(a))},
        {DNNL_ARG_WEIGHTS, memory(b_md, engine, const_cast<float*>(b))},
        {DNNL_ARG_DST, memory(c_md, engine, c.data())},
    };
    if (bias != nullptr) {
        args.emplace(DNNL_ARG_BIAS, memory(it->second.pd.bias_desc(), engine, const_cast<float*>(bias)));
    }
    it->second.prim.execute(cpu_stream(), args);
    cpu_stream().wait();
    return c;
}

/** @return A tensor over `t`'s values viewed with `dims` (same element count), in descriptor `want`. */
memory memory_with_dims(const Tensor& t, const memory::dims& dims, const memory::desc& want) {
    if (detail::to_dims(t.shape()) == dims) {
        return detail::memory_as(t, want);
    }
    const FloatBuffer& data = t.data();
    return detail::convert(memory(detail::plain_desc(dims), cpu_engine(), const_cast<float*>(data.data())), want);
}

/** @return A `format_tag::any` descriptor with `desc`'s dims, letting a backward primitive pick its own best layout. */
memory::desc any_like(const memory::desc& desc) {
    return memory::desc(desc.get_dims(), memory::data_type::f32, memory::format_tag::any);
}

/** @return Whether `m` is a oneDNN-owned conversion rather than a view over `t`'s own storage. */
bool is_converted(const memory& m, const Tensor& t) {
    return m.get_data_handle() != static_cast<void*>(t.impl()->storage->data.data());
}

struct PoolKey {
    memory::dims src_dims;
    size_t kh, kw, sh, sw;
    dnnl::algorithm algorithm;
    dnnl::prop_kind prop_kind;
    const Layout* src_layout;
    bool operator==(const PoolKey&) const = default;
};

struct PoolKeyHash {
    size_t operator()(const PoolKey& key) const {
        size_t h = hash_values({key.kh, key.kw, key.sh, key.sw, static_cast<size_t>(key.algorithm),
                                static_cast<size_t>(key.prop_kind), std::hash<const Layout*>{}(key.src_layout)});
        return hash_dims(h, key.src_dims);
    }
};

struct PoolCacheEntry {
    dnnl::pooling_forward::primitive_desc fwd_pd;
    dnnl::pooling_forward fwd_prim;
    std::shared_ptr<const Layout> dst_layout;
    std::optional<dnnl::pooling_backward::primitive_desc> bwd_pd;
    std::optional<dnnl::pooling_backward> bwd_prim;
    std::shared_ptr<const Layout> diff_src_layout;
};

// src keeps whatever layout the input arrived in and dst is `any`, so a blocked input is pooled
// by the blocked kernel and stays blocked for the next layer.
Tensor pool2d(const Tensor& x, size_t kh, size_t kw, size_t sh, size_t sw, dnnl::algorithm algorithm,
              const char* op_name) {
    if (x.shape().size() != 4) {
        throw std::runtime_error(std::string(op_name) + ": expected a 4D (N, C, H, W) tensor");
    }
    const size_t N = x.shape()[0];
    const size_t C = x.shape()[1];
    const size_t H = x.shape()[2];
    const size_t W = x.shape()[3];
    if (kh > H || kw > W) {
        throw std::runtime_error(std::string(op_name) + ": kernel_size exceeds H or W");
    }
    const size_t h_out = (H - kh) / sh + 1;
    const size_t w_out = (W - kw) / sw + 1;

    dnnl::engine& engine = cpu_engine();
    const memory::desc src_md = detail::desc_of(x);
    const memory::dims dst_dims = {static_cast<memory::dim>(N), static_cast<memory::dim>(C),
                                   static_cast<memory::dim>(h_out), static_cast<memory::dim>(w_out)};
    const memory::dims strides_dims = {static_cast<memory::dim>(sh), static_cast<memory::dim>(sw)};
    const memory::dims kernel_dims = {static_cast<memory::dim>(kh), static_cast<memory::dim>(kw)};
    const memory::dims dilation_dims = {0, 0};
    const memory::dims padding_dims = {0, 0};

    // Inference primitives skip the max-pool argmax workspace that only backward needs.
    const bool record_graph = detail::should_record({&x});
    const dnnl::prop_kind prop_kind =
        record_graph ? dnnl::prop_kind::forward_training : dnnl::prop_kind::forward_inference;

    PoolKey key{src_md.get_dims(), kh, kw, sh, sw, algorithm, prop_kind, x.impl()->storage->layout.get()};
    auto& cache = cache_for<PoolKey, PoolCacheEntry, PoolKeyHash>();
    auto cache_it = cache.find(key);
    if (cache_it == cache.end()) {
        auto dst_any = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::any);
        auto fwd_pd = dnnl::pooling_forward::primitive_desc(engine, prop_kind, algorithm, src_md, dst_any,
                                                             strides_dims, kernel_dims, dilation_dims, padding_dims,
                                                             padding_dims);
        dnnl::pooling_forward fwd_prim(fwd_pd);
        auto dst_layout = detail::storage_layout(fwd_pd.dst_desc());
        cache_it = cache.emplace(key, PoolCacheEntry{std::move(fwd_pd), std::move(fwd_prim), std::move(dst_layout),
                                                      std::nullopt, std::nullopt, nullptr}).first;
    }
    PoolCacheEntry& entry = cache_it->second;
    const dnnl::pooling_forward::primitive_desc& fwd_pd = entry.fwd_pd;

    FloatBuffer out_data = detail::buffer_for(fwd_pd.dst_desc());
    std::unordered_map<int, memory> fwd_args = {
        {DNNL_ARG_SRC, memory(src_md, engine, x.impl()->storage->data.data())},
        {DNNL_ARG_DST, memory(fwd_pd.dst_desc(), engine, out_data.data())},
    };
    const bool needs_workspace = record_graph && algorithm == dnnl::algorithm::pooling_max;
    std::optional<memory> workspace;
    if (needs_workspace) {
        workspace = memory(fwd_pd.workspace_desc(), engine);
        fwd_args.emplace(DNNL_ARG_WORKSPACE, *workspace);
    }
    entry.fwd_prim.execute(cpu_stream(), fwd_args);
    cpu_stream().wait();

    Tensor out = detail::make_tensor(std::move(out_data), {N, C, h_out, w_out}, entry.dst_layout);
    if (record_graph) {
        detail::record(out, {&x}, {}, [key, strides_dims, kernel_dims, dilation_dims, padding_dims,
                                        x_shape = x.shape(), workspace](const Tensor& grad_output) -> InputGrads {
            dnnl::engine& engine = cpu_engine();
            PoolCacheEntry& entry = cache_for<PoolKey, PoolCacheEntry, PoolKeyHash>().at(key);
            if (!entry.bwd_pd.has_value()) {
                entry.bwd_pd = dnnl::pooling_backward::primitive_desc(
                    engine, key.algorithm, entry.fwd_pd.src_desc(), entry.fwd_pd.dst_desc(), strides_dims,
                    kernel_dims, dilation_dims, padding_dims, padding_dims, entry.fwd_pd);
                entry.bwd_prim = dnnl::pooling_backward(*entry.bwd_pd);
                entry.diff_src_layout = detail::storage_layout(entry.bwd_pd->diff_src_desc());
            }
            const dnnl::pooling_backward::primitive_desc& bwd_pd = *entry.bwd_pd;

            memory diff_dst = detail::memory_as(grad_output, bwd_pd.diff_dst_desc());
            FloatBuffer grad_x_data = detail::buffer_for(bwd_pd.diff_src_desc());
            std::unordered_map<int, memory> bwd_args = {
                {DNNL_ARG_DIFF_DST, diff_dst},
                {DNNL_ARG_DIFF_SRC, memory(bwd_pd.diff_src_desc(), engine, grad_x_data.data())},
            };
            if (workspace.has_value()) {
                bwd_args.emplace(DNNL_ARG_WORKSPACE, *workspace);
            }
            entry.bwd_prim->execute(cpu_stream(), bwd_args);
            cpu_stream().wait();

            return {detail::make_tensor(std::move(grad_x_data), x_shape, entry.diff_src_layout)};
        });
    }
    return out;
}

struct ConvKey {
    size_t N, c_in, H, W, c_out, kh, kw, stride, padding;
    dnnl::prop_kind prop_kind;
    bool operator==(const ConvKey&) const = default;
};

struct ConvKeyHash {
    size_t operator()(const ConvKey& key) const {
        return hash_values({key.N, key.c_in, key.H, key.W, key.c_out, key.kh, key.kw, key.stride, key.padding,
                            static_cast<size_t>(key.prop_kind)});
    }
};

struct ConvCacheEntry {
    dnnl::convolution_forward::primitive_desc fwd_pd;
    dnnl::convolution_forward fwd_prim;
    std::shared_ptr<const Layout> dst_layout;
    std::optional<dnnl::convolution_backward_data::primitive_desc> bwd_data_pd;
    std::optional<dnnl::convolution_backward_data> bwd_data_prim;
    std::shared_ptr<const Layout> diff_src_layout;
    std::optional<dnnl::convolution_backward_weights::primitive_desc> bwd_weights_pd;
    std::optional<dnnl::convolution_backward_weights> bwd_weights_prim;
    // Blocked weight gradient, reused every step before it is reordered into the plain .grad buffer.
    std::optional<memory> diff_weights_scratch;
};

struct BatchNormKey {
    memory::dims dims;
    const Layout* src_layout;
    dnnl::prop_kind prop_kind;
    bool global_stats;
    float eps;
    bool operator==(const BatchNormKey&) const = default;
};

struct BatchNormKeyHash {
    size_t operator()(const BatchNormKey& key) const {
        size_t h = hash_values({std::hash<const Layout*>{}(key.src_layout), static_cast<size_t>(key.prop_kind),
                                key.global_stats, std::hash<float>{}(key.eps)});
        return hash_dims(h, key.dims);
    }
};

struct BatchNormCacheEntry {
    dnnl::batch_normalization_forward::primitive_desc fwd_pd;
    dnnl::batch_normalization_forward fwd_prim;
    std::optional<dnnl::batch_normalization_backward::primitive_desc> bwd_pd;
    std::optional<dnnl::batch_normalization_backward> bwd_prim;
    std::shared_ptr<const Layout> diff_src_layout;
};

/**
 * Shared batch normalization over oneDNN's primitive. `dims` is the 4D
 * `(N, C, H, W)` view of `x` (`(N, C, 1, 1)` for `batch_norm1d`); a 4D input
 * keeps its (possibly blocked) layout end to end.
 */
Tensor batch_norm(const Tensor& x, const memory::dims& dims, const Tensor& gamma, const Tensor& beta,
                  std::vector<float>& running_mean, std::vector<float>& running_var, bool training, float momentum,
                  float eps) {
    const size_t C = static_cast<size_t>(dims[1]);
    dnnl::engine& engine = cpu_engine();

    const bool same_dims = detail::to_dims(x.shape()) == dims;
    const memory::desc src_md = same_dims ? detail::desc_of(x) : detail::plain_desc(dims);
    const float* const src_ptr = same_dims ? x.impl()->storage->data.data() : x.data().data();
    const std::shared_ptr<const Layout> src_layout = same_dims ? x.impl()->storage->layout : nullptr;

    const bool record_graph = detail::should_record({&x, &gamma, &beta});
    // Batch statistics are only produced by forward_training, so training mode uses it even
    // when nothing is recorded (the running statistics still have to be updated).
    const dnnl::prop_kind prop_kind = record_graph || training ? dnnl::prop_kind::forward_training
                                                               : dnnl::prop_kind::forward_inference;
    dnnl::normalization_flags flags = dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift;
    if (!training) {
        flags = flags | dnnl::normalization_flags::use_global_stats;
    }

    BatchNormKey key{dims, src_layout.get(), prop_kind, !training, eps};
    auto& cache = cache_for<BatchNormKey, BatchNormCacheEntry, BatchNormKeyHash>();
    auto cache_it = cache.find(key);
    if (cache_it == cache.end()) {
        auto fwd_pd = dnnl::batch_normalization_forward::primitive_desc(engine, prop_kind, src_md, src_md, eps, flags);
        dnnl::batch_normalization_forward fwd_prim(fwd_pd);
        cache_it = cache.emplace(key, BatchNormCacheEntry{std::move(fwd_pd), std::move(fwd_prim), std::nullopt,
                                                           std::nullopt, nullptr}).first;
    }
    BatchNormCacheEntry& entry = cache_it->second;

    const memory::desc stats_md = detail::plain_desc({static_cast<memory::dim>(C)});
    auto mean = std::make_shared<FloatBuffer>(C);
    auto var = std::make_shared<FloatBuffer>(C);
    if (!training) {
        std::copy(running_mean.begin(), running_mean.end(), mean->begin());
        std::copy(running_var.begin(), running_var.end(), var->begin());
    }

    FloatBuffer out_data = detail::buffer_for(src_md);
    entry.fwd_prim.execute(cpu_stream(), {
        {DNNL_ARG_SRC, memory(src_md, engine, const_cast<float*>(src_ptr))},
        {DNNL_ARG_DST, memory(src_md, engine, out_data.data())},
        {DNNL_ARG_SCALE, memory(stats_md, engine, const_cast<float*>(gamma.data().data()))},
        {DNNL_ARG_SHIFT, memory(stats_md, engine, const_cast<float*>(beta.data().data()))},
        {DNNL_ARG_MEAN, memory(stats_md, engine, mean->data())},
        {DNNL_ARG_VARIANCE, memory(stats_md, engine, var->data())},
    });
    cpu_stream().wait();

    if (training) {
        for (size_t c = 0; c < C; ++c) {
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * (*mean)[c];
            running_var[c] = (1.0f - momentum) * running_var[c] + momentum * (*var)[c];
        }
    }

    Tensor out = detail::make_tensor(std::move(out_data), x.shape(), src_layout);
    if (!record_graph) {
        return out;
    }
    detail::record(out, {&x, &gamma, &beta}, {&x, &gamma},
                   [key, x, gamma, mean, var, flags](const Tensor& grad_output,
                                                     const NeedsInputGrad& needs) -> InputGrads {
        dnnl::engine& engine = cpu_engine();
        BatchNormCacheEntry& entry = cache_for<BatchNormKey, BatchNormCacheEntry, BatchNormKeyHash>().at(key);
        const memory::desc& fwd_src_md = entry.fwd_pd.src_desc();
        if (!entry.bwd_pd.has_value()) {
            entry.bwd_pd = dnnl::batch_normalization_backward::primitive_desc(
                engine, dnnl::prop_kind::backward, fwd_src_md, fwd_src_md, fwd_src_md, key.eps, flags, entry.fwd_pd);
            entry.bwd_prim = dnnl::batch_normalization_backward(*entry.bwd_pd);
            entry.diff_src_layout = detail::storage_layout(entry.bwd_pd->diff_src_desc());
        }
        const dnnl::batch_normalization_backward::primitive_desc& bwd_pd = *entry.bwd_pd;
        const size_t C = static_cast<size_t>(key.dims[1]);
        const memory::desc stats_md = detail::plain_desc({static_cast<memory::dim>(C)});

        memory src = memory_with_dims(x, key.dims, bwd_pd.src_desc());
        memory diff_dst = memory_with_dims(grad_output, key.dims, bwd_pd.diff_dst_desc());
        FloatBuffer grad_x_data = detail::buffer_for(bwd_pd.diff_src_desc());
        FloatBuffer grad_gamma(C);
        FloatBuffer grad_beta(C);
        entry.bwd_prim->execute(cpu_stream(), {
            {DNNL_ARG_SRC, src},
            {DNNL_ARG_DIFF_DST, diff_dst},
            {DNNL_ARG_SCALE, memory(stats_md, engine, const_cast<float*>(gamma.data().data()))},
            {DNNL_ARG_MEAN, memory(stats_md, engine, mean->data())},
            {DNNL_ARG_VARIANCE, memory(stats_md, engine, var->data())},
            {DNNL_ARG_DIFF_SRC, memory(bwd_pd.diff_src_desc(), engine, grad_x_data.data())},
            {DNNL_ARG_DIFF_SCALE, memory(stats_md, engine, grad_gamma.data())},
            {DNNL_ARG_DIFF_SHIFT, memory(stats_md, engine, grad_beta.data())},
        });
        cpu_stream().wait();

        InputGrads grads;
        if (needs[0]) {
            grads[0] = detail::make_tensor(std::move(grad_x_data), x.shape(), entry.diff_src_layout);
        }
        if (needs[1]) {
            grads[1] = Tensor(std::move(grad_gamma), {C});
        }
        if (needs[2]) {
            grads[2] = Tensor(std::move(grad_beta), {C});
        }
        return grads;
    });
    return out;
}

struct EltwiseKey {
    memory::dims dims;
    const Layout* layout;
    dnnl::prop_kind prop_kind;
    bool operator==(const EltwiseKey&) const = default;
};

struct EltwiseKeyHash {
    size_t operator()(const EltwiseKey& key) const {
        return hash_dims(hash_values({std::hash<const Layout*>{}(key.layout), static_cast<size_t>(key.prop_kind)}),
                         key.dims);
    }
};

struct EltwiseForwardEntry {
    dnnl::eltwise_forward::primitive_desc pd;
    dnnl::eltwise_forward prim;
};

struct EltwiseBackwardEntry {
    dnnl::eltwise_backward::primitive_desc pd;
    dnnl::eltwise_backward prim;
    std::shared_ptr<const Layout> diff_src_layout;
};

}  // namespace

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape().size() != 2 || b.shape().size() != 2 || a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error("matmul: expected 2D tensors with compatible inner dimensions");
    }
    const size_t m = a.shape()[0];
    const size_t k = a.shape()[1];
    const size_t n = b.shape()[1];

    Tensor out(gemm(a.data().data(), false, b.data().data(), false, nullptr, m, k, n), {m, n});
    if (detail::should_record({&a, &b})) {
        detail::record(out, {&a, &b}, {&a, &b},
                       [a, b, m, k, n](const Tensor& grad_output, const NeedsInputGrad& needs) -> InputGrads {
                           const float* g = grad_output.data().data();
                           InputGrads grads;
                           if (needs[0]) {
                               grads[0] = Tensor(gemm(g, false, b.data().data(), true, nullptr, m, n, k), {m, k});
                           }
                           if (needs[1]) {
                               grads[1] = Tensor(gemm(a.data().data(), true, g, false, nullptr, k, m, n), {k, n});
                           }
                           return grads;
                       });
    }
    return out;
}

Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    if (x.shape().size() != 2 || weight.shape().size() != 2 || x.shape()[1] != weight.shape()[0]) {
        throw std::runtime_error("linear: expected a 2D x and weight with compatible inner dimensions");
    }
    if (bias.shape().size() != 1 || bias.shape()[0] != weight.shape()[1]) {
        throw std::runtime_error("linear: bias must be 1D with weight.shape()[1] entries");
    }
    const size_t m = x.shape()[0];
    const size_t k = x.shape()[1];
    const size_t n = weight.shape()[1];

    Tensor out(gemm(x.data().data(), false, weight.data().data(), false, bias.data().data(), m, k, n), {m, n});
    if (detail::should_record({&x, &weight, &bias})) {
        detail::record(out, {&x, &weight, &bias}, {&x, &weight},
                       [x, weight, m, k, n](const Tensor& grad_output, const NeedsInputGrad& needs) -> InputGrads {
                           const FloatBuffer& g = grad_output.data();
                           InputGrads grads;
                           if (needs[0]) {
                               grads[0] =
                                   Tensor(gemm(g.data(), false, weight.data().data(), true, nullptr, m, n, k), {m, k});
                           }
                           if (needs[1]) {
                               grads[1] =
                                   Tensor(gemm(x.data().data(), true, g.data(), false, nullptr, k, m, n), {k, n});
                           }
                           if (needs[2]) {
                               FloatBuffer grad_b(n, 0.0f);
                               for (size_t r = 0; r < m; ++r) {
                                   const float* row = g.data() + r * n;
                                   for (size_t c = 0; c < n; ++c) {
                                       grad_b[c] += row[c];
                                   }
                               }
                               grads[2] = Tensor(std::move(grad_b), {n});
                           }
                           return grads;
                       });
    }
    return out;
}

// oneDNN eltwise keeps the input's (possibly blocked) layout, and the *_use_dst_for_bwd variant
// lets backward read the saved output instead of keeping the input alive as well.
Tensor relu(const Tensor& x) {
    dnnl::engine& engine = cpu_engine();
    const memory::desc src_md = detail::desc_of(x);
    const bool record_graph = detail::should_record({&x});
    const dnnl::prop_kind prop_kind =
        record_graph ? dnnl::prop_kind::forward_training : dnnl::prop_kind::forward_inference;

    const EltwiseKey key{src_md.get_dims(), x.impl()->storage->layout.get(), prop_kind};
    auto& cache = cache_for<EltwiseKey, EltwiseForwardEntry, EltwiseKeyHash>();
    auto it = cache.find(key);
    if (it == cache.end()) {
        auto pd = dnnl::eltwise_forward::primitive_desc(engine, prop_kind, dnnl::algorithm::eltwise_relu_use_dst_for_bwd,
                                                        src_md, src_md, 0.0f);
        dnnl::eltwise_forward prim(pd);
        it = cache.emplace(key, EltwiseForwardEntry{std::move(pd), std::move(prim)}).first;
    }

    FloatBuffer out_data = detail::buffer_for(src_md);
    it->second.prim.execute(cpu_stream(), {
        {DNNL_ARG_SRC, memory(src_md, engine, x.impl()->storage->data.data())},
        {DNNL_ARG_DST, memory(src_md, engine, out_data.data())},
    });
    cpu_stream().wait();

    Tensor out = detail::make_tensor(std::move(out_data), x.shape(), x.impl()->storage->layout);
    if (record_graph) {
        detail::record(out, {&x}, {&out}, [y = out.impl()->storage, dims = src_md.get_dims()](
                                               const Tensor& grad_output) -> InputGrads {
            dnnl::engine& engine = cpu_engine();
            const memory::desc data_md = detail::desc_of(*y, dims);
            const EltwiseKey bwd_key{dims, detail::layout_id(data_md), dnnl::prop_kind::backward};
            auto& bwd_cache = cache_for<EltwiseKey, EltwiseBackwardEntry, EltwiseKeyHash>();
            auto bwd_it = bwd_cache.find(bwd_key);
            if (bwd_it == bwd_cache.end()) {
                auto hint = dnnl::eltwise_forward::primitive_desc(engine, dnnl::prop_kind::forward_training,
                                                                  dnnl::algorithm::eltwise_relu_use_dst_for_bwd,
                                                                  data_md, data_md, 0.0f);
                auto pd = dnnl::eltwise_backward::primitive_desc(engine, dnnl::algorithm::eltwise_relu_use_dst_for_bwd,
                                                                 data_md, data_md, data_md, 0.0f, hint);
                dnnl::eltwise_backward prim(pd);
                auto diff_src_layout = detail::storage_layout(pd.diff_src_desc());
                bwd_it = bwd_cache.emplace(bwd_key, EltwiseBackwardEntry{std::move(pd), std::move(prim),
                                                                          std::move(diff_src_layout)}).first;
            }
            const EltwiseBackwardEntry& entry = bwd_it->second;

            memory diff_dst = detail::memory_as(grad_output, entry.pd.diff_dst_desc());
            FloatBuffer grad_x_data = detail::buffer_for(entry.pd.diff_src_desc());
            entry.prim.execute(cpu_stream(), {
                {DNNL_ARG_DST, memory(data_md, engine, y->data.data())},
                {DNNL_ARG_DIFF_DST, diff_dst},
                {DNNL_ARG_DIFF_SRC, memory(entry.pd.diff_src_desc(), engine, grad_x_data.data())},
            });
            cpu_stream().wait();
            return {detail::make_tensor(std::move(grad_x_data), grad_output.shape(), entry.diff_src_layout)};
        });
    }
    return out;
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

    Tensor out({loss_value}, {1});
    if (detail::should_record({&pred, &target})) {
        detail::record(out, {&pred, &target}, {&pred, &target},
                       [pred, target, n](const Tensor& grad_output, const NeedsInputGrad& needs) -> InputGrads {
                           const float upstream = grad_output.data()[0];
                           FloatBuffer grad_pred(needs[0] ? n : 0);
                           FloatBuffer grad_target(needs[1] ? n : 0);
                           for (size_t i = 0; i < n; ++i) {
                               const float diff = pred.data()[i] - target.data()[i];
                               const float g = upstream * 2.0f * diff / static_cast<float>(n);
                               if (needs[0]) {
                                   grad_pred[i] = g;
                               }
                               if (needs[1]) {
                                   grad_target[i] = -g;
                               }
                           }
                           InputGrads grads;
                           if (needs[0]) {
                               grads[0] = Tensor(std::move(grad_pred), pred.shape());
                           }
                           if (needs[1]) {
                               grads[1] = Tensor(std::move(grad_target), target.shape());
                           }
                           return grads;
                       });
    }
    return out;
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

    Tensor out({loss_value}, {1});
    if (detail::should_record({&pred, &target})) {
        detail::record(out, {&pred, &target}, {&pred, &target},
                       [pred, target, n](const Tensor& grad_output, const NeedsInputGrad& needs) -> InputGrads {
                           const float upstream = grad_output.data()[0];
                           FloatBuffer grad_pred(needs[0] ? n : 0);
                           FloatBuffer grad_target(needs[1] ? n : 0);
                           for (size_t i = 0; i < n; ++i) {
                               const float diff = pred.data()[i] - target.data()[i];
                               const float sign = diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f);
                               const float g = upstream * sign / static_cast<float>(n);
                               if (needs[0]) {
                                   grad_pred[i] = g;
                               }
                               if (needs[1]) {
                                   grad_target[i] = -g;
                               }
                           }
                           InputGrads grads;
                           if (needs[0]) {
                               grads[0] = Tensor(std::move(grad_pred), pred.shape());
                           }
                           if (needs[1]) {
                               grads[1] = Tensor(std::move(grad_target), target.shape());
                           }
                           return grads;
                       });
    }
    return out;
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

    Tensor out({loss_value}, {1});
    if (detail::should_record({&pred, &target})) {
        detail::record(out, {&pred, &target}, {&pred, &target},
                       [pred, target, n, numel](const Tensor& grad_output, const NeedsInputGrad& needs) -> InputGrads {
                           constexpr float kEpsilon = 1e-15f;
                           const float upstream = grad_output.data()[0];
                           FloatBuffer grad_pred(needs[0] ? numel : 0);
                           FloatBuffer grad_target(needs[1] ? numel : 0);
                           for (size_t i = 0; i < numel; ++i) {
                               const float clipped = std::clamp(pred.data()[i], kEpsilon, 1.0f - kEpsilon);
                               const float t = target.data()[i];
                               if (needs[0]) {
                                   grad_pred[i] = -upstream * (t / clipped - (1.0f - t) / (1.0f - clipped)) /
                                                  static_cast<float>(n);
                               }
                               if (needs[1]) {
                                   grad_target[i] =
                                       -upstream * (std::log(clipped) - std::log(1.0f - clipped)) / static_cast<float>(n);
                               }
                           }
                           InputGrads grads;
                           if (needs[0]) {
                               grads[0] = Tensor(std::move(grad_pred), pred.shape());
                           }
                           if (needs[1]) {
                               grads[1] = Tensor(std::move(grad_target), target.shape());
                           }
                           return grads;
                       });
    }
    return out;
}

Tensor sigmoid(const Tensor& x) {
    const FloatBuffer& x_data = x.data();
    FloatBuffer out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        out_data[i] = 1.0f / (1.0f + std::exp(-x_data[i]));
    }

    Tensor out(std::move(out_data), x.shape());
    if (detail::should_record({&x})) {
        detail::record(out, {&x}, {&out}, [y = out.impl()->storage](const Tensor& grad_output) -> InputGrads {
            const FloatBuffer& y_data = y->data;
            const FloatBuffer& g = grad_output.data();
            FloatBuffer grad_x(y_data.size());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                grad_x[i] = g[i] * y_data[i] * (1.0f - y_data[i]);
            }
            return {Tensor(std::move(grad_x), grad_output.shape())};
        });
    }
    return out;
}

Tensor leaky_relu(const Tensor& x, float negative_slope) {
    const FloatBuffer& x_data = x.data();
    FloatBuffer out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        const float v = x_data[i];
        out_data[i] = v > 0.0f ? v : negative_slope * v;
    }

    Tensor out(std::move(out_data), x.shape());
    if (detail::should_record({&x})) {
        detail::record(out, {&x}, {&x}, [x, negative_slope](const Tensor& grad_output) -> InputGrads {
            const FloatBuffer& x_data = x.data();
            const FloatBuffer& g = grad_output.data();
            FloatBuffer grad_x(x.numel());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                grad_x[i] = x_data[i] > 0.0f ? g[i] : negative_slope * g[i];
            }
            return {Tensor(std::move(grad_x), x.shape())};
        });
    }
    return out;
}

Tensor gelu(const Tensor& x) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    constexpr float kInvSqrt2Pi = 0.39894228040143267794f;

    const FloatBuffer& x_data = x.data();
    FloatBuffer out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        const float v = x_data[i];
        out_data[i] = v * 0.5f * (1.0f + std::erf(v * kInvSqrt2));
    }

    Tensor out(std::move(out_data), x.shape());
    if (detail::should_record({&x})) {
        detail::record(out, {&x}, {&x}, [x](const Tensor& grad_output) -> InputGrads {
            const FloatBuffer& x_data = x.data();
            const FloatBuffer& g = grad_output.data();
            FloatBuffer grad_x(x.numel());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                const float v = x_data[i];
                const float cdf = 0.5f * (1.0f + std::erf(v * kInvSqrt2));
                const float pdf = kInvSqrt2Pi * std::exp(-0.5f * v * v);
                grad_x[i] = g[i] * (cdf + v * pdf);
            }
            return {Tensor(std::move(grad_x), x.shape())};
        });
    }
    return out;
}

Tensor tanh(const Tensor& x) {
    const FloatBuffer& x_data = x.data();
    FloatBuffer out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        out_data[i] = std::tanh(x_data[i]);
    }

    Tensor out(std::move(out_data), x.shape());
    if (detail::should_record({&x})) {
        detail::record(out, {&x}, {&out}, [y = out.impl()->storage](const Tensor& grad_output) -> InputGrads {
            const FloatBuffer& y_data = y->data;
            const FloatBuffer& g = grad_output.data();
            FloatBuffer grad_x(y_data.size());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                grad_x[i] = g[i] * (1.0f - y_data[i] * y_data[i]);
            }
            return {Tensor(std::move(grad_x), grad_output.shape())};
        });
    }
    return out;
}

Tensor identity(const Tensor& x) {
    Tensor out = detail::view(x, x.shape());
    if (detail::should_record({&x})) {
        detail::record(out, {&x}, {}, [](const Tensor& grad_output) -> InputGrads {
            return {grad_output};
        });
    }
    return out;
}

Tensor prelu(const Tensor& x, const Tensor& a) {
    const size_t num_parameters = a.numel();
    if (num_parameters != 1 && (x.shape().empty() || x.shape().back() != num_parameters)) {
        throw std::runtime_error("prelu: a.numel() must be 1 or equal to x.shape().back()");
    }

    const FloatBuffer& x_data = x.data();
    const FloatBuffer& a_data = a.data();
    FloatBuffer out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        const size_t c = num_parameters == 1 ? 0 : i % num_parameters;
        const float v = x_data[i];
        out_data[i] = v > 0.0f ? v : a_data[c] * v;
    }

    Tensor out(std::move(out_data), x.shape());
    if (detail::should_record({&x, &a})) {
        detail::record(out, {&x, &a}, {&x, &a},
                       [x, a, num_parameters](const Tensor& grad_output, const NeedsInputGrad& needs) -> InputGrads {
                           const FloatBuffer& x_data = x.data();
                           const FloatBuffer& a_data = a.data();
                           const FloatBuffer& g = grad_output.data();
                           FloatBuffer grad_x(needs[0] ? x.numel() : 0);
                           FloatBuffer grad_a(num_parameters, 0.0f);
                           for (size_t i = 0; i < x_data.size(); ++i) {
                               const size_t c = num_parameters == 1 ? 0 : i % num_parameters;
                               const float v = x_data[i];
                               if (v > 0.0f) {
                                   if (needs[0]) {
                                       grad_x[i] = g[i];
                                   }
                               } else {
                                   if (needs[0]) {
                                       grad_x[i] = a_data[c] * g[i];
                                   }
                                   grad_a[c] += g[i] * v;
                               }
                           }
                           InputGrads grads;
                           if (needs[0]) {
                               grads[0] = Tensor(std::move(grad_x), x.shape());
                           }
                           if (needs[1]) {
                               grads[1] = Tensor(std::move(grad_a), {num_parameters});
                           }
                           return grads;
                       });
    }
    return out;
}

Tensor elu(const Tensor& x, float alpha) {
    const FloatBuffer& x_data = x.data();
    FloatBuffer out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        const float v = x_data[i];
        out_data[i] = v > 0.0f ? v : alpha * std::expm1(v);
    }

    Tensor out(std::move(out_data), x.shape());
    if (detail::should_record({&x})) {
        detail::record(out, {&x}, {&x}, [x, alpha](const Tensor& grad_output) -> InputGrads {
            const FloatBuffer& x_data = x.data();
            const FloatBuffer& g = grad_output.data();
            FloatBuffer grad_x(x.numel());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                const float v = x_data[i];
                const float local = v > 0.0f ? 1.0f : alpha * std::exp(v);
                grad_x[i] = g[i] * local;
            }
            return {Tensor(std::move(grad_x), x.shape())};
        });
    }
    return out;
}

Tensor selu(const Tensor& x) {
    constexpr float kAlpha = 1.6732632423543772f;
    constexpr float kScale = 1.0507009873554805f;

    const FloatBuffer& x_data = x.data();
    FloatBuffer out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        const float v = x_data[i];
        out_data[i] = kScale * (v > 0.0f ? v : kAlpha * std::expm1(v));
    }

    Tensor out(std::move(out_data), x.shape());
    if (detail::should_record({&x})) {
        detail::record(out, {&x}, {&x}, [x](const Tensor& grad_output) -> InputGrads {
            const FloatBuffer& x_data = x.data();
            const FloatBuffer& g = grad_output.data();
            FloatBuffer grad_x(x.numel());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                const float v = x_data[i];
                const float local = kScale * (v > 0.0f ? 1.0f : kAlpha * std::exp(v));
                grad_x[i] = g[i] * local;
            }
            return {Tensor(std::move(grad_x), x.shape())};
        });
    }
    return out;
}

Tensor softplus(const Tensor& x) {
    const FloatBuffer& x_data = x.data();
    FloatBuffer out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        const float v = x_data[i];
        out_data[i] = std::max(v, 0.0f) + std::log1p(std::exp(-std::fabs(v)));
    }

    Tensor out(std::move(out_data), x.shape());
    if (detail::should_record({&x})) {
        detail::record(out, {&x}, {&x}, [x](const Tensor& grad_output) -> InputGrads {
            const FloatBuffer& x_data = x.data();
            const FloatBuffer& g = grad_output.data();
            FloatBuffer grad_x(x.numel());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                const float sig = 1.0f / (1.0f + std::exp(-x_data[i]));
                grad_x[i] = g[i] * sig;
            }
            return {Tensor(std::move(grad_x), x.shape())};
        });
    }
    return out;
}

Tensor swish(const Tensor& x, float beta) {
    const bool record_graph = detail::should_record({&x});
    const FloatBuffer& x_data = x.data();
    FloatBuffer out_data(x.numel());
    auto sig = std::make_shared<FloatBuffer>(record_graph ? x.numel() : 0);
    for (size_t i = 0; i < out_data.size(); ++i) {
        const float v = x_data[i];
        const float s = 1.0f / (1.0f + std::exp(-beta * v));
        if (record_graph) {
            (*sig)[i] = s;
        }
        out_data[i] = v * s;
    }

    Tensor out(std::move(out_data), x.shape());
    if (record_graph) {
        detail::record(out, {&x}, {&x}, [x, sig, beta](const Tensor& grad_output) -> InputGrads {
            const FloatBuffer& x_data = x.data();
            const FloatBuffer& g = grad_output.data();
            FloatBuffer grad_x(x.numel());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                const float s = (*sig)[i];
                const float local = s + beta * x_data[i] * s * (1.0f - s);
                grad_x[i] = g[i] * local;
            }
            return {Tensor(std::move(grad_x), x.shape())};
        });
    }
    return out;
}

Tensor mish(const Tensor& x) {
    const FloatBuffer& x_data = x.data();
    FloatBuffer out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        const float v = x_data[i];
        const float sp = std::max(v, 0.0f) + std::log1p(std::exp(-std::fabs(v)));
        out_data[i] = v * std::tanh(sp);
    }

    Tensor out(std::move(out_data), x.shape());
    if (detail::should_record({&x})) {
        detail::record(out, {&x}, {&x}, [x](const Tensor& grad_output) -> InputGrads {
            const FloatBuffer& x_data = x.data();
            const FloatBuffer& g = grad_output.data();
            FloatBuffer grad_x(x.numel());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                const float v = x_data[i];
                const float sp = std::max(v, 0.0f) + std::log1p(std::exp(-std::fabs(v)));
                const float t = std::tanh(sp);
                const float sig = 1.0f / (1.0f + std::exp(-v));
                const float local = t + v * (1.0f - t * t) * sig;
                grad_x[i] = g[i] * local;
            }
            return {Tensor(std::move(grad_x), x.shape())};
        });
    }
    return out;
}

Tensor hardtanh(const Tensor& x, float min_val, float max_val) {
    if (max_val <= min_val) {
        throw std::runtime_error("hardtanh: max_val must be greater than min_val");
    }

    const FloatBuffer& x_data = x.data();
    FloatBuffer out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        out_data[i] = std::clamp(x_data[i], min_val, max_val);
    }

    Tensor out(std::move(out_data), x.shape());
    if (detail::should_record({&x})) {
        detail::record(out, {&x}, {&x}, [x, min_val, max_val](const Tensor& grad_output) -> InputGrads {
            const FloatBuffer& x_data = x.data();
            const FloatBuffer& g = grad_output.data();
            FloatBuffer grad_x(x.numel());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                const float v = x_data[i];
                grad_x[i] = (v > min_val && v < max_val) ? g[i] : 0.0f;
            }
            return {Tensor(std::move(grad_x), x.shape())};
        });
    }
    return out;
}

Tensor hardsigmoid(const Tensor& x) {
    const FloatBuffer& x_data = x.data();
    FloatBuffer out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        out_data[i] = std::clamp(x_data[i] / 6.0f + 0.5f, 0.0f, 1.0f);
    }

    Tensor out(std::move(out_data), x.shape());
    if (detail::should_record({&x})) {
        detail::record(out, {&x}, {&x}, [x](const Tensor& grad_output) -> InputGrads {
            const FloatBuffer& x_data = x.data();
            const FloatBuffer& g = grad_output.data();
            FloatBuffer grad_x(x.numel());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                const float v = x_data[i];
                grad_x[i] = (v > -3.0f && v < 3.0f) ? g[i] / 6.0f : 0.0f;
            }
            return {Tensor(std::move(grad_x), x.shape())};
        });
    }
    return out;
}

Tensor softmax(const Tensor& x) {
    if (x.shape().size() != 1 && x.shape().size() != 2) {
        throw std::runtime_error("softmax: expected a 1D or 2D tensor");
    }
    const size_t rows = x.shape().size() == 2 ? x.shape()[0] : 1;
    const size_t cols = x.shape().size() == 2 ? x.shape()[1] : x.shape()[0];

    const FloatBuffer& x_data = x.data();
    FloatBuffer out_data(x.numel());
    for (size_t r = 0; r < rows; ++r) {
        const size_t base = r * cols;
        float row_max = x_data[base];
        for (size_t c = 1; c < cols; ++c) {
            row_max = std::max(row_max, x_data[base + c]);
        }
        float sum_exp = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            const float e = std::exp(x_data[base + c] - row_max);
            out_data[base + c] = e;
            sum_exp += e;
        }
        for (size_t c = 0; c < cols; ++c) {
            out_data[base + c] /= sum_exp;
        }
    }

    Tensor out(std::move(out_data), x.shape());
    if (detail::should_record({&x})) {
        detail::record(out, {&x}, {&out}, [y = out.impl()->storage, rows, cols](const Tensor& grad_output) -> InputGrads {
            const FloatBuffer& y_data = y->data;
            const FloatBuffer& g = grad_output.data();
            FloatBuffer grad_x(y_data.size());
            for (size_t r = 0; r < rows; ++r) {
                const size_t base = r * cols;
                float dot = 0.0f;
                for (size_t c = 0; c < cols; ++c) {
                    dot += g[base + c] * y_data[base + c];
                }
                for (size_t c = 0; c < cols; ++c) {
                    grad_x[base + c] = y_data[base + c] * (g[base + c] - dot);
                }
            }
            return {Tensor(std::move(grad_x), grad_output.shape())};
        });
    }
    return out;
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

    Tensor out({loss_value}, {1});
    if (detail::should_record({&pred, &target})) {
        detail::record(out, {&pred, &target}, {&pred, &target},
                       [pred, target, n, numel](const Tensor& grad_output, const NeedsInputGrad& needs) -> InputGrads {
                           constexpr float kEpsilon = 1e-15f;
                           const float upstream = grad_output.data()[0];
                           FloatBuffer grad_pred(needs[0] ? numel : 0);
                           FloatBuffer grad_target(needs[1] ? numel : 0);
                           for (size_t i = 0; i < numel; ++i) {
                               const float clipped = std::clamp(pred.data()[i], kEpsilon, 1.0f - kEpsilon);
                               if (needs[0]) {
                                   grad_pred[i] = -upstream * target.data()[i] / (clipped * static_cast<float>(n));
                               }
                               if (needs[1]) {
                                   grad_target[i] = -upstream * std::log(clipped) / static_cast<float>(n);
                               }
                           }
                           InputGrads grads;
                           if (needs[0]) {
                               grads[0] = Tensor(std::move(grad_pred), pred.shape());
                           }
                           if (needs[1]) {
                               grads[1] = Tensor(std::move(grad_target), target.shape());
                           }
                           return grads;
                       });
    }
    return out;
}

Tensor softmax_cross_entropy(const Tensor& logits, const Tensor& target) {
    if (logits.shape() != target.shape()) {
        throw std::runtime_error("softmax_cross_entropy: logits and target shapes must match");
    }
    if (logits.shape().size() != 1 && logits.shape().size() != 2) {
        throw std::runtime_error("softmax_cross_entropy: expected a 1D or 2D tensor");
    }
    const size_t rows = logits.shape().size() == 2 ? logits.shape()[0] : 1;
    const size_t cols = logits.shape().size() == 2 ? logits.shape()[1] : logits.shape()[0];

    const float* const x = logits.data().data();
    const float* const t = target.data().data();
    // log_softmax(x) = x - (max + log(sum(exp(x - max)))), so no probability is ever clipped or
    // divided by; the probabilities come out of the same pass for backward.
    auto probs = std::make_shared<FloatBuffer>(rows * cols);
    auto log_norm = std::make_shared<FloatBuffer>(rows);
    float sum = 0.0f;
    for (size_t r = 0; r < rows; ++r) {
        const size_t base = r * cols;
        float row_max = x[base];
        for (size_t c = 1; c < cols; ++c) {
            row_max = std::max(row_max, x[base + c]);
        }
        float sum_exp = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            const float e = std::exp(x[base + c] - row_max);
            (*probs)[base + c] = e;
            sum_exp += e;
        }
        const float lse = row_max + std::log(sum_exp);
        (*log_norm)[r] = lse;
        for (size_t c = 0; c < cols; ++c) {
            (*probs)[base + c] /= sum_exp;
            sum += t[base + c] * (x[base + c] - lse);
        }
    }
    const float n = static_cast<float>(rows);

    Tensor out({-sum / n}, {1});
    if (detail::should_record({&logits, &target})) {
        detail::record(out, {&logits, &target}, {&logits, &target},
                       [logits, target, probs, log_norm, rows, cols, n](const Tensor& grad_output,
                                                                        const NeedsInputGrad& needs) -> InputGrads {
                           const float upstream = grad_output.data()[0];
                           const float* const x = logits.data().data();
                           const float* const t = target.data().data();
                           FloatBuffer grad_logits(needs[0] ? rows * cols : 0);
                           FloatBuffer grad_target(needs[1] ? rows * cols : 0);
                           for (size_t r = 0; r < rows; ++r) {
                               const size_t base = r * cols;
                               if (needs[0]) {
                                   float target_mass = 0.0f;
                                   for (size_t c = 0; c < cols; ++c) {
                                       target_mass += t[base + c];
                                   }
                                   for (size_t c = 0; c < cols; ++c) {
                                       grad_logits[base + c] =
                                           upstream * ((*probs)[base + c] * target_mass - t[base + c]) / n;
                                   }
                               }
                               if (needs[1]) {
                                   for (size_t c = 0; c < cols; ++c) {
                                       grad_target[base + c] = -upstream * (x[base + c] - (*log_norm)[r]) / n;
                                   }
                               }
                           }
                           InputGrads grads;
                           if (needs[0]) {
                               grads[0] = Tensor(std::move(grad_logits), logits.shape());
                           }
                           if (needs[1]) {
                               grads[1] = Tensor(std::move(grad_target), target.shape());
                           }
                           return grads;
                       });
    }
    return out;
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

    dnnl::engine& engine = cpu_engine();

    memory::dims src_dims = detail::to_dims(x.shape());
    memory::dims weights_dims = detail::to_dims(weight.shape());
    memory::dims bias_dims = {static_cast<memory::dim>(c_out)};
    memory::dims dst_dims = {static_cast<memory::dim>(N), static_cast<memory::dim>(c_out),
                             static_cast<memory::dim>(h_out), static_cast<memory::dim>(w_out)};
    memory::dims strides_dims = {static_cast<memory::dim>(stride), static_cast<memory::dim>(stride)};
    memory::dims padding_dims = {static_cast<memory::dim>(padding), static_cast<memory::dim>(padding)};

    // format_tag::any lets oneDNN pick its optimized blocked-layout AVX-512 kernel instead of the
    // reference path plain NCHW/OIHW forces (confirmed via ONEDNN_VERBOSE). The output keeps that
    // layout, and an input already in the chosen layout (the previous conv/BN/ReLU/pool's output)
    // is used without conversion, so a conv -> BN -> ReLU -> pool chain never round-trips to NCHW.
    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::any);
    auto weights_md = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::any);
    auto bias_md = memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::x);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::any);

    const bool record_graph = detail::should_record({&x, &weight, &bias});
    const dnnl::prop_kind prop_kind =
        record_graph ? dnnl::prop_kind::forward_training : dnnl::prop_kind::forward_inference;

    ConvKey key{N, c_in, H, W, c_out, kh, kw, stride, padding, prop_kind};
    auto& cache = cache_for<ConvKey, ConvCacheEntry, ConvKeyHash>();
    auto cache_it = cache.find(key);
    if (cache_it == cache.end()) {
        auto fwd_pd = dnnl::convolution_forward::primitive_desc(
            engine, prop_kind, dnnl::algorithm::convolution_direct, src_md, weights_md, bias_md, dst_md,
            strides_dims, padding_dims, padding_dims);
        dnnl::convolution_forward fwd_prim(fwd_pd);
        auto dst_layout = detail::storage_layout(fwd_pd.dst_desc());
        ConvCacheEntry entry{std::move(fwd_pd), std::move(fwd_prim), std::move(dst_layout)};
        cache_it = cache.emplace(key, std::move(entry)).first;
    }
    ConvCacheEntry& entry = cache_it->second;
    const dnnl::convolution_forward::primitive_desc& fwd_pd = entry.fwd_pd;

    memory src_mem = detail::memory_as(x, fwd_pd.src_desc());
    memory weights_mem = detail::memory_as(weight, fwd_pd.weights_desc());
    FloatBuffer out_data = detail::buffer_for(fwd_pd.dst_desc());

    entry.fwd_prim.execute(cpu_stream(), {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_BIAS, memory(fwd_pd.bias_desc(), engine, const_cast<float*>(bias.data().data()))},
        {DNNL_ARG_DST, memory(fwd_pd.dst_desc(), engine, out_data.data())},
    });
    cpu_stream().wait();

    Tensor out = detail::make_tensor(std::move(out_data), {N, c_out, h_out, w_out}, entry.dst_layout);
    if (!record_graph) {
        return out;
    }
    // Backward reuses the forward's converted src/weights instead of converting them again. A
    // memory that is merely a view over a tensor's own storage is not kept: that storage could be
    // converted to plain in the meantime (a data() read), so it is looked up again in backward.
    std::optional<memory> saved_src;
    if (is_converted(src_mem, x)) {
        saved_src = src_mem;
    }
    std::optional<memory> saved_weights;
    if (is_converted(weights_mem, weight)) {
        saved_weights = weights_mem;
    }
    detail::record(out, {&x, &weight, &bias}, {&x, &weight},
                   [key, x, weight, saved_src, saved_weights](const Tensor& grad_output,
                                                              const NeedsInputGrad& needs) -> InputGrads {
        dnnl::engine& engine = cpu_engine();
        dnnl::stream& stream = cpu_stream();
        ConvCacheEntry& entry = cache_for<ConvKey, ConvCacheEntry, ConvKeyHash>().at(key);
        const dnnl::convolution_forward::primitive_desc& fwd_pd = entry.fwd_pd;
        const memory::dims strides_dims = {static_cast<memory::dim>(key.stride), static_cast<memory::dim>(key.stride)};
        const memory::dims padding_dims = {static_cast<memory::dim>(key.padding),
                                           static_cast<memory::dim>(key.padding)};

        auto src_as = [&](const memory::desc& want) {
            return saved_src.has_value() ? detail::convert(*saved_src, want) : detail::memory_as(x, want);
        };
        auto weights_as = [&](const memory::desc& want) {
            return saved_weights.has_value() ? detail::convert(*saved_weights, want) : detail::memory_as(weight, want);
        };

        InputGrads grads;
        std::optional<memory> diff_dst;
        // The data gradient is a full extra convolution pass, and is wasted on a first layer
        // whose input is a data batch -- skip it unless x actually needs a gradient.
        if (needs[0]) {
            if (!entry.bwd_data_pd.has_value()) {
                entry.bwd_data_pd = dnnl::convolution_backward_data::primitive_desc(
                    engine, dnnl::algorithm::convolution_direct, any_like(fwd_pd.src_desc()),
                    any_like(fwd_pd.weights_desc()), any_like(fwd_pd.dst_desc()), strides_dims, padding_dims,
                    padding_dims, fwd_pd);
                entry.bwd_data_prim = dnnl::convolution_backward_data(*entry.bwd_data_pd);
                entry.diff_src_layout = detail::storage_layout(entry.bwd_data_pd->diff_src_desc());
            }
            const dnnl::convolution_backward_data::primitive_desc& bwd_data_pd = *entry.bwd_data_pd;
            diff_dst = detail::memory_as(grad_output, bwd_data_pd.diff_dst_desc());
            FloatBuffer grad_x_data = detail::buffer_for(bwd_data_pd.diff_src_desc());
            entry.bwd_data_prim->execute(stream, {
                {DNNL_ARG_DIFF_DST, *diff_dst},
                {DNNL_ARG_WEIGHTS, weights_as(bwd_data_pd.weights_desc())},
                {DNNL_ARG_DIFF_SRC, memory(bwd_data_pd.diff_src_desc(), engine, grad_x_data.data())},
            });
            stream.wait();
            grads[0] = detail::make_tensor(std::move(grad_x_data), x.shape(), entry.diff_src_layout);
        }

        if (needs[1] || needs[2]) {
            if (!entry.bwd_weights_pd.has_value()) {
                entry.bwd_weights_pd = dnnl::convolution_backward_weights::primitive_desc(
                    engine, dnnl::algorithm::convolution_direct, any_like(fwd_pd.src_desc()),
                    any_like(fwd_pd.weights_desc()), fwd_pd.bias_desc(), any_like(fwd_pd.dst_desc()), strides_dims,
                    padding_dims, padding_dims, fwd_pd);
                entry.bwd_weights_prim = dnnl::convolution_backward_weights(*entry.bwd_weights_pd);
            }
            const dnnl::convolution_backward_weights::primitive_desc& bwd_weights_pd = *entry.bwd_weights_pd;
            const memory::desc plain_weights_md = detail::plain_desc(detail::to_dims(weight.shape()));
            const memory::desc& diff_weights_md = bwd_weights_pd.diff_weights_desc();

            memory diff_dst_weights = diff_dst.has_value()
                                          ? detail::convert(*diff_dst, bwd_weights_pd.diff_dst_desc())
                                          : detail::memory_as(grad_output, bwd_weights_pd.diff_dst_desc());
            FloatBuffer grad_w_data(weight.numel());
            memory grad_w_plain(plain_weights_md, engine, grad_w_data.data());
            const bool weights_plain = diff_weights_md == plain_weights_md;
            if (!weights_plain && !entry.diff_weights_scratch.has_value()) {
                entry.diff_weights_scratch = memory(diff_weights_md, engine);
            }
            FloatBuffer grad_b_data(key.c_out);
            entry.bwd_weights_prim->execute(stream, {
                {DNNL_ARG_SRC, src_as(bwd_weights_pd.src_desc())},
                {DNNL_ARG_DIFF_DST, diff_dst_weights},
                {DNNL_ARG_DIFF_WEIGHTS, weights_plain ? grad_w_plain : *entry.diff_weights_scratch},
                {DNNL_ARG_DIFF_BIAS, memory(bwd_weights_pd.diff_bias_desc(), engine, grad_b_data.data())},
            });
            stream.wait();
            if (needs[1]) {
                if (!weights_plain) {
                    detail::reorder(*entry.diff_weights_scratch, grad_w_plain);
                }
                grads[1] = Tensor(std::move(grad_w_data), weight.shape());
            }
            if (needs[2]) {
                grads[2] = Tensor(std::move(grad_b_data), {key.c_out});
            }
        }
        return grads;
    });
    return out;
}

Tensor max_pool2d(const Tensor& x, size_t kernel_size, size_t stride) {
    return pool2d(x, kernel_size, kernel_size, stride, stride, dnnl::algorithm::pooling_max, "max_pool2d");
}

Tensor avg_pool2d(const Tensor& x, size_t kernel_size, size_t stride) {
    return pool2d(x, kernel_size, kernel_size, stride, stride, dnnl::algorithm::pooling_avg_exclude_padding,
                  "avg_pool2d");
}

Tensor flatten(const Tensor& x) {
    if (x.shape().empty()) {
        throw std::runtime_error("flatten: expected a tensor with at least 1 dimension");
    }
    const size_t n = x.shape()[0];
    const size_t rest = n == 0 ? 0 : x.numel() / n;

    Tensor out = detail::view(x, {n, rest});
    if (detail::should_record({&x})) {
        detail::record(out, {&x}, {}, [x_shape = x.shape()](const Tensor& grad_output) -> InputGrads {
            return {detail::view(grad_output, x_shape)};
        });
    }
    return out;
}

Tensor batch_norm2d(const Tensor& x, const Tensor& gamma, const Tensor& beta, std::vector<float>& running_mean,
                     std::vector<float>& running_var, bool training, float momentum, float eps) {
    if (x.shape().size() != 4) {
        throw std::runtime_error("batch_norm2d: expected a 4D (N, C, H, W) input tensor");
    }
    const size_t C = x.shape()[1];
    if (gamma.shape() != std::vector<size_t>{C} || beta.shape() != std::vector<size_t>{C}) {
        throw std::runtime_error("batch_norm2d: gamma and beta must be 1D with C entries");
    }
    if (running_mean.size() != C || running_var.size() != C) {
        throw std::runtime_error("batch_norm2d: running_mean and running_var must have C entries");
    }
    return batch_norm(x, detail::to_dims(x.shape()), gamma, beta, running_mean, running_var, training, momentum, eps);
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
    const memory::dims dims = {static_cast<memory::dim>(N), static_cast<memory::dim>(C), 1, 1};
    return batch_norm(x, dims, gamma, beta, running_mean, running_var, training, momentum, eps);
}

Tensor global_avg_pool2d(const Tensor& x) {
    if (x.shape().size() != 4) {
        throw std::runtime_error("global_avg_pool2d: expected a 4D (N, C, H, W) tensor");
    }
    return pool2d(x, x.shape()[2], x.shape()[3], 1, 1, dnnl::algorithm::pooling_avg_exclude_padding,
                  "global_avg_pool2d");
}

Tensor global_max_pool2d(const Tensor& x) {
    if (x.shape().size() != 4) {
        throw std::runtime_error("global_max_pool2d: expected a 4D (N, C, H, W) tensor");
    }
    return pool2d(x, x.shape()[2], x.shape()[3], 1, 1, dnnl::algorithm::pooling_max, "global_max_pool2d");
}

Tensor dropout(const Tensor& x, float p, bool training, std::mt19937& rng) {
    if (p < 0.0f || p >= 1.0f) {
        throw std::runtime_error("dropout: p must be in [0, 1)");
    }
    if (!training || p == 0.0f) {
        Tensor out = detail::view(x, x.shape());
        if (detail::should_record({&x})) {
            detail::record(out, {&x}, {}, [](const Tensor& grad_output) -> InputGrads {
                return {grad_output};
            });
        }
        return out;
    }

    const float keep_prob = 1.0f - p;
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    const FloatBuffer& x_data = x.data();
    auto mask = std::make_shared<FloatBuffer>(x.numel());
    FloatBuffer out_data(x.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        const float keep = uniform(rng) < keep_prob ? 1.0f / keep_prob : 0.0f;
        (*mask)[i] = keep;
        out_data[i] = x_data[i] * keep;
    }

    Tensor out(std::move(out_data), x.shape());
    if (detail::should_record({&x})) {
        detail::record(out, {&x}, {}, [mask](const Tensor& grad_output) -> InputGrads {
            const FloatBuffer& g = grad_output.data();
            FloatBuffer grad_x(mask->size());
            for (size_t i = 0; i < grad_x.size(); ++i) {
                grad_x[i] = g[i] * (*mask)[i];
            }
            return {Tensor(std::move(grad_x), grad_output.shape())};
        });
    }
    return out;
}

}  // namespace advanceml
