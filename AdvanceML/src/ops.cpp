#include "advanceml/ops.hpp"

#include <dnnl.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

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

    auto matmul_pd = dnnl::matmul::primitive_desc(engine, a_md, b_md, c_md);
    dnnl::matmul(matmul_pd).execute(cpu_stream(), {
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

}  // namespace advanceml
