#include "advanceml/tensor.hpp"

#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace advanceml {

namespace {

size_t numel_of(const std::vector<size_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>());
}

// Post-order DFS over the graph (a tensor's grad_fn->inputs are its
// dependencies), so reversing this list visits the root before any
// tensor it depends on -- the order backward() needs to guarantee
// every consumer of a tensor has already contributed to its gradient
// before that tensor's own backward_fn runs.
void build_topological_order(const std::shared_ptr<TensorImpl>& node,
                              std::unordered_set<TensorImpl*>& visited,
                              std::vector<std::shared_ptr<TensorImpl>>& topo_order) {
    if (visited.contains(node.get())) {
        return;
    }
    visited.insert(node.get());
    if (node->grad_fn) {
        for (const auto& input : node->grad_fn->inputs) {
            build_topological_order(input, visited, topo_order);
        }
    }
    topo_order.push_back(node);
}

}  // namespace

Tensor::Tensor(std::vector<float> data, std::vector<size_t> shape, bool requires_grad) {
    impl_ = std::make_shared<TensorImpl>();
    impl_->data = std::move(data);
    impl_->shape = std::move(shape);
    impl_->requires_grad = requires_grad;
    if (impl_->data.size() != numel_of(impl_->shape)) {
        throw std::runtime_error("Tensor: data size does not match shape");
    }
}

Tensor Tensor::zeros(std::vector<size_t> shape, bool requires_grad) {
    const size_t n = numel_of(shape);
    return Tensor(std::vector<float>(n, 0.0f), std::move(shape), requires_grad);
}

Tensor Tensor::random_uniform(std::vector<size_t> shape, float low, float high, unsigned seed, bool requires_grad) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<float> data(numel_of(shape));
    for (float& value : data) {
        value = dist(rng);
    }
    return Tensor(std::move(data), std::move(shape), requires_grad);
}

Tensor Tensor::from_impl(std::shared_ptr<TensorImpl> impl) {
    Tensor t;
    t.impl_ = std::move(impl);
    return t;
}

size_t Tensor::numel() const noexcept {
    return impl_->data.size();
}

const std::vector<size_t>& Tensor::shape() const noexcept {
    return impl_->shape;
}

std::vector<float>& Tensor::data() noexcept {
    return impl_->data;
}

const std::vector<float>& Tensor::data() const noexcept {
    return impl_->data;
}

bool Tensor::requires_grad() const noexcept {
    return impl_->requires_grad;
}

bool Tensor::has_grad() const noexcept {
    return impl_->grad != nullptr;
}

Tensor Tensor::grad() const {
    if (!impl_->grad) {
        throw std::runtime_error("Tensor::grad(): no gradient accumulated yet");
    }
    return Tensor::from_impl(impl_->grad);
}

void Tensor::zero_grad() {
    impl_->grad.reset();
}

const std::shared_ptr<TensorImpl>& Tensor::impl() const noexcept {
    return impl_;
}

void Tensor::backward() {
    if (numel() != 1) {
        throw std::runtime_error("Tensor::backward(): only supported on scalar tensors");
    }

    std::unordered_set<TensorImpl*> visited;
    std::vector<std::shared_ptr<TensorImpl>> topo_order;
    build_topological_order(impl_, visited, topo_order);

    std::unordered_map<TensorImpl*, Tensor> grad_of;
    grad_of.emplace(impl_.get(), Tensor({1.0f}, {1}));

    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
        const std::shared_ptr<TensorImpl>& node = *it;
        auto grad_it = grad_of.find(node.get());
        if (grad_it == grad_of.end()) {
            continue;
        }
        const Tensor& grad_output = grad_it->second;

        if (node->requires_grad && !node->grad_fn) {
            if (!node->grad) {
                node->grad = std::make_shared<TensorImpl>();
                node->grad->shape = node->shape;
                node->grad->data.assign(node->data.size(), 0.0f);
            }
            for (size_t i = 0; i < node->grad->data.size(); ++i) {
                node->grad->data[i] += grad_output.data()[i];
            }
        }

        if (node->grad_fn) {
            std::vector<Tensor> input_grads = node->grad_fn->backward_fn(grad_output);
            for (size_t i = 0; i < node->grad_fn->inputs.size(); ++i) {
                TensorImpl* input = node->grad_fn->inputs[i].get();
                auto existing = grad_of.find(input);
                if (existing == grad_of.end()) {
                    grad_of.emplace(input, input_grads[i]);
                } else {
                    existing->second = existing->second + input_grads[i];
                }
            }
        }
    }
}

Tensor operator+(const Tensor& a, const Tensor& b) {
    const bool elementwise = a.shape() == b.shape();
    const bool row_broadcast = a.shape().size() == 2 && b.shape().size() == 1 && a.shape()[1] == b.shape()[0];
    if (!elementwise && !row_broadcast) {
        throw std::runtime_error("operator+: incompatible shapes");
    }

    std::vector<float> out_data(a.numel());
    if (elementwise) {
        for (size_t i = 0; i < out_data.size(); ++i) {
            out_data[i] = a.data()[i] + b.data()[i];
        }
    } else {
        const size_t rows = a.shape()[0];
        const size_t cols = a.shape()[1];
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                out_data[r * cols + c] = a.data()[r * cols + c] + b.data()[c];
            }
        }
    }

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = std::move(out_data);
    out_impl->shape = a.shape();
    out_impl->requires_grad = a.requires_grad() || b.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {a.impl(), b.impl()};
        const bool is_row_broadcast = row_broadcast;
        const size_t rows = elementwise ? 0 : a.shape()[0];
        const size_t cols = elementwise ? 0 : a.shape()[1];
        node->backward_fn = [is_row_broadcast, rows, cols](const Tensor& grad_output) -> std::vector<Tensor> {
            if (!is_row_broadcast) {
                return {grad_output, grad_output};
            }
            std::vector<float> grad_b(cols, 0.0f);
            for (size_t r = 0; r < rows; ++r) {
                for (size_t c = 0; c < cols; ++c) {
                    grad_b[c] += grad_output.data()[r * cols + c];
                }
            }
            return {grad_output, Tensor(std::move(grad_b), {cols})};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

Tensor operator*(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("operator*: shapes must match (elementwise only)");
    }

    std::vector<float> out_data(a.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        out_data[i] = a.data()[i] * b.data()[i];
    }

    auto out_impl = std::make_shared<TensorImpl>();
    out_impl->data = std::move(out_data);
    out_impl->shape = a.shape();
    out_impl->requires_grad = a.requires_grad() || b.requires_grad();

    if (out_impl->requires_grad) {
        auto node = std::make_shared<Node>();
        node->inputs = {a.impl(), b.impl()};
        Tensor a_copy = a;
        Tensor b_copy = b;
        node->backward_fn = [a_copy, b_copy](const Tensor& grad_output) -> std::vector<Tensor> {
            return {grad_output * b_copy, grad_output * a_copy};
        };
        out_impl->grad_fn = std::move(node);
    }

    return Tensor::from_impl(std::move(out_impl));
}

}  // namespace advanceml
