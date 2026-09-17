#include "advanceml/tensor.hpp"

#include "advanceml/detail/autograd.hpp"

#include <atomic>
#include <random>
#include <stdexcept>
#include <string>

namespace advanceml {

namespace {

thread_local bool grad_enabled = true;

uint64_t next_backward_mark() {
    static std::atomic<uint64_t> counter{0};
    return ++counter;
}

std::shared_ptr<TensorImpl> copy_impl(const TensorImpl& source, const std::vector<size_t>& shape) {
    auto impl = std::make_shared<TensorImpl>();
    impl->storage = std::make_shared<Storage>(Storage{source.storage->data, 0});
    impl->shape = shape;
    return impl;
}

// Adds `grad` into `slot`. The first gradient is adopted as-is (no copy); later ones are added
// in place only when nothing else can observe the slot's buffer -- gradients are freely shared
// (operator+ hands the same tensor to both inputs, reshape views share storage), so mutating a
// shared buffer would corrupt another input's gradient.
void accumulate(std::shared_ptr<TensorImpl>& slot, std::shared_ptr<TensorImpl> grad) {
    if (!slot) {
        slot = std::move(grad);
        return;
    }
    if (slot->storage->data.size() != grad->storage->data.size()) {
        throw std::runtime_error("backward: gradient size does not match an earlier gradient for the same tensor");
    }
    if (slot.use_count() != 1 || slot->storage.use_count() != 1) {
        slot = copy_impl(*slot, slot->shape);
    }
    std::vector<float>& dst = slot->storage->data;
    const std::vector<float>& src = grad->storage->data;
    for (size_t i = 0; i < dst.size(); ++i) {
        dst[i] += src[i];
    }
}

void accumulate_leaf(TensorImpl& leaf, std::shared_ptr<TensorImpl> grad) {
    if (grad->storage->data.size() != leaf.storage->data.size()) {
        throw std::runtime_error("backward: gradient size does not match its leaf tensor");
    }
    if (leaf.grad) {
        accumulate(leaf.grad, std::move(grad));
        return;
    }
    // The leaf's gradient is user-visible via grad(), so it may only adopt a buffer no one else holds.
    if (grad.use_count() == 1 && grad->storage.use_count() == 1) {
        grad->shape = leaf.shape;
        leaf.grad = std::move(grad);
    } else {
        leaf.grad = copy_impl(*grad, leaf.shape);
    }
}

// Iterative post-order DFS (explicit stack, so graph depth is not bounded by the call stack).
// Only edges whose input needs a gradient are followed, which prunes data-only subgraphs.
// Reversing the result visits every tensor after all of its consumers.
std::vector<std::shared_ptr<TensorImpl>> topological_order(const std::shared_ptr<TensorImpl>& root, uint64_t mark) {
    struct Frame {
        const std::shared_ptr<TensorImpl>* impl;
        size_t next_input;
    };
    std::vector<std::shared_ptr<TensorImpl>> order;
    std::vector<Frame> stack;
    root->visit_mark = mark;
    stack.push_back({&root, 0});
    while (!stack.empty()) {
        Frame& frame = stack.back();
        const Node* node = (*frame.impl)->grad_fn.get();
        if (node != nullptr && frame.next_input < node->num_inputs) {
            const size_t i = frame.next_input++;
            const std::shared_ptr<TensorImpl>& input = node->inputs[i];
            if (node->needs_input_grad[i] && input && input->visit_mark != mark) {
                input->visit_mark = mark;
                stack.push_back({&input, 0});
            }
            continue;
        }
        order.push_back(*frame.impl);
        stack.pop_back();
    }
    return order;
}

}  // namespace

Node::~Node() {
    // Destroying a node releases its inputs, whose nodes release theirs, and so on: a naive
    // destructor recurses once per graph level. Detach every node this one exclusively owns and
    // destroy them from a flat worklist instead.
    std::vector<std::shared_ptr<Node>> worklist;
    auto detach_inputs = [&worklist](std::array<std::shared_ptr<TensorImpl>, kMaxNodeInputs>& edges) {
        for (std::shared_ptr<TensorImpl>& input : edges) {
            if (input && input.use_count() == 1 && input->grad_fn && input->grad_fn.use_count() == 1) {
                worklist.push_back(std::move(input->grad_fn));
            }
            input.reset();
        }
    };
    detach_inputs(inputs);
    while (!worklist.empty()) {
        std::shared_ptr<Node> node = std::move(worklist.back());
        worklist.pop_back();
        node->release_saved();
        detach_inputs(node->inputs);
    }
}

void Node::release_saved() noexcept {
    saved = {};
    num_saved = 0;
}

void Node::release() noexcept {
    release_saved();
    inputs = {};
    released_ = true;
}

bool Node::released() const noexcept {
    return released_;
}

void Node::check_saved_versions() const {
    if (released_) {
        throw std::runtime_error(
            "backward: this graph's saved tensors were already released by an earlier backward(); "
            "pass retain_graph=true to the first backward() to backpropagate through it again");
    }
    for (size_t i = 0; i < num_saved; ++i) {
        if (saved[i].storage->version != saved[i].version) {
            throw std::runtime_error("backward: a tensor saved for backward was modified in place (version " +
                                     std::to_string(saved[i].storage->version) + ", expected " +
                                     std::to_string(saved[i].version) + ")");
        }
    }
}

bool is_grad_enabled() noexcept {
    return grad_enabled;
}

NoGradGuard::NoGradGuard() noexcept : previous_(grad_enabled) {
    grad_enabled = false;
}

NoGradGuard::~NoGradGuard() {
    grad_enabled = previous_;
}

Tensor::Tensor(std::vector<float> data, std::vector<size_t> shape, bool requires_grad) {
    if (data.size() != detail::numel_of(shape)) {
        throw std::runtime_error("Tensor: data size does not match shape");
    }
    impl_ = std::make_shared<TensorImpl>();
    impl_->storage = std::make_shared<Storage>(Storage{std::move(data), 0});
    impl_->shape = std::move(shape);
    impl_->requires_grad = requires_grad;
}

Tensor Tensor::zeros(std::vector<size_t> shape, bool requires_grad) {
    const size_t n = detail::numel_of(shape);
    return Tensor(std::vector<float>(n, 0.0f), std::move(shape), requires_grad);
}

Tensor Tensor::random_uniform(std::vector<size_t> shape, float low, float high, unsigned seed, bool requires_grad) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<float> data(detail::numel_of(shape));
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
    return impl_->storage->data.size();
}

const std::vector<size_t>& Tensor::shape() const noexcept {
    return impl_->shape;
}

const std::vector<float>& Tensor::data() const noexcept {
    return impl_->storage->data;
}

std::vector<float>& Tensor::mutable_data() noexcept {
    ++impl_->storage->version;
    return impl_->storage->data;
}

uint64_t Tensor::version() const noexcept {
    return impl_->storage->version;
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

void Tensor::backward(bool retain_graph) {
    if (numel() != 1) {
        throw std::runtime_error("Tensor::backward(): only supported on scalar tensors");
    }
    if (!impl_->requires_grad) {
        return;
    }

    NoGradGuard no_grad;
    std::vector<std::shared_ptr<TensorImpl>> order = topological_order(impl_, next_backward_mark());
    impl_->pending_grad = Tensor({1.0f}, {1}).impl_;

    // Pending gradients live on the tensors themselves; if a backward function throws, clear
    // the ones not yet consumed so they cannot leak into the next backward pass.
    struct PendingCleanup {
        std::vector<std::shared_ptr<TensorImpl>>& order;
        ~PendingCleanup() {
            for (const std::shared_ptr<TensorImpl>& impl : order) {
                if (impl) {
                    impl->pending_grad.reset();
                }
            }
        }
    } cleanup{order};

    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        TensorImpl& impl = **it;
        std::shared_ptr<TensorImpl> grad = std::move(impl.pending_grad);
        if (grad) {
            if (!impl.grad_fn) {
                accumulate_leaf(impl, std::move(grad));
            } else {
                Node& node = *impl.grad_fn;
                node.check_saved_versions();
                InputGrads input_grads = node.apply(Tensor::from_impl(std::move(grad)));
                for (size_t i = 0; i < node.num_inputs; ++i) {
                    if (node.needs_input_grad[i] && input_grads[i].has_value()) {
                        accumulate(node.inputs[i]->pending_grad, std::move(input_grads[i]->impl_));
                        input_grads[i].reset();
                    }
                }
                if (!retain_graph) {
                    node.release();
                }
            }
        }
        // Dropping the order's reference lets a released intermediate (and its activation buffer)
        // be freed now rather than when the whole pass ends; every input still to be processed is
        // held by its own later entry.
        if (!retain_graph) {
            it->reset();
        }
    }
}

Tensor operator+(const Tensor& a, const Tensor& b) {
    const bool elementwise = a.shape() == b.shape();
    const bool row_broadcast = a.shape().size() == 2 && b.shape().size() == 1 && a.shape()[1] == b.shape()[0];
    if (!elementwise && !row_broadcast) {
        throw std::runtime_error("operator+: incompatible shapes");
    }

    const std::vector<float>& a_data = a.data();
    const std::vector<float>& b_data = b.data();
    std::vector<float> out_data(a.numel());
    if (elementwise) {
        for (size_t i = 0; i < out_data.size(); ++i) {
            out_data[i] = a_data[i] + b_data[i];
        }
    } else {
        const size_t rows = a.shape()[0];
        const size_t cols = a.shape()[1];
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                out_data[r * cols + c] = a_data[r * cols + c] + b_data[c];
            }
        }
    }

    Tensor out(std::move(out_data), a.shape());
    if (detail::should_record({&a, &b})) {
        const size_t rows = elementwise ? 0 : a.shape()[0];
        const size_t cols = elementwise ? 0 : a.shape()[1];
        detail::record(out, {&a, &b}, {},
                       [row_broadcast, rows, cols](const Tensor& grad_output, const NeedsInputGrad& needs) -> InputGrads {
                           InputGrads grads;
                           if (needs[0]) {
                               grads[0] = grad_output;
                           }
                           if (needs[1]) {
                               if (!row_broadcast) {
                                   grads[1] = grad_output;
                               } else {
                                   const std::vector<float>& g = grad_output.data();
                                   std::vector<float> grad_b(cols, 0.0f);
                                   for (size_t r = 0; r < rows; ++r) {
                                       for (size_t c = 0; c < cols; ++c) {
                                           grad_b[c] += g[r * cols + c];
                                       }
                                   }
                                   grads[1] = Tensor(std::move(grad_b), {cols});
                               }
                           }
                           return grads;
                       });
    }
    return out;
}

Tensor operator*(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("operator*: shapes must match (elementwise only)");
    }

    const std::vector<float>& a_data = a.data();
    const std::vector<float>& b_data = b.data();
    std::vector<float> out_data(a.numel());
    for (size_t i = 0; i < out_data.size(); ++i) {
        out_data[i] = a_data[i] * b_data[i];
    }

    Tensor out(std::move(out_data), a.shape());
    if (detail::should_record({&a, &b})) {
        detail::record(out, {&a, &b}, {&a, &b},
                       [a, b](const Tensor& grad_output, const NeedsInputGrad& needs) -> InputGrads {
                           InputGrads grads;
                           if (needs[0]) {
                               grads[0] = grad_output * b;
                           }
                           if (needs[1]) {
                               grads[1] = grad_output * a;
                           }
                           return grads;
                       });
    }
    return out;
}

}  // namespace advanceml
