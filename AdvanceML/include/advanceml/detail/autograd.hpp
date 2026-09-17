#pragma once

#include "advanceml/tensor.hpp"

#include <functional>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * Helpers for op implementations to build autograd graph nodes. Not part of
 * the user-facing API: user code composes ops, it does not record nodes.
 */
namespace advanceml::detail {

/** @return The product of `shape`'s entries (`1` for an empty shape). */
inline size_t numel_of(const std::vector<size_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>());
}

/**
 * A `Node` whose backward is the op's lambda `F`, stored in the same
 * allocation as the node itself (no separate `std::function` heap state).
 *
 * `F` is callable either as `InputGrads(const Tensor& grad_output)` or, for
 * ops that can skip work per input, as `InputGrads(const Tensor& grad_output,
 * const NeedsInputGrad& needs)`.
 */
template <typename F>
class LambdaNode final : public Node {
public:
    explicit LambdaNode(F fn) : fn_(std::move(fn)) {}

    InputGrads apply(const Tensor& grad_output) override {
        if constexpr (std::is_invocable_v<F&, const Tensor&, const NeedsInputGrad&>) {
            return (*fn_)(grad_output, needs_input_grad);
        } else {
            return (*fn_)(grad_output);
        }
    }

    void release_saved() noexcept override {
        fn_.reset();
        Node::release_saved();
    }

private:
    std::optional<F> fn_;
};

/**
 * @return Whether an op reading `inputs` must record a node: recording is on
 * (no `NoGradGuard` alive) and at least one input requires a gradient.
 */
inline bool should_record(std::initializer_list<const Tensor*> inputs) noexcept {
    if (!is_grad_enabled()) {
        return false;
    }
    for (const Tensor* input : inputs) {
        if (input->requires_grad()) {
            return true;
        }
    }
    return false;
}

/**
 * Attaches a backward node to `out`, making it require a gradient.
 *
 * @param out The op's freshly created result tensor.
 * @param inputs The op's inputs, in the order `backward` returns their gradients.
 * @param saved Tensors whose values `backward` reads (inputs or `out` itself);
 * their storage versions are checked before `backward` runs. Capture `out`'s
 * storage in the lambda (not `out` itself) to avoid an ownership cycle through
 * `grad_fn`.
 * @param backward The op's backward lambda (see `LambdaNode`).
 * @throws std::logic_error if more than `kMaxNodeInputs` inputs or saved tensors are passed.
 */
template <typename F>
void record(Tensor& out, std::initializer_list<const Tensor*> inputs, std::initializer_list<const Tensor*> saved,
            F&& backward) {
    if (inputs.size() > kMaxNodeInputs || saved.size() > kMaxNodeInputs) {
        throw std::logic_error("record: too many inputs or saved tensors for one node");
    }
    auto node = std::make_shared<LambdaNode<std::decay_t<F>>>(std::forward<F>(backward));
    for (const Tensor* input : inputs) {
        node->inputs[node->num_inputs] = input->impl();
        node->needs_input_grad[node->num_inputs] = input->requires_grad();
        ++node->num_inputs;
    }
    for (const Tensor* tensor : saved) {
        const std::shared_ptr<Storage>& storage = tensor->impl()->storage;
        node->saved[node->num_saved++] = SavedVersion{storage, storage->version};
    }
    out.impl()->requires_grad = true;
    out.impl()->grad_fn = std::move(node);
}

/**
 * @return A tensor with the given `shape` sharing `source`'s whole storage
 * (no copy), with no gradient or graph attached.
 * @throws std::runtime_error if `shape`'s element count differs from `source.numel()`.
 */
inline Tensor view(const Tensor& source, std::vector<size_t> shape) {
    if (numel_of(shape) != source.numel()) {
        throw std::runtime_error("view: shape does not match the source element count");
    }
    auto impl = std::make_shared<TensorImpl>();
    impl->storage = source.impl()->storage;
    impl->shape = std::move(shape);
    return Tensor::from_impl(std::move(impl));
}

}  // namespace advanceml::detail
