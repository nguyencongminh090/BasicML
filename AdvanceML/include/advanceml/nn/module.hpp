#pragma once

#include "advanceml/tensor.hpp"

#include <vector>

namespace advanceml {

/**
 * Abstract base for a layer or composite model in the autograd-tracked API.
 *
 * Mirrors `basicml.nn.module.Module`'s contract (`forward()`, `parameters()`),
 * but `forward()` need not cache anything for a manual `backward()`: the
 * `Tensor` ops it calls already record their own `grad_fn`, so the graph
 * built during `forward()` is enough for `Tensor::backward()` to work.
 */
class Module {
public:
    virtual ~Module() = default;

    /** Computes this layer's output from input `x`, extending the autograd graph. */
    virtual Tensor forward(const Tensor& x) = 0;

    /** @returns This layer's trainable parameters (empty for parameterless layers). */
    [[nodiscard]] virtual std::vector<Tensor> parameters() const;

    /** Equivalent to `forward(x)`. */
    Tensor operator()(const Tensor& x);
};

}  // namespace advanceml
