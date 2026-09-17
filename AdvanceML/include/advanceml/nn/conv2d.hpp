#pragma once

#include "advanceml/nn/module.hpp"

#include <cstddef>

namespace advanceml {

/**
 * 2D convolution layer: `y = conv2d(x, weight, bias)`, with `weight`
 * shaped `(out_channels, in_channels, kernel_size, kernel_size)` and
 * `bias` shaped `(out_channels)`.
 *
 * `weight` is Xavier-uniform initialized (`limit = sqrt(6 / (fan_in +
 * fan_out))`, with `fan_in = in_channels * kernel_size^2` and `fan_out =
 * out_channels * kernel_size^2`, drawn from `[-limit, limit)`); `bias`
 * starts at zero. Both are `requires_grad` leaves owned by this layer.
 * Backward is handled entirely by the `conv2d` op `forward()` calls --
 * this class adds no `grad_fn` of its own.
 */
class Conv2D : public Module {
public:
    /** Constructs a `(out_channels, in_channels, kernel_size, kernel_size)` layer; `seed` drives `weight`'s init draw. */
    Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride, size_t padding, unsigned seed);

    Tensor forward(const Tensor& x) override;
    [[nodiscard]] std::vector<Tensor> parameters() const override;

private:
    Tensor weight_;
    Tensor bias_;
    size_t stride_;
    size_t padding_;
};

}  // namespace advanceml
