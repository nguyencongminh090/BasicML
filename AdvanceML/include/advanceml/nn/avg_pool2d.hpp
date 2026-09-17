#pragma once

#include "advanceml/nn/module.hpp"

#include <cstddef>

namespace advanceml {

/**
 * Parameterless 2D average pooling layer: `forward(x) = avg_pool2d(x,
 * kernel_size, stride)`. `parameters()` returns empty (inherited from
 * `Module`); backward is handled by the `avg_pool2d` op itself.
 */
class AvgPool2D : public Module {
public:
    AvgPool2D(size_t kernel_size, size_t stride);

    Tensor forward(const Tensor& x) override;

private:
    size_t kernel_size_;
    size_t stride_;
};

}  // namespace advanceml
