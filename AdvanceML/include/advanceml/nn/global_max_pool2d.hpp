#pragma once

#include "advanceml/nn/module.hpp"

namespace advanceml {

/**
 * Parameterless global max pooling layer: `forward(x) =
 * global_max_pool2d(x)`, collapsing a `(N, C, H, W)` input to `(N, C, 1,
 * 1)`. `parameters()` returns empty (inherited from `Module`); backward is
 * handled by the `global_max_pool2d` op itself.
 */
class GlobalMaxPool2D : public Module {
public:
    Tensor forward(const Tensor& x) override;
};

}  // namespace advanceml
