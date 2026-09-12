#pragma once

#include "advanceml/nn/module.hpp"

namespace advanceml {

/**
 * Parameterless leaky rectified linear unit layer: `forward(x) = x` where
 * `x > 0`, else `negative_slope * x`, elementwise. `parameters()` returns
 * empty (inherited from `Module`); backward is handled by the
 * `leaky_relu` op itself.
 */
class LeakyReLU : public Module {
public:
    /** Constructs the layer with the given slope for `x <= 0` (default `0.01`). */
    explicit LeakyReLU(float negative_slope = 0.01f);

    Tensor forward(const Tensor& x) override;

private:
    float negative_slope_;
};

}  // namespace advanceml
