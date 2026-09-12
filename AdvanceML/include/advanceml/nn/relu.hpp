#pragma once

#include "advanceml/nn/module.hpp"

namespace advanceml {

/**
 * Parameterless rectified linear unit layer: `forward(x) = relu(x) =
 * max(0, x)`, elementwise. `parameters()` returns empty (inherited
 * from `Module`); backward is handled by the `relu` op itself.
 */
class ReLU : public Module {
public:
    Tensor forward(const Tensor& x) override;
};

}  // namespace advanceml
