#pragma once

#include "advanceml/nn/module.hpp"

namespace advanceml {

/**
 * Parameterless hyperbolic tangent layer: `forward(x) = tanh(x)`.
 * `parameters()` returns empty (inherited from `Module`); backward is
 * handled by the `tanh` op itself.
 */
class Tanh : public Module {
public:
    Tensor forward(const Tensor& x) override;
};

}  // namespace advanceml
