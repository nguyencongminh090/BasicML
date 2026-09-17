#pragma once

#include "advanceml/nn/module.hpp"

namespace advanceml {

/**
 * Parameterless hard sigmoid layer: `forward(x) = hardsigmoid(x)`.
 * `parameters()` returns empty (inherited from `Module`); backward is
 * handled by the `hardsigmoid` op itself.
 */
class Hardsigmoid : public Module {
public:
    Tensor forward(const Tensor& x) override;
};

}  // namespace advanceml
