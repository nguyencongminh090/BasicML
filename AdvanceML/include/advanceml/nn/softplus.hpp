#pragma once

#include "advanceml/nn/module.hpp"

namespace advanceml {

/**
 * Parameterless softplus layer: `forward(x) = softplus(x)`.
 * `parameters()` returns empty (inherited from `Module`); backward is
 * handled by the `softplus` op itself.
 */
class Softplus : public Module {
public:
    Tensor forward(const Tensor& x) override;
};

}  // namespace advanceml
