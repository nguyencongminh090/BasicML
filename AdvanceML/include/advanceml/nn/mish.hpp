#pragma once

#include "advanceml/nn/module.hpp"

namespace advanceml {

/**
 * Parameterless Mish layer: `forward(x) = mish(x)`. `parameters()`
 * returns empty (inherited from `Module`); backward is handled by the
 * `mish` op itself.
 */
class Mish : public Module {
public:
    Tensor forward(const Tensor& x) override;
};

}  // namespace advanceml
