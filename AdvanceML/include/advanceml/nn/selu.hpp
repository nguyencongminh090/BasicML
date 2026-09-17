#pragma once

#include "advanceml/nn/module.hpp"

namespace advanceml {

/**
 * Parameterless scaled exponential linear unit layer: `forward(x) =
 * selu(x)`. `parameters()` returns empty (inherited from `Module`);
 * backward is handled by the `selu` op itself.
 */
class SELU : public Module {
public:
    Tensor forward(const Tensor& x) override;
};

}  // namespace advanceml
