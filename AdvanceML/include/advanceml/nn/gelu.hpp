#pragma once

#include "advanceml/nn/module.hpp"

namespace advanceml {

/**
 * Parameterless Gaussian Error Linear Unit layer: `forward(x) = x * 0.5 *
 * (1 + erf(x / sqrt(2)))`, elementwise (exact, erf-based form).
 * `parameters()` returns empty (inherited from `Module`); backward is
 * handled by the `gelu` op itself.
 */
class GELU : public Module {
public:
    Tensor forward(const Tensor& x) override;
};

}  // namespace advanceml
