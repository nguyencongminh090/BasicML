#pragma once

#include "advanceml/nn/module.hpp"

namespace advanceml {

/**
 * Parameterless Swish (SiLU when `beta = 1`) layer: `forward(x) =
 * swish(x, beta)`. `parameters()` returns empty (inherited from
 * `Module`); backward is handled by the `swish` op itself.
 */
class Swish : public Module {
public:
    /** Constructs a layer with the given sigmoid steepness `beta`. */
    explicit Swish(float beta = 1.0f);

    Tensor forward(const Tensor& x) override;

private:
    float beta_;
};

}  // namespace advanceml
