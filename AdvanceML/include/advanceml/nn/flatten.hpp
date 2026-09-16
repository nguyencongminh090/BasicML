#pragma once

#include "advanceml/nn/module.hpp"

namespace advanceml {

/**
 * Parameterless layer that flattens every dimension but the first
 * (batch) one: `forward(x) = flatten(x)`, bridging a conv/pooling
 * stack's `(N, C, H, W)` output to a `Linear` layer's `(N, D)` input.
 * `parameters()` returns empty (inherited from `Module`); backward is
 * handled by the `flatten` op itself.
 */
class Flatten : public Module {
public:
    Tensor forward(const Tensor& x) override;
};

}  // namespace advanceml
