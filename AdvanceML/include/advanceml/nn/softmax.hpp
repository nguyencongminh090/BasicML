#pragma once

#include "advanceml/nn/module.hpp"

namespace advanceml {

/**
 * Parameterless softmax layer over the last axis: for a 1D input, over the
 * whole vector; for a 2D `(N, D)` input, independently per row.
 * `parameters()` returns empty (inherited from `Module`); backward is
 * handled by the `softmax` op itself.
 */
class Softmax : public Module {
public:
    Tensor forward(const Tensor& x) override;
};

}  // namespace advanceml
