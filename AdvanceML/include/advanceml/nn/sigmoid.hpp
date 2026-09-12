#pragma once

#include "advanceml/nn/module.hpp"

namespace advanceml {

/**
 * Parameterless logistic sigmoid layer: `forward(x) = sigmoid(x) = 1 /
 * (1 + exp(-x))`, elementwise. `parameters()` returns empty (inherited
 * from `Module`); backward is handled by the `sigmoid` op itself.
 */
class Sigmoid : public Module {
public:
    Tensor forward(const Tensor& x) override;
};

}  // namespace advanceml
