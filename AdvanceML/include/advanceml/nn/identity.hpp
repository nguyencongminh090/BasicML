#pragma once

#include "advanceml/nn/module.hpp"

namespace advanceml {

/**
 * Parameterless pass-through layer: `forward(x) = x`. `parameters()`
 * returns empty (inherited from `Module`); backward is handled by the
 * `identity` op itself.
 */
class Identity : public Module {
public:
    Tensor forward(const Tensor& x) override;
};

}  // namespace advanceml
