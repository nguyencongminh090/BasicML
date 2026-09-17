#pragma once

#include "advanceml/nn/module.hpp"

namespace advanceml {

/**
 * Parameterless exponential linear unit layer: `forward(x) = elu(x,
 * alpha)`. `parameters()` returns empty (inherited from `Module`);
 * backward is handled by the `elu` op itself.
 */
class ELU : public Module {
public:
    /** Constructs a layer with the given negative-side scale `alpha`. */
    explicit ELU(float alpha = 1.0f);

    Tensor forward(const Tensor& x) override;

private:
    float alpha_;
};

}  // namespace advanceml
