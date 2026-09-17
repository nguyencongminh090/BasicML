#pragma once

#include "advanceml/nn/module.hpp"

namespace advanceml {

/**
 * Parameterless hard tanh layer: `forward(x) = hardtanh(x, min_val,
 * max_val)`. `parameters()` returns empty (inherited from `Module`);
 * backward is handled by the `hardtanh` op itself.
 */
class Hardtanh : public Module {
public:
    /** Constructs a layer clipping to `[min_val, max_val]`. @throws std::runtime_error if `max_val <= min_val`. */
    explicit Hardtanh(float min_val = -1.0f, float max_val = 1.0f);

    Tensor forward(const Tensor& x) override;

private:
    float min_val_;
    float max_val_;
};

}  // namespace advanceml
