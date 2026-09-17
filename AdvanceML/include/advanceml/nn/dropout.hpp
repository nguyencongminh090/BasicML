#pragma once

#include "advanceml/nn/module.hpp"

#include <random>

namespace advanceml {

/**
 * Inverted dropout layer: `forward(x) = dropout(x, p, training, rng)`,
 * zeroing each element independently with probability `p` during training
 * and scaling survivors by `1 / (1 - p)`; a no-op during inference.
 * `parameters()` returns empty (inherited from `Module`); backward is
 * handled by the `dropout` op itself.
 */
class Dropout : public Module {
public:
    /** Constructs a layer with drop probability `p` (`0 <= p < 1`), seeding its own RNG. */
    explicit Dropout(float p = 0.5f, unsigned seed = std::random_device{}());

    Tensor forward(const Tensor& x) override;

    /** Switches to training mode: `forward()` randomly zeroes elements (the default). */
    void train();

    /** Switches to inference mode: `forward()` passes `x` through unchanged. */
    void eval();

private:
    float p_;
    std::mt19937 rng_;
    bool training_ = true;
};

}  // namespace advanceml
