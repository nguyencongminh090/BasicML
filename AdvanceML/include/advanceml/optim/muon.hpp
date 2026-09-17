#pragma once

#include "advanceml/optim/optimizer.hpp"
#include "advanceml/tensor.hpp"

#include <vector>

namespace advanceml {

/**
 * Muon: momentum (optionally Nesterov-style) combined with a Newton-Schulz
 * orthogonalization of the update for every parameter with at least 2
 * dimensions, applied to that parameter reshaped to `(shape[0], numel /
 * shape[0])`:
 * `velocity = momentum * velocity + grad`;
 * `update = momentum * velocity + grad` (Nesterov) or `velocity` otherwise;
 * for 2D+ parameters, `update` is replaced by `scale *
 * newton_schulz5(update)`, `scale = sqrt(max(1, rows / cols))`;
 * `param -= lr * update`.
 *
 * 1D (and 0D) parameters (typically biases) skip orthogonalization and use
 * the raw momentum `update` directly, matching `basicml.optim.Muon`.
 *
 * Conceptually the same shape as `basicml.optim.Muon`, not a port of it --
 * operates directly on `Tensor::grad()`/`Tensor::zero_grad()`.
 */
class Muon : public Optimizer {
public:
    Muon(std::vector<Tensor> parameters, float learning_rate, float momentum = 0.95f, bool nesterov = true,
         int newton_schulz_steps = 5, float eps = 1e-7f);

    /** Applies one Muon update to every parameter that has an accumulated gradient. */
    void step() override;

    /** Resets every parameter's accumulated gradient (must be called between steps). */
    void zero_grad() override;

private:
    std::vector<Tensor> parameters_;
    float learning_rate_;
    float momentum_;
    bool nesterov_;
    int newton_schulz_steps_;
    float eps_;
    std::vector<std::vector<float>> velocities_;
};

}  // namespace advanceml
