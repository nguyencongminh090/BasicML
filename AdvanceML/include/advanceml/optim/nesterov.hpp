#pragma once

#include "advanceml/optim/optimizer.hpp"
#include "advanceml/tensor.hpp"

#include <vector>

namespace advanceml {

/**
 * Nesterov accelerated gradient: `velocity = momentum * velocity_prev -
 * lr * grad; param += -momentum * velocity_prev + (1 + momentum) * velocity`
 * -- the "lookahead" update rearranged to avoid needing a separate
 * lookahead evaluation point.
 *
 * Conceptually the same shape as `basicml.optim.Nesterov`, not a port of
 * it -- operates directly on `Tensor::grad()`/`Tensor::zero_grad()`.
 */
class Nesterov : public Optimizer {
public:
    Nesterov(std::vector<Tensor> parameters, float learning_rate, float momentum = 0.9f);

    /** Applies one Nesterov-accelerated update to every parameter that has an accumulated gradient. */
    void step() override;

    /** Resets every parameter's accumulated gradient (must be called between steps). */
    void zero_grad() override;

private:
    std::vector<Tensor> parameters_;
    float learning_rate_;
    float momentum_;
    std::vector<std::vector<float>> velocities_;
};

}  // namespace advanceml
