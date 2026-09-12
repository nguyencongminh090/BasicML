#pragma once

#include "advanceml/optim/optimizer.hpp"
#include "advanceml/tensor.hpp"

#include <vector>

namespace advanceml {

/**
 * Gradient descent with a per-parameter velocity buffer:
 * `velocity = momentum * velocity + grad; param -= lr * velocity`.
 *
 * Conceptually the same shape as `basicml.optim.Momentum`, not a
 * port of it -- operates directly on `Tensor::grad()`/`Tensor::zero_grad()`.
 */
class Momentum : public Optimizer {
public:
    Momentum(std::vector<Tensor> parameters, float learning_rate, float momentum = 0.9f);

    /** Applies one momentum-accumulated update to every parameter that has an accumulated gradient. */
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
