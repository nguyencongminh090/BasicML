#pragma once

#include "advanceml/optim/optimizer.hpp"
#include "advanceml/tensor.hpp"

#include <vector>

namespace advanceml {

/**
 * Adadelta: a learning-rate-free variant of Adagrad that replaces the raw
 * gradient accumulator with an exponential moving average and derives the
 * step size from a matching moving average of past update magnitudes:
 * `mean_sq_grad = rho * mean_sq_grad + (1 - rho) * grad^2`;
 * `delta = sqrt(mean_sq_step + eps) / sqrt(mean_sq_grad + eps) * grad`;
 * `mean_sq_step = rho * mean_sq_step + (1 - rho) * delta^2`;
 * `param -= lr * delta`.
 *
 * Conceptually the same shape as `basicml.optim.Adadelta`, not a port of
 * it -- operates directly on `Tensor::grad()`/`Tensor::zero_grad()`.
 */
class Adadelta : public Optimizer {
public:
    Adadelta(std::vector<Tensor> parameters, float learning_rate = 1.0f, float rho = 0.95f, float eps = 1e-6f);

    /** Applies one Adadelta update to every parameter that has an accumulated gradient. */
    void step() override;

    /** Resets every parameter's accumulated gradient (must be called between steps). */
    void zero_grad() override;

private:
    std::vector<Tensor> parameters_;
    float learning_rate_;
    float rho_;
    float eps_;
    std::vector<std::vector<float>> mean_sq_grad_;
    std::vector<std::vector<float>> mean_sq_step_;
};

}  // namespace advanceml
