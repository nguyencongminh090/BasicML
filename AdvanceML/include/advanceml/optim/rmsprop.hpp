#pragma once

#include "advanceml/optim/optimizer.hpp"
#include "advanceml/tensor.hpp"

#include <vector>

namespace advanceml {

/**
 * RMSProp: per-parameter learning rate scaled by the inverse square root of
 * an exponential moving average of squared gradients:
 * `mean_sq = rho * mean_sq + (1 - rho) * grad^2`;
 * `param -= lr * grad / (sqrt(mean_sq) + eps)`.
 *
 * Conceptually the same shape as `basicml.optim.RMSProp`, not a port of
 * it -- operates directly on `Tensor::grad()`/`Tensor::zero_grad()`.
 */
class RMSprop : public Optimizer {
public:
    RMSprop(std::vector<Tensor> parameters, float learning_rate, float rho = 0.9f, float eps = 1e-8f);

    /** Applies one RMSprop update to every parameter that has an accumulated gradient. */
    void step() override;

    /** Resets every parameter's accumulated gradient (must be called between steps). */
    void zero_grad() override;

private:
    std::vector<Tensor> parameters_;
    float learning_rate_;
    float rho_;
    float eps_;
    std::vector<std::vector<float>> mean_sq_;
};

}  // namespace advanceml
