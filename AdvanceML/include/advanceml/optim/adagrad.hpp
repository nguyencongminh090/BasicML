#pragma once

#include "advanceml/optim/optimizer.hpp"
#include "advanceml/tensor.hpp"

#include <vector>

namespace advanceml {

/**
 * Adagrad: per-parameter learning rate scaled by the inverse square root
 * of the running sum of squared gradients:
 * `accumulated_sq += grad^2; param -= lr * grad / (sqrt(accumulated_sq) + eps)`.
 *
 * Conceptually the same shape as `basicml.optim.Adagrad`, not a port of
 * it -- operates directly on `Tensor::grad()`/`Tensor::zero_grad()`.
 */
class Adagrad : public Optimizer {
public:
    Adagrad(std::vector<Tensor> parameters, float learning_rate, float eps = 1e-8f);

    /** Applies one Adagrad update to every parameter that has an accumulated gradient. */
    void step() override;

    /** Resets every parameter's accumulated gradient (must be called between steps). */
    void zero_grad() override;

private:
    std::vector<Tensor> parameters_;
    float learning_rate_;
    float eps_;
    std::vector<std::vector<float>> accumulated_sq_;
};

}  // namespace advanceml
