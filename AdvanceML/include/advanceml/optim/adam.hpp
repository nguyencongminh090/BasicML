#pragma once

#include "advanceml/optim/optimizer.hpp"
#include "advanceml/tensor.hpp"

#include <vector>

namespace advanceml {

/**
 * Adam: per-parameter first/second raw-gradient moment estimates with
 * bias correction, `param -= lr * m_hat / (sqrt(v_hat) + eps)`.
 *
 * Conceptually the same shape as `basicml.optim.Adam`, not a port of
 * it -- operates directly on `Tensor::grad()`/`Tensor::zero_grad()`.
 */
class Adam : public Optimizer {
public:
    Adam(std::vector<Tensor> parameters,
         float learning_rate,
         float beta1 = 0.9f,
         float beta2 = 0.999f,
         float eps = 1e-8f);

    /** Applies one Adam update to every parameter that has an accumulated gradient. */
    void step() override;

    /** Resets every parameter's accumulated gradient (must be called between steps). */
    void zero_grad() override;

private:
    std::vector<Tensor> parameters_;
    float learning_rate_;
    float beta1_;
    float beta2_;
    float eps_;
    int step_count_ = 0;
    std::vector<std::vector<float>> first_moment_;
    std::vector<std::vector<float>> second_moment_;
};

}  // namespace advanceml
