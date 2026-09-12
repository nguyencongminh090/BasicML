#pragma once

#include "advanceml/optim/optimizer.hpp"
#include "advanceml/tensor.hpp"

#include <vector>

namespace advanceml {

/**
 * Adam with decoupled weight decay: the same first/second moment
 * update as `Adam`, but weight decay is applied directly to the
 * parameter rather than folded into the gradient (Loshchilov &
 * Hutter, 2019) --
 * `param -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)`.
 *
 * Conceptually the same shape as `basicml.optim.AdamW`, not a port
 * of it -- operates directly on `Tensor::grad()`/`Tensor::zero_grad()`.
 */
class AdamW : public Optimizer {
public:
    AdamW(std::vector<Tensor> parameters,
          float learning_rate,
          float beta1 = 0.9f,
          float beta2 = 0.999f,
          float eps = 1e-8f,
          float weight_decay = 0.01f);

    /** Applies one AdamW update to every parameter that has an accumulated gradient. */
    void step() override;

    /** Resets every parameter's accumulated gradient (must be called between steps). */
    void zero_grad() override;

private:
    std::vector<Tensor> parameters_;
    float learning_rate_;
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    int step_count_ = 0;
    std::vector<std::vector<float>> first_moment_;
    std::vector<std::vector<float>> second_moment_;
};

}  // namespace advanceml
