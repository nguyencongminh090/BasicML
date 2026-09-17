#include "advanceml/optim/adamw.hpp"

#include <cmath>

namespace advanceml {

AdamW::AdamW(std::vector<Tensor> parameters,
             float learning_rate,
             float beta1,
             float beta2,
             float eps,
             float weight_decay)
    : parameters_(std::move(parameters)),
      learning_rate_(learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      weight_decay_(weight_decay) {
    first_moment_.reserve(parameters_.size());
    second_moment_.reserve(parameters_.size());
    for (const Tensor& param : parameters_) {
        first_moment_.emplace_back(param.data().size(), 0.0f);
        second_moment_.emplace_back(param.data().size(), 0.0f);
    }
}

void AdamW::step() {
    ++step_count_;
    const float bias_correction1 = 1.0f - std::pow(beta1_, step_count_);
    const float bias_correction2 = 1.0f - std::pow(beta2_, step_count_);

    for (size_t i = 0; i < parameters_.size(); ++i) {
        Tensor& param = parameters_[i];
        if (!param.has_grad()) {
            continue;
        }
        Tensor grad = param.grad();
        // Raw pointers and loop-local constants, so the loop has no calls or aliasing through members
        // and vectorizes (sqrt included, given -fno-math-errno).
        const float* const g = grad.data().data();
        FloatBuffer& param_values = param.mutable_data();
        float* const values = param_values.data();
        float* const m = first_moment_[i].data();
        float* const v = second_moment_[i].data();
        const size_t n = param_values.size();
        const float beta1 = beta1_;
        const float beta2 = beta2_;
        const float learning_rate = learning_rate_;
        const float eps = eps_;
        const float weight_decay = weight_decay_;
        for (size_t j = 0; j < n; ++j) {
            m[j] = beta1 * m[j] + (1.0f - beta1) * g[j];
            v[j] = beta2 * v[j] + (1.0f - beta2) * g[j] * g[j];
            const float m_hat = m[j] / bias_correction1;
            const float v_hat = v[j] / bias_correction2;
            values[j] -= learning_rate * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * values[j]);
        }
    }
}

void AdamW::zero_grad() {
    for (Tensor& param : parameters_) {
        param.zero_grad();
    }
}

}  // namespace advanceml
