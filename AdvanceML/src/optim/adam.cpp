#include "advanceml/optim/adam.hpp"

#include <cmath>

namespace advanceml {

Adam::Adam(std::vector<Tensor> parameters, float learning_rate, float beta1, float beta2, float eps)
    : parameters_(std::move(parameters)), learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), eps_(eps) {
    first_moment_.reserve(parameters_.size());
    second_moment_.reserve(parameters_.size());
    for (const Tensor& param : parameters_) {
        first_moment_.emplace_back(param.data().size(), 0.0f);
        second_moment_.emplace_back(param.data().size(), 0.0f);
    }
}

void Adam::step() {
    ++step_count_;
    const float bias_correction1 = 1.0f - std::pow(beta1_, step_count_);
    const float bias_correction2 = 1.0f - std::pow(beta2_, step_count_);

    for (size_t i = 0; i < parameters_.size(); ++i) {
        Tensor& param = parameters_[i];
        if (!param.has_grad()) {
            continue;
        }
        Tensor grad = param.grad();
        FloatBuffer& values = param.mutable_data();
        std::vector<float>& m = first_moment_[i];
        std::vector<float>& v = second_moment_[i];
        for (size_t j = 0; j < values.size(); ++j) {
            m[j] = beta1_ * m[j] + (1.0f - beta1_) * grad.data()[j];
            v[j] = beta2_ * v[j] + (1.0f - beta2_) * grad.data()[j] * grad.data()[j];
            const float m_hat = m[j] / bias_correction1;
            const float v_hat = v[j] / bias_correction2;
            values[j] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}

void Adam::zero_grad() {
    for (Tensor& param : parameters_) {
        param.zero_grad();
    }
}

}  // namespace advanceml
