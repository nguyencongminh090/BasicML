#include "advanceml/optim/adadelta.hpp"

#include <cmath>

namespace advanceml {

Adadelta::Adadelta(std::vector<Tensor> parameters, float learning_rate, float rho, float eps)
    : parameters_(std::move(parameters)), learning_rate_(learning_rate), rho_(rho), eps_(eps) {
    mean_sq_grad_.reserve(parameters_.size());
    mean_sq_step_.reserve(parameters_.size());
    for (const Tensor& param : parameters_) {
        mean_sq_grad_.emplace_back(param.data().size(), 0.0f);
        mean_sq_step_.emplace_back(param.data().size(), 0.0f);
    }
}

void Adadelta::step() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        Tensor& param = parameters_[i];
        if (!param.has_grad()) {
            continue;
        }
        Tensor grad = param.grad();
        std::vector<float>& values = param.mutable_data();
        std::vector<float>& mean_sq_grad = mean_sq_grad_[i];
        std::vector<float>& mean_sq_step = mean_sq_step_[i];
        for (size_t j = 0; j < values.size(); ++j) {
            const float g = grad.data()[j];
            mean_sq_grad[j] = rho_ * mean_sq_grad[j] + (1.0f - rho_) * g * g;
            const float delta = std::sqrt(mean_sq_step[j] + eps_) / std::sqrt(mean_sq_grad[j] + eps_) * g;
            mean_sq_step[j] = rho_ * mean_sq_step[j] + (1.0f - rho_) * delta * delta;
            values[j] -= learning_rate_ * delta;
        }
    }
}

void Adadelta::zero_grad() {
    for (Tensor& param : parameters_) {
        param.zero_grad();
    }
}

}  // namespace advanceml
