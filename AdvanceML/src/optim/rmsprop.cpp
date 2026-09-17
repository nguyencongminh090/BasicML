#include "advanceml/optim/rmsprop.hpp"

#include <cmath>

namespace advanceml {

RMSprop::RMSprop(std::vector<Tensor> parameters, float learning_rate, float rho, float eps)
    : parameters_(std::move(parameters)), learning_rate_(learning_rate), rho_(rho), eps_(eps) {
    mean_sq_.reserve(parameters_.size());
    for (const Tensor& param : parameters_) {
        mean_sq_.emplace_back(param.data().size(), 0.0f);
    }
}

void RMSprop::step() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        Tensor& param = parameters_[i];
        if (!param.has_grad()) {
            continue;
        }
        Tensor grad = param.grad();
        std::vector<float>& mean_sq = mean_sq_[i];
        for (size_t j = 0; j < param.data().size(); ++j) {
            const float g = grad.data()[j];
            mean_sq[j] = rho_ * mean_sq[j] + (1.0f - rho_) * g * g;
            param.data()[j] -= learning_rate_ * g / (std::sqrt(mean_sq[j]) + eps_);
        }
    }
}

void RMSprop::zero_grad() {
    for (Tensor& param : parameters_) {
        param.zero_grad();
    }
}

}  // namespace advanceml
