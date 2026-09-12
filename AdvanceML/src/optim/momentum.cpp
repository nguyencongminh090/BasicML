#include "advanceml/optim/momentum.hpp"

namespace advanceml {

Momentum::Momentum(std::vector<Tensor> parameters, float learning_rate, float momentum)
    : parameters_(std::move(parameters)), learning_rate_(learning_rate), momentum_(momentum) {
    velocities_.reserve(parameters_.size());
    for (const Tensor& param : parameters_) {
        velocities_.emplace_back(param.data().size(), 0.0f);
    }
}

void Momentum::step() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        Tensor& param = parameters_[i];
        if (!param.has_grad()) {
            continue;
        }
        Tensor grad = param.grad();
        std::vector<float>& velocity = velocities_[i];
        for (size_t j = 0; j < param.data().size(); ++j) {
            velocity[j] = momentum_ * velocity[j] + grad.data()[j];
            param.data()[j] -= learning_rate_ * velocity[j];
        }
    }
}

void Momentum::zero_grad() {
    for (Tensor& param : parameters_) {
        param.zero_grad();
    }
}

}  // namespace advanceml
