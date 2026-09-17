#include "advanceml/optim/nesterov.hpp"

namespace advanceml {

Nesterov::Nesterov(std::vector<Tensor> parameters, float learning_rate, float momentum)
    : parameters_(std::move(parameters)), learning_rate_(learning_rate), momentum_(momentum) {
    velocities_.reserve(parameters_.size());
    for (const Tensor& param : parameters_) {
        velocities_.emplace_back(param.data().size(), 0.0f);
    }
}

void Nesterov::step() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        Tensor& param = parameters_[i];
        if (!param.has_grad()) {
            continue;
        }
        Tensor grad = param.grad();
        std::vector<float>& values = param.mutable_data();
        std::vector<float>& velocity = velocities_[i];
        for (size_t j = 0; j < values.size(); ++j) {
            const float previous = velocity[j];
            velocity[j] = momentum_ * previous - learning_rate_ * grad.data()[j];
            values[j] += -momentum_ * previous + (1.0f + momentum_) * velocity[j];
        }
    }
}

void Nesterov::zero_grad() {
    for (Tensor& param : parameters_) {
        param.zero_grad();
    }
}

}  // namespace advanceml
