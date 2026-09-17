#include "advanceml/optim/adagrad.hpp"

#include <cmath>

namespace advanceml {

Adagrad::Adagrad(std::vector<Tensor> parameters, float learning_rate, float eps)
    : parameters_(std::move(parameters)), learning_rate_(learning_rate), eps_(eps) {
    accumulated_sq_.reserve(parameters_.size());
    for (const Tensor& param : parameters_) {
        accumulated_sq_.emplace_back(param.data().size(), 0.0f);
    }
}

void Adagrad::step() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        Tensor& param = parameters_[i];
        if (!param.has_grad()) {
            continue;
        }
        Tensor grad = param.grad();
        FloatBuffer& values = param.mutable_data();
        std::vector<float>& accumulated_sq = accumulated_sq_[i];
        for (size_t j = 0; j < values.size(); ++j) {
            accumulated_sq[j] += grad.data()[j] * grad.data()[j];
            values[j] -= learning_rate_ * grad.data()[j] / (std::sqrt(accumulated_sq[j]) + eps_);
        }
    }
}

void Adagrad::zero_grad() {
    for (Tensor& param : parameters_) {
        param.zero_grad();
    }
}

}  // namespace advanceml
