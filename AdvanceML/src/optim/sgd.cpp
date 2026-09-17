#include "advanceml/optim/sgd.hpp"

namespace advanceml {

SGD::SGD(std::vector<Tensor> parameters, float learning_rate)
    : parameters_(std::move(parameters)), learning_rate_(learning_rate) {}

void SGD::step() {
    for (Tensor& param : parameters_) {
        if (!param.has_grad()) {
            continue;
        }
        Tensor grad = param.grad();
        FloatBuffer& values = param.mutable_data();
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] -= learning_rate_ * grad.data()[i];
        }
    }
}

void SGD::zero_grad() {
    for (Tensor& param : parameters_) {
        param.zero_grad();
    }
}

}  // namespace advanceml
