#include "advanceml/nn/loss.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

Loss::~Loss() = default;

Tensor MSELoss::operator()(const Tensor& pred, const Tensor& target) {
    return mse_loss(pred, target);
}

Tensor CrossEntropyLoss::operator()(const Tensor& pred, const Tensor& target) {
    return cross_entropy_loss(pred, target);
}

Tensor SoftmaxCrossEntropyLoss::operator()(const Tensor& pred, const Tensor& target) {
    return softmax_cross_entropy(pred, target);
}

Tensor AbsoluteLoss::operator()(const Tensor& pred, const Tensor& target) {
    return abs_loss(pred, target);
}

Tensor BinaryCrossEntropy::operator()(const Tensor& pred, const Tensor& target) {
    return binary_cross_entropy(pred, target);
}

}  // namespace advanceml
