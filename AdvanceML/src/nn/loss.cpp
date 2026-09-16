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

}  // namespace advanceml
