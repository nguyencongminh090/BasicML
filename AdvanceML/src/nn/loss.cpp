#include "advanceml/nn/loss.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

Loss::~Loss() = default;

Tensor MSELoss::operator()(const Tensor& pred, const Tensor& target) {
    return mse_loss(pred, target);
}

}  // namespace advanceml
