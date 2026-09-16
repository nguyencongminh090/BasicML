#include "advanceml/nn/batch_norm2d.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

BatchNorm2D::BatchNorm2D(size_t num_channels, float momentum, float eps)
    : gamma_(Tensor(std::vector<float>(num_channels, 1.0f), {num_channels}, /*requires_grad=*/true)),
      beta_(Tensor::zeros({num_channels}, /*requires_grad=*/true)),
      running_mean_(num_channels, 0.0f),
      running_var_(num_channels, 1.0f),
      momentum_(momentum),
      eps_(eps) {}

Tensor BatchNorm2D::forward(const Tensor& x) {
    return batch_norm2d(x, gamma_, beta_, running_mean_, running_var_, training_, momentum_, eps_);
}

std::vector<Tensor> BatchNorm2D::parameters() const {
    return {gamma_, beta_};
}

void BatchNorm2D::train() {
    training_ = true;
}

void BatchNorm2D::eval() {
    training_ = false;
}

}  // namespace advanceml
