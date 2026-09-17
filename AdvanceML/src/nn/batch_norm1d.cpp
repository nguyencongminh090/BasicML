#include "advanceml/nn/batch_norm1d.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

BatchNorm1D::BatchNorm1D(size_t num_features, float momentum, float eps)
    : gamma_(Tensor(std::vector<float>(num_features, 1.0f), {num_features}, /*requires_grad=*/true)),
      beta_(Tensor::zeros({num_features}, /*requires_grad=*/true)),
      running_mean_(num_features, 0.0f),
      running_var_(num_features, 1.0f),
      momentum_(momentum),
      eps_(eps) {}

Tensor BatchNorm1D::forward(const Tensor& x) {
    return batch_norm1d(x, gamma_, beta_, running_mean_, running_var_, training_, momentum_, eps_);
}

std::vector<Tensor> BatchNorm1D::parameters() const {
    return {gamma_, beta_};
}

void BatchNorm1D::train() {
    training_ = true;
}

void BatchNorm1D::eval() {
    training_ = false;
}

}  // namespace advanceml
