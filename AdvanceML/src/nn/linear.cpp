#include "advanceml/nn/linear.hpp"

#include "advanceml/ops.hpp"

#include <cmath>

namespace advanceml {

Linear::Linear(size_t in_features, size_t out_features, unsigned seed)
    : w_(Tensor::random_uniform(
          {in_features, out_features},
          -std::sqrt(6.0f / static_cast<float>(in_features + out_features)),
          std::sqrt(6.0f / static_cast<float>(in_features + out_features)),
          seed,
          /*requires_grad=*/true)),
      b_(Tensor::zeros({out_features}, /*requires_grad=*/true)) {}

Tensor Linear::forward(const Tensor& x) {
    return linear(x, w_, b_);
}

std::vector<Tensor> Linear::parameters() const {
    return {w_, b_};
}

}  // namespace advanceml
