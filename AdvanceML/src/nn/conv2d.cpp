#include "advanceml/nn/conv2d.hpp"

#include "advanceml/ops.hpp"

#include <cmath>

namespace advanceml {

Conv2D::Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride, size_t padding,
               unsigned seed)
    : weight_(Tensor::random_uniform(
          {out_channels, in_channels, kernel_size, kernel_size},
          -std::sqrt(6.0f / static_cast<float>((in_channels + out_channels) * kernel_size * kernel_size)),
          std::sqrt(6.0f / static_cast<float>((in_channels + out_channels) * kernel_size * kernel_size)), seed,
          /*requires_grad=*/true)),
      bias_(Tensor::zeros({out_channels}, /*requires_grad=*/true)),
      stride_(stride),
      padding_(padding) {}

Tensor Conv2D::forward(const Tensor& x) {
    return conv2d(x, weight_, bias_, stride_, padding_);
}

std::vector<Tensor> Conv2D::parameters() const {
    return {weight_, bias_};
}

}  // namespace advanceml
