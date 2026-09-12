#include "advanceml/nn/leaky_relu.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

LeakyReLU::LeakyReLU(float negative_slope) : negative_slope_(negative_slope) {}

Tensor LeakyReLU::forward(const Tensor& x) {
    return leaky_relu(x, negative_slope_);
}

}  // namespace advanceml
