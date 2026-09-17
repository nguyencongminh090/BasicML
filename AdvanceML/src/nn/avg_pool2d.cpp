#include "advanceml/nn/avg_pool2d.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

AvgPool2D::AvgPool2D(size_t kernel_size, size_t stride) : kernel_size_(kernel_size), stride_(stride) {}

Tensor AvgPool2D::forward(const Tensor& x) {
    return avg_pool2d(x, kernel_size_, stride_);
}

}  // namespace advanceml
