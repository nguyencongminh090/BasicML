#include "advanceml/nn/max_pool2d.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

MaxPool2D::MaxPool2D(size_t kernel_size, size_t stride) : kernel_size_(kernel_size), stride_(stride) {}

Tensor MaxPool2D::forward(const Tensor& x) {
    return max_pool2d(x, kernel_size_, stride_);
}

}  // namespace advanceml
