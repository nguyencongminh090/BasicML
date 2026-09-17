#include "advanceml/nn/global_max_pool2d.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

Tensor GlobalMaxPool2D::forward(const Tensor& x) {
    return global_max_pool2d(x);
}

}  // namespace advanceml
