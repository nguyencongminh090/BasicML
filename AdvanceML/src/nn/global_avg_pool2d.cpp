#include "advanceml/nn/global_avg_pool2d.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

Tensor GlobalAvgPool2D::forward(const Tensor& x) {
    return global_avg_pool2d(x);
}

}  // namespace advanceml
