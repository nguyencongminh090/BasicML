#include "advanceml/nn/softmax.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

Tensor Softmax::forward(const Tensor& x) {
    return softmax(x);
}

}  // namespace advanceml
