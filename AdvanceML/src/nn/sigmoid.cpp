#include "advanceml/nn/sigmoid.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

Tensor Sigmoid::forward(const Tensor& x) {
    return sigmoid(x);
}

}  // namespace advanceml
