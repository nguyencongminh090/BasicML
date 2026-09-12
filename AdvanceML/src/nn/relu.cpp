#include "advanceml/nn/relu.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

Tensor ReLU::forward(const Tensor& x) {
    return relu(x);
}

}  // namespace advanceml
