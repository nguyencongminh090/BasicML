#include "advanceml/nn/tanh.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

Tensor Tanh::forward(const Tensor& x) {
    return tanh(x);
}

}  // namespace advanceml
