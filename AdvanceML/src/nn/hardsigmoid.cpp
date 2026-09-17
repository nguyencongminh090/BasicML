#include "advanceml/nn/hardsigmoid.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

Tensor Hardsigmoid::forward(const Tensor& x) {
    return hardsigmoid(x);
}

}  // namespace advanceml
