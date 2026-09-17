#include "advanceml/nn/swish.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

Swish::Swish(float beta) : beta_(beta) {}

Tensor Swish::forward(const Tensor& x) {
    return swish(x, beta_);
}

}  // namespace advanceml
