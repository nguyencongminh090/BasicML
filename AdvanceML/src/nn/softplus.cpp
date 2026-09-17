#include "advanceml/nn/softplus.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

Tensor Softplus::forward(const Tensor& x) {
    return softplus(x);
}

}  // namespace advanceml
