#include "advanceml/nn/selu.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

Tensor SELU::forward(const Tensor& x) {
    return selu(x);
}

}  // namespace advanceml
