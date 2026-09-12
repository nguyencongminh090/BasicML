#include "advanceml/nn/gelu.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

Tensor GELU::forward(const Tensor& x) {
    return gelu(x);
}

}  // namespace advanceml
