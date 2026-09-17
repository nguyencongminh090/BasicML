#include "advanceml/nn/flatten.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

Tensor Flatten::forward(const Tensor& x) {
    return flatten(x);
}

}  // namespace advanceml
