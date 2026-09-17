#include "advanceml/nn/mish.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

Tensor Mish::forward(const Tensor& x) {
    return mish(x);
}

}  // namespace advanceml
