#include "advanceml/nn/identity.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

Tensor Identity::forward(const Tensor& x) {
    return identity(x);
}

}  // namespace advanceml
