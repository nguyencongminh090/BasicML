#include "advanceml/nn/prelu.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

PReLU::PReLU(size_t num_parameters, float init)
    : a_(Tensor(std::vector<float>(num_parameters, init), {num_parameters}, /*requires_grad=*/true)) {}

Tensor PReLU::forward(const Tensor& x) {
    return prelu(x, a_);
}

std::vector<Tensor> PReLU::parameters() const {
    return {a_};
}

}  // namespace advanceml
