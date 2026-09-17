#include "advanceml/nn/elu.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

ELU::ELU(float alpha) : alpha_(alpha) {}

Tensor ELU::forward(const Tensor& x) {
    return elu(x, alpha_);
}

}  // namespace advanceml
