#include "advanceml/nn/dropout.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

Dropout::Dropout(float p, unsigned seed) : p_(p), rng_(seed) {}

Tensor Dropout::forward(const Tensor& x) {
    return dropout(x, p_, training_, rng_);
}

void Dropout::train() {
    training_ = true;
}

void Dropout::eval() {
    training_ = false;
}

}  // namespace advanceml
