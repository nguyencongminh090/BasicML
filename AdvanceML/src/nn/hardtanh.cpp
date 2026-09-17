#include "advanceml/nn/hardtanh.hpp"

#include "advanceml/ops.hpp"

namespace advanceml {

Hardtanh::Hardtanh(float min_val, float max_val) : min_val_(min_val), max_val_(max_val) {}

Tensor Hardtanh::forward(const Tensor& x) {
    return hardtanh(x, min_val_, max_val_);
}

}  // namespace advanceml
