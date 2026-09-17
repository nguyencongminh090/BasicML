#pragma once

#include "advanceml/nn/module.hpp"

#include <cstddef>

namespace advanceml {

/**
 * Affine layer: `y = x @ w + b`, with `w` shaped `(in_features,
 * out_features)` and `b` shaped `(out_features)`.
 *
 * `w` is Xavier-uniform initialized (`limit = sqrt(6 / (in_features +
 * out_features))`, drawn from `[-limit, limit)`); `b` starts at zero.
 * Both are `requires_grad` leaves owned by this layer. Backward is
 * handled entirely by the fused `linear` op `forward()` calls
 * — this class adds no `grad_fn` of its own.
 */
class Linear : public Module {
public:
    /** Constructs a `(in_features, out_features)` layer; `seed` drives `w`'s init draw. */
    Linear(size_t in_features, size_t out_features, unsigned seed);

    Tensor forward(const Tensor& x) override;
    [[nodiscard]] std::vector<Tensor> parameters() const override;

private:
    Tensor w_;
    Tensor b_;
};

}  // namespace advanceml
