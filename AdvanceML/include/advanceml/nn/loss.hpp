#pragma once

#include "advanceml/tensor.hpp"

namespace advanceml {

/**
 * Abstract base for a loss function: `operator()(pred, target)` computes
 * and returns a scalar `Tensor`. Unlike `basicml.nn.loss.Loss`, no
 * separate `backward()` method is needed here — the returned `Tensor`
 * already carries a `grad_fn` back to `pred` (built by whichever ops
 * the concrete subclass calls), so `.backward()` on the result is
 * sufficient to populate every upstream parameter's `.grad`.
 */
class Loss {
public:
    virtual ~Loss();

    /** @returns The scalar loss `Tensor` for `pred` against `target`. */
    virtual Tensor operator()(const Tensor& pred, const Tensor& target) = 0;
};

/**
 * Mean squared error loss: thin `Loss` wrapper around the `mse_loss` op
 * (`loss = mean((pred - target)^2)`). See `mse_loss` in `ops.hpp` for
 * the forward/backward formulas.
 */
class MSELoss : public Loss {
public:
    Tensor operator()(const Tensor& pred, const Tensor& target) override;
};

}  // namespace advanceml
