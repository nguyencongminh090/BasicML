#pragma once

namespace advanceml {

/**
 * Abstract base for a parameter-update rule, matching
 * `basicml.optim.optimizer.Optimizer`'s contract: an optimizer owns no
 * gradient-computation logic itself, only how each parameter's already
 * accumulated `.grad()` (from `Tensor::backward()`) is applied to and
 * cleared from that parameter.
 */
class Optimizer {
public:
    virtual ~Optimizer();

    /** Applies one update step to every owned parameter that has an accumulated gradient. */
    virtual void step() = 0;

    /** Resets every owned parameter's accumulated gradient (must be called between steps). */
    virtual void zero_grad() = 0;
};

}  // namespace advanceml
