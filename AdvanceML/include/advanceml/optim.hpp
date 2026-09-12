#pragma once

#include "advanceml/tensor.hpp"

#include <vector>

namespace advanceml {

/**
 * Plain (non-momentum) gradient descent: `param -= lr * param.grad()`.
 *
 * Conceptually the same shape as `basicml.optim.SGD`, not a port of
 * it -- operates directly on `Tensor::grad()`/`Tensor::zero_grad()`.
 */
class SGD {
public:
    SGD(std::vector<Tensor> parameters, float learning_rate);

    /** Applies one gradient-descent step to every parameter that has an accumulated gradient. */
    void step();

    /** Resets every parameter's accumulated gradient (must be called between steps). */
    void zero_grad();

private:
    std::vector<Tensor> parameters_;
    float learning_rate_;
};

}  // namespace advanceml
