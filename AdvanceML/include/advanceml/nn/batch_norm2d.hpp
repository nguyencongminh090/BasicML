#pragma once

#include "advanceml/nn/module.hpp"

#include <cstddef>
#include <vector>

namespace advanceml {

/**
 * 2D batch normalization layer over a `(N, C, H, W)` input, normalizing
 * per channel: `y = gamma * (x - mean) / sqrt(var + eps) + beta`.
 *
 * `gamma` starts at one and `beta` at zero, both `requires_grad` leaves
 * owned by this layer; `running_mean`/`running_var` are plain buffers
 * (not part of the autograd graph), initialized to zero/one and updated
 * in place by `forward()` while `train()` mode is active. Backward is
 * handled entirely by the `batch_norm2d` op `forward()` calls -- this
 * class adds no `grad_fn` of its own.
 */
class BatchNorm2D : public Module {
public:
    /** Constructs a layer over `num_channels` channels; `momentum`/`eps` as in `batch_norm2d`. */
    explicit BatchNorm2D(size_t num_channels, float momentum = 0.1f, float eps = 1e-5f);

    Tensor forward(const Tensor& x) override;
    [[nodiscard]] std::vector<Tensor> parameters() const override;

    /** Switches to training mode: `forward()` uses and updates batch statistics (the default). */
    void train();

    /** Switches to inference mode: `forward()` uses the frozen running statistics. */
    void eval();

private:
    Tensor gamma_;
    Tensor beta_;
    std::vector<float> running_mean_;
    std::vector<float> running_var_;
    float momentum_;
    float eps_;
    bool training_ = true;
};

}  // namespace advanceml
