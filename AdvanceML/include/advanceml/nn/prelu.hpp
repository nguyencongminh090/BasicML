#pragma once

#include "advanceml/nn/module.hpp"

#include <cstddef>
#include <vector>

namespace advanceml {

/**
 * Parametric rectified linear unit layer: `forward(x) = prelu(x, a)`,
 * where `a` is a learned per-channel (or single shared, when
 * `num_parameters == 1`) negative-side slope, broadcast over `x`'s
 * trailing axis. `a` starts at `init` and is a `requires_grad` leaf owned
 * by this layer; backward is handled entirely by the `prelu` op.
 */
class PReLU : public Module {
public:
    /** Constructs a layer with `num_parameters` slopes (1 for a single shared slope), each initialized to `init`. */
    explicit PReLU(size_t num_parameters = 1, float init = 0.25f);

    Tensor forward(const Tensor& x) override;
    [[nodiscard]] std::vector<Tensor> parameters() const override;

private:
    Tensor a_;
};

}  // namespace advanceml
