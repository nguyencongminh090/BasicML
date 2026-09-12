#pragma once

#include "advanceml/nn/module.hpp"

#include <memory>
#include <vector>

namespace advanceml {

/**
 * Composite `Module` that chains a list of layers in order: `forward(x)`
 * feeds `x` through each layer in turn, and `parameters()` concatenates
 * every child layer's parameters (in the same order).
 *
 * Layers are owned via `shared_ptr<Module>` so heterogeneous layer types
 * (`Linear`, `ReLU`, another `Sequential`, ...) can share one container.
 */
class Sequential : public Module {
public:
    explicit Sequential(std::vector<std::shared_ptr<Module>> layers);

    Tensor forward(const Tensor& x) override;
    [[nodiscard]] std::vector<Tensor> parameters() const override;

private:
    std::vector<std::shared_ptr<Module>> layers_;
};

}  // namespace advanceml
