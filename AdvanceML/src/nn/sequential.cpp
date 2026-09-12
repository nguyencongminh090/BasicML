#include "advanceml/nn/sequential.hpp"

#include <utility>

namespace advanceml {

Sequential::Sequential(std::vector<std::shared_ptr<Module>> layers) : layers_(std::move(layers)) {}

Tensor Sequential::forward(const Tensor& x) {
    Tensor out = x;
    for (const std::shared_ptr<Module>& layer : layers_) {
        out = layer->forward(out);
    }
    return out;
}

std::vector<Tensor> Sequential::parameters() const {
    std::vector<Tensor> all_parameters;
    for (const std::shared_ptr<Module>& layer : layers_) {
        std::vector<Tensor> layer_parameters = layer->parameters();
        all_parameters.insert(all_parameters.end(), layer_parameters.begin(), layer_parameters.end());
    }
    return all_parameters;
}

}  // namespace advanceml
