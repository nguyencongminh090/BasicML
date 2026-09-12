#include "advanceml/nn/module.hpp"

namespace advanceml {

std::vector<Tensor> Module::parameters() const {
    return {};
}

Tensor Module::operator()(const Tensor& x) {
    return forward(x);
}

}  // namespace advanceml
