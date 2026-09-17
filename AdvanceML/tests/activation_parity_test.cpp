#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "advanceml/nn/elu.hpp"
#include "advanceml/nn/hardsigmoid.hpp"
#include "advanceml/nn/hardtanh.hpp"
#include "advanceml/nn/identity.hpp"
#include "advanceml/nn/mish.hpp"
#include "advanceml/nn/prelu.hpp"
#include "advanceml/nn/selu.hpp"
#include "advanceml/nn/softplus.hpp"
#include "advanceml/nn/swish.hpp"
#include "advanceml/nn/tanh.hpp"
#include "advanceml/ops.hpp"
#include "advanceml/tensor.hpp"

using namespace advanceml;

namespace {

// Same recipe as nn_test.cpp's numerical_grad/check_activation_gradient,
// kept local to this file per the existing per-test-file convention.
float numerical_grad(float& element, const std::function<float()>& loss_fn, float eps) {
    const float original = element;
    element = original + eps;
    const float loss_plus = loss_fn();
    element = original - eps;
    const float loss_minus = loss_fn();
    element = original;
    return (loss_plus - loss_minus) / (2.0f * eps);
}

void check_activation_gradient(const std::function<Tensor(const Tensor&)>& activation, Tensor x) {
    constexpr float eps = 1e-2f;
    constexpr float tolerance = 5e-2f;

    Tensor y = activation(x);
    Tensor loss = mse_loss(y, Tensor::zeros(y.shape()));
    loss.backward();

    REQUIRE(x.has_grad());
    Tensor grad = x.grad();

    auto loss_fn = [&]() {
        Tensor out = activation(x);
        return mse_loss(out, Tensor::zeros(out.shape())).data()[0];
    };

    for (size_t i = 0; i < x.numel(); ++i) {
        const float analytic = grad.data()[i];
        const float numeric = numerical_grad(x.data()[i], loss_fn, eps);
        REQUIRE(analytic == Catch::Approx(numeric).margin(tolerance));
    }
}

}  // namespace

TEST_CASE("Tanh::forward matches std::tanh and its numerical gradient", "[nn][activation]") {
    Tensor x = Tensor::random_uniform({3, 4}, -2.0f, 2.0f, /*seed=*/501, /*requires_grad=*/true);
    Tensor y = Tanh().forward(x);
    for (size_t i = 0; i < x.numel(); ++i) {
        REQUIRE(y.data()[i] == Catch::Approx(std::tanh(x.data()[i])));
    }
    check_activation_gradient([](const Tensor& t) { return Tanh().forward(t); }, x);
}

TEST_CASE("Identity::forward passes input through unchanged and its backward is the identity", "[nn][activation]") {
    Tensor x = Tensor::random_uniform({3, 4}, -2.0f, 2.0f, /*seed=*/502, /*requires_grad=*/true);
    Tensor y = Identity().forward(x);
    for (size_t i = 0; i < x.numel(); ++i) {
        REQUIRE(y.data()[i] == Catch::Approx(x.data()[i]));
    }
    check_activation_gradient([](const Tensor& t) { return Identity().forward(t); }, x);
}

TEST_CASE("PReLU with a single shared slope matches LeakyReLU-style behavior and its numerical gradient",
          "[nn][activation]") {
    Tensor x = Tensor::random_uniform({3, 4}, -2.0f, 2.0f, /*seed=*/503, /*requires_grad=*/true);
    PReLU layer(/*num_parameters=*/1, /*init=*/0.25f);
    Tensor y = layer.forward(x);
    for (size_t i = 0; i < x.numel(); ++i) {
        const float v = x.data()[i];
        REQUIRE(y.data()[i] == Catch::Approx(v > 0.0f ? v : 0.25f * v));
    }
    check_activation_gradient([&](const Tensor& t) { return layer.forward(t); }, x);

    REQUIRE(layer.parameters().size() == 1);
}

TEST_CASE("PReLU with per-channel slopes broadcasts over the trailing axis and matches its numerical gradient",
          "[nn][activation]") {
    Tensor x = Tensor::random_uniform({5, 3}, -2.0f, 2.0f, /*seed=*/504, /*requires_grad=*/true);
    PReLU layer(/*num_parameters=*/3, /*init=*/0.2f);
    Tensor y = layer.forward(x);
    REQUIRE(y.shape() == x.shape());
    check_activation_gradient([&](const Tensor& t) { return layer.forward(t); }, x);

    Tensor loss = mse_loss(layer.forward(x), Tensor::zeros(x.shape()));
    loss.backward();
    REQUIRE(layer.parameters()[0].has_grad());
}

TEST_CASE("ELU::forward matches its closed form and its numerical gradient", "[nn][activation]") {
    Tensor x = Tensor::random_uniform({3, 4}, -2.0f, 2.0f, /*seed=*/505, /*requires_grad=*/true);
    constexpr float alpha = 1.5f;
    Tensor y = ELU(alpha).forward(x);
    for (size_t i = 0; i < x.numel(); ++i) {
        const float v = x.data()[i];
        const float expected = v > 0.0f ? v : alpha * std::expm1(v);
        REQUIRE(y.data()[i] == Catch::Approx(expected));
    }
    check_activation_gradient([alpha](const Tensor& t) { return ELU(alpha).forward(t); }, x);
}

TEST_CASE("SELU::forward matches its closed form and its numerical gradient", "[nn][activation]") {
    constexpr float kAlpha = 1.6732632423543772f;
    constexpr float kScale = 1.0507009873554805f;
    Tensor x = Tensor::random_uniform({3, 4}, -2.0f, 2.0f, /*seed=*/506, /*requires_grad=*/true);
    Tensor y = SELU().forward(x);
    for (size_t i = 0; i < x.numel(); ++i) {
        const float v = x.data()[i];
        const float expected = kScale * (v > 0.0f ? v : kAlpha * std::expm1(v));
        REQUIRE(y.data()[i] == Catch::Approx(expected));
    }
    check_activation_gradient([](const Tensor& t) { return SELU().forward(t); }, x);
}

TEST_CASE("Softplus::forward matches its closed form and its numerical gradient", "[nn][activation]") {
    Tensor x = Tensor::random_uniform({3, 4}, -2.0f, 2.0f, /*seed=*/507, /*requires_grad=*/true);
    Tensor y = Softplus().forward(x);
    for (size_t i = 0; i < x.numel(); ++i) {
        const float expected = std::log1p(std::exp(x.data()[i]));
        REQUIRE(y.data()[i] == Catch::Approx(expected).margin(1e-4f));
    }
    check_activation_gradient([](const Tensor& t) { return Softplus().forward(t); }, x);
}

TEST_CASE("Swish::forward matches its closed form and its numerical gradient", "[nn][activation]") {
    Tensor x = Tensor::random_uniform({3, 4}, -2.0f, 2.0f, /*seed=*/508, /*requires_grad=*/true);
    constexpr float beta = 1.5f;
    Tensor y = Swish(beta).forward(x);
    for (size_t i = 0; i < x.numel(); ++i) {
        const float v = x.data()[i];
        const float expected = v / (1.0f + std::exp(-beta * v));
        REQUIRE(y.data()[i] == Catch::Approx(expected));
    }
    check_activation_gradient([beta](const Tensor& t) { return Swish(beta).forward(t); }, x);
}

TEST_CASE("Mish::forward matches its closed form and its numerical gradient", "[nn][activation]") {
    Tensor x = Tensor::random_uniform({3, 4}, -2.0f, 2.0f, /*seed=*/509, /*requires_grad=*/true);
    Tensor y = Mish().forward(x);
    for (size_t i = 0; i < x.numel(); ++i) {
        const float v = x.data()[i];
        const float expected = v * std::tanh(std::log1p(std::exp(v)));
        REQUIRE(y.data()[i] == Catch::Approx(expected).margin(1e-4f));
    }
    check_activation_gradient([](const Tensor& t) { return Mish().forward(t); }, x);
}

TEST_CASE("Hardtanh::forward clips to [min_val, max_val] and matches its numerical gradient", "[nn][activation]") {
    Tensor x = Tensor::random_uniform({3, 4}, -2.0f, 2.0f, /*seed=*/510, /*requires_grad=*/true);
    Tensor y = Hardtanh(-1.0f, 1.0f).forward(x);
    for (size_t i = 0; i < x.numel(); ++i) {
        const float expected = std::clamp(x.data()[i], -1.0f, 1.0f);
        REQUIRE(y.data()[i] == Catch::Approx(expected));
    }
    // Values kept away from the clip boundaries (|v| far from 1) so the
    // central-difference check doesn't straddle hardtanh's kinks.
    Tensor x_grad = Tensor::random_uniform({3, 4}, -0.7f, 0.7f, /*seed=*/511, /*requires_grad=*/true);
    check_activation_gradient([](const Tensor& t) { return Hardtanh(-1.0f, 1.0f).forward(t); }, x_grad);
}

TEST_CASE("Hardsigmoid::forward matches its closed form and its numerical gradient", "[nn][activation]") {
    Tensor x = Tensor::random_uniform({3, 4}, -2.0f, 2.0f, /*seed=*/512, /*requires_grad=*/true);
    Tensor y = Hardsigmoid().forward(x);
    for (size_t i = 0; i < x.numel(); ++i) {
        const float expected = std::clamp(x.data()[i] / 6.0f + 0.5f, 0.0f, 1.0f);
        REQUIRE(y.data()[i] == Catch::Approx(expected));
    }
    // Values kept away from the +-3 clip boundaries so the central-difference
    // check doesn't straddle hardsigmoid's kinks.
    Tensor x_grad = Tensor::random_uniform({3, 4}, -2.0f, 2.0f, /*seed=*/513, /*requires_grad=*/true);
    check_activation_gradient([](const Tensor& t) { return Hardsigmoid().forward(t); }, x_grad);
}
