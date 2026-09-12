#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>
#include <vector>

#include "advanceml/nn/linear.hpp"
#include "advanceml/nn/loss.hpp"
#include "advanceml/nn/relu.hpp"
#include "advanceml/nn/sequential.hpp"
#include "advanceml/optim/adam.hpp"
#include "advanceml/optim/adamw.hpp"
#include "advanceml/optim/momentum.hpp"
#include "advanceml/tensor.hpp"

using namespace advanceml;

namespace {

Tensor single_param(float value) {
    return Tensor({value}, {1}, /*requires_grad=*/true);
}

void backward_with_grad(Tensor& param, float grad) {
    Tensor loss = param * Tensor({grad}, {1});
    loss.backward();
}

}  // namespace

TEST_CASE("Momentum::step accumulates velocity across steps with a constant gradient", "[optim]") {
    Tensor param = single_param(1.0f);
    Momentum optimizer({param}, /*learning_rate=*/0.1f, /*momentum=*/0.9f);

    backward_with_grad(param, 1.0f);
    optimizer.step();
    optimizer.zero_grad();
    // velocity_1 = 0.9 * 0 + 1 = 1; param = 1.0 - 0.1 * 1 = 0.9
    REQUIRE(param.data()[0] == Catch::Approx(0.9f));

    backward_with_grad(param, 1.0f);
    optimizer.step();
    optimizer.zero_grad();
    // velocity_2 = 0.9 * 1 + 1 = 1.9; param = 0.9 - 0.1 * 1.9 = 0.71
    REQUIRE(param.data()[0] == Catch::Approx(0.71f));
}

TEST_CASE("Adam::step matches the closed-form bias-corrected update on the first step", "[optim]") {
    constexpr float lr = 0.1f;
    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.999f;
    constexpr float eps = 1e-8f;
    constexpr float grad_value = 2.0f;

    Tensor param = single_param(1.0f);
    Adam optimizer({param}, lr, beta1, beta2, eps);

    backward_with_grad(param, grad_value);
    optimizer.step();

    // On step 1, Adam's bias correction exactly cancels: m_hat == grad, v_hat == grad^2.
    const float expected = 1.0f - lr * grad_value / (std::abs(grad_value) + eps);

    REQUIRE(param.data()[0] == Catch::Approx(expected).margin(1e-5f));
}

TEST_CASE("AdamW::step applies decoupled weight decay on top of the Adam update", "[optim]") {
    constexpr float lr = 0.1f;
    constexpr float weight_decay = 0.1f;
    constexpr float grad_value = 2.0f;

    Tensor param = single_param(1.0f);
    AdamW optimizer({param}, lr, 0.9f, 0.999f, 1e-8f, weight_decay);

    backward_with_grad(param, grad_value);
    optimizer.step();

    // Same closed-form Adam term as the plain-Adam test, plus the decoupled -lr * weight_decay * param(=1.0) term.
    constexpr float eps = 1e-8f;
    constexpr float initial_param = 1.0f;
    const float adam_term = initial_param - lr * grad_value / (std::abs(grad_value) + eps);
    const float expected = adam_term - lr * weight_decay * initial_param;

    REQUIRE(param.data()[0] == Catch::Approx(expected).margin(1e-5f));
}

TEST_CASE("Sequential/Linear/ReLU/MSELoss/Adam reduces the XOR MLP's loss", "[nn][optim][training]") {
    const std::vector<float> xor_inputs = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
    const std::vector<float> xor_targets = {0.0f, 1.0f, 1.0f, 0.0f};
    Tensor x(xor_inputs, {4, 2});
    Tensor target(xor_targets, {4, 1});

    Sequential model({
        std::make_shared<Linear>(2, 8, /*seed=*/30),
        std::make_shared<ReLU>(),
        std::make_shared<Linear>(8, 1, /*seed=*/31),
    });
    MSELoss loss_fn;
    Adam optimizer(model.parameters(), /*learning_rate=*/0.05f);

    auto forward_loss = [&]() { return loss_fn(model.forward(x), target); };

    const float initial_loss = forward_loss().data()[0];

    float final_loss = initial_loss;
    for (int epoch = 0; epoch < 300; ++epoch) {
        Tensor loss = forward_loss();
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
        final_loss = loss.data()[0];
    }

    REQUIRE(final_loss < initial_loss * 0.5f);
}
