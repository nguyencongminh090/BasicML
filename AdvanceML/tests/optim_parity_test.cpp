#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include "advanceml/optim/adadelta.hpp"
#include "advanceml/optim/adagrad.hpp"
#include "advanceml/optim/muon.hpp"
#include "advanceml/optim/nesterov.hpp"
#include "advanceml/optim/rmsprop.hpp"
#include "advanceml/ops.hpp"
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

TEST_CASE("Nesterov::step matches the closed-form update on the first step", "[optim]") {
    constexpr float lr = 0.1f;
    constexpr float momentum = 0.9f;
    constexpr float grad_value = 2.0f;

    Tensor param = single_param(1.0f);
    Nesterov optimizer({param}, lr, momentum);

    backward_with_grad(param, grad_value);
    optimizer.step();

    // velocity_1 = momentum * 0 - lr * grad = -lr * grad
    // param += -momentum * 0 + (1 + momentum) * velocity_1
    const float velocity = -lr * grad_value;
    const float expected = 1.0f + (1.0f + momentum) * velocity;
    REQUIRE(param.data()[0] == Catch::Approx(expected));
}

TEST_CASE("Adagrad::step matches the closed-form update across two steps", "[optim]") {
    constexpr float lr = 0.5f;
    constexpr float eps = 1e-8f;
    constexpr float grad_value = 2.0f;

    Tensor param = single_param(1.0f);
    Adagrad optimizer({param}, lr, eps);

    backward_with_grad(param, grad_value);
    optimizer.step();
    optimizer.zero_grad();
    float accumulated_sq = grad_value * grad_value;
    float expected = 1.0f - lr * grad_value / (std::sqrt(accumulated_sq) + eps);
    REQUIRE(param.data()[0] == Catch::Approx(expected).margin(1e-5f));

    backward_with_grad(param, grad_value);
    optimizer.step();
    optimizer.zero_grad();
    accumulated_sq += grad_value * grad_value;
    expected -= lr * grad_value / (std::sqrt(accumulated_sq) + eps);
    REQUIRE(param.data()[0] == Catch::Approx(expected).margin(1e-5f));
}

TEST_CASE("Adadelta::step matches the closed-form update on the first step", "[optim]") {
    constexpr float lr = 1.0f;
    constexpr float rho = 0.95f;
    constexpr float eps = 1e-6f;
    constexpr float grad_value = 2.0f;

    Tensor param = single_param(1.0f);
    Adadelta optimizer({param}, lr, rho, eps);

    backward_with_grad(param, grad_value);
    optimizer.step();

    const float mean_sq_grad = (1.0f - rho) * grad_value * grad_value;
    const float delta = std::sqrt(0.0f + eps) / std::sqrt(mean_sq_grad + eps) * grad_value;
    const float expected = 1.0f - lr * delta;
    REQUIRE(param.data()[0] == Catch::Approx(expected).margin(1e-5f));
}

TEST_CASE("RMSprop::step matches the closed-form update on the first step", "[optim]") {
    constexpr float lr = 0.1f;
    constexpr float rho = 0.9f;
    constexpr float eps = 1e-8f;
    constexpr float grad_value = 2.0f;

    Tensor param = single_param(1.0f);
    RMSprop optimizer({param}, lr, rho, eps);

    backward_with_grad(param, grad_value);
    optimizer.step();

    const float mean_sq = (1.0f - rho) * grad_value * grad_value;
    const float expected = 1.0f - lr * grad_value / (std::sqrt(mean_sq) + eps);
    REQUIRE(param.data()[0] == Catch::Approx(expected).margin(1e-5f));
}

TEST_CASE("Muon::step on a 1D parameter matches plain Nesterov-momentum (no orthogonalization)", "[optim]") {
    constexpr float lr = 0.1f;
    constexpr float momentum = 0.95f;
    constexpr float grad_value = 2.0f;

    Tensor param = single_param(1.0f);
    Muon optimizer({param}, lr, momentum, /*nesterov=*/true);

    backward_with_grad(param, grad_value);
    optimizer.step();

    const float velocity = momentum * 0.0f + grad_value;
    const float update = momentum * velocity + grad_value;
    const float expected = 1.0f - lr * update;
    REQUIRE(param.data()[0] == Catch::Approx(expected).margin(1e-5f));
}

TEST_CASE("Muon::step's Newton-Schulz orthogonalization sharply reduces a 2D gradient's singular-value spread",
          "[optim]") {
    // A diagonal, highly anisotropic 2x2 gradient (singular values 3 and
    // 0.5, ratio 6): 5 Newton-Schulz steps only approximate the orthogonal
    // polar factor (a documented property of Muon's fixed step count, not
    // exact convergence to singular values of exactly 1), but should still
    // collapse the squared-singular-value ratio from 36 down to near 1,
    // and -- since the input is diagonal -- leave the result diagonal too.
    const std::vector<float> initial = {1.0f, 0.0f, 0.0f, 1.0f};
    Tensor param(initial, {2, 2}, /*requires_grad=*/true);
    Muon optimizer({param}, /*learning_rate=*/1.0f, /*momentum=*/0.0f, /*nesterov=*/false);

    // mse_loss(param, target) has dL/dparam = 2*(param - target)/N; picking
    // target = param - manual_grad * N/2 makes that gradient exactly
    // manual_grad, without needing a dedicated sum-reduction op.
    const std::vector<float> manual_grad = {3.0f, 0.0f, 0.0f, 0.5f};
    constexpr float n = 4.0f;
    std::vector<float> target_data(4);
    for (size_t i = 0; i < 4; ++i) {
        target_data[i] = initial[i] - manual_grad[i] * (n / 2.0f);
    }
    Tensor target(target_data, {2, 2});
    Tensor loss = mse_loss(param, target);
    loss.backward();

    REQUIRE(param.has_grad());
    for (size_t i = 0; i < 4; ++i) {
        REQUIRE(param.grad().data()[i] == Catch::Approx(manual_grad[i]).margin(1e-5f));
    }

    optimizer.step();

    // lr = 1, momentum = 0 => update = grad, scale = 1 => new_param = initial - ortho.
    std::vector<float> ortho(4);
    for (size_t i = 0; i < 4; ++i) {
        ortho[i] = initial[i] - param.data()[i];
    }
    // The input gradient is diagonal, so ortho @ ortho^T should stay
    // (near-)diagonal too, regardless of how close to 1 the diagonal ends up.
    const float gram_00 = ortho[0] * ortho[0] + ortho[1] * ortho[1];
    const float gram_01 = ortho[0] * ortho[2] + ortho[1] * ortho[3];
    const float gram_11 = ortho[2] * ortho[2] + ortho[3] * ortho[3];
    REQUIRE(gram_01 == Catch::Approx(0.0f).margin(1e-4f));

    // Squared-singular-value ratio: 36 before orthogonalization, should
    // collapse to well under an order of magnitude after 5 NS steps.
    const float pre_ratio = (manual_grad[0] * manual_grad[0]) / (manual_grad[3] * manual_grad[3]);
    const float post_ratio = std::max(gram_00, gram_11) / std::min(gram_00, gram_11);
    REQUIRE(pre_ratio == Catch::Approx(36.0f).margin(1e-3f));
    REQUIRE(post_ratio < 5.0f);
}
