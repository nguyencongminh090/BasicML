#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <functional>
#include <vector>

#include "advanceml/nn/loss.hpp"
#include "advanceml/tensor.hpp"

using namespace advanceml;

namespace {

// Same recipe as nn_test.cpp's numerical_grad/check_loss_gradient, kept
// local to this file per the existing per-test-file convention.
float numerical_grad(float& element, const std::function<float()>& loss_fn, float eps) {
    const float original = element;
    element = original + eps;
    const float loss_plus = loss_fn();
    element = original - eps;
    const float loss_minus = loss_fn();
    element = original;
    return (loss_plus - loss_minus) / (2.0f * eps);
}

void check_loss_gradient(const std::function<Tensor(const Tensor&, const Tensor&)>& loss_fn, Tensor pred,
                          const Tensor& target) {
    constexpr float eps = 1e-3f;
    constexpr float tolerance = 5e-2f;

    Tensor loss = loss_fn(pred, target);
    loss.backward();

    REQUIRE(pred.has_grad());
    Tensor grad = pred.grad();

    auto scalar_loss = [&]() { return loss_fn(pred, target).data()[0]; };

    for (size_t i = 0; i < pred.numel(); ++i) {
        const float analytic = grad.data()[i];
        const float numeric = numerical_grad(pred.data()[i], scalar_loss, eps);
        REQUIRE(analytic == Catch::Approx(numeric).margin(tolerance));
    }
}

}  // namespace

TEST_CASE("AbsoluteLoss matches a manual computation and its numerical gradient", "[nn][loss]") {
    const std::vector<float> pred_values = {1.5f, -0.5f, 2.0f, 0.3f};
    const std::vector<float> target_values = {1.0f, 0.0f, 2.5f, -0.2f};
    Tensor pred(pred_values, {2, 2}, /*requires_grad=*/true);
    Tensor target(target_values, {2, 2});

    AbsoluteLoss loss_fn;
    Tensor loss = loss_fn(pred, target);

    const float expected = (0.5f + 0.5f + 0.5f + 0.5f) / 4.0f;
    REQUIRE(loss.data()[0] == Catch::Approx(expected).margin(1e-5f));

    // Values kept away from equal (|diff| > 0.1) so the central-difference
    // check doesn't straddle |.|'s kink, same rationale as LeakyReLU's test.
    check_loss_gradient([](const Tensor& p, const Tensor& t) { return AbsoluteLoss()(p, t); }, pred, target);
}

TEST_CASE("BinaryCrossEntropy matches a manual computation and its numerical gradient", "[nn][loss]") {
    const std::vector<float> probs = {0.8f, 0.3f, 0.6f, 0.1f};
    const std::vector<float> labels = {1.0f, 0.0f, 1.0f, 0.0f};
    Tensor pred(probs, {2, 2}, /*requires_grad=*/true);
    Tensor target(labels, {2, 2});

    BinaryCrossEntropy loss_fn;
    Tensor loss = loss_fn(pred, target);

    float expected_sum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        expected_sum += labels[i] * std::log(probs[i]) + (1.0f - labels[i]) * std::log(1.0f - probs[i]);
    }
    const float expected = -expected_sum / 2.0f;  // batch size = 2
    REQUIRE(loss.data()[0] == Catch::Approx(expected).margin(1e-5f));

    check_loss_gradient([](const Tensor& p, const Tensor& t) { return BinaryCrossEntropy()(p, t); }, pred, target);
}
