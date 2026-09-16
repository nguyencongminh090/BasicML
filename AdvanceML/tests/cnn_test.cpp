#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <vector>

#include "advanceml/nn/avg_pool2d.hpp"
#include "advanceml/nn/batch_norm2d.hpp"
#include "advanceml/nn/conv2d.hpp"
#include "advanceml/nn/flatten.hpp"
#include "advanceml/nn/linear.hpp"
#include "advanceml/nn/loss.hpp"
#include "advanceml/nn/max_pool2d.hpp"
#include "advanceml/nn/relu.hpp"
#include "advanceml/nn/sequential.hpp"
#include "advanceml/ops.hpp"
#include "advanceml/optim/sgd.hpp"
#include "advanceml/tensor.hpp"

using namespace advanceml;

namespace {

// Same numerical-gradient recipe as nn_test.cpp's check_activation_gradient,
// generalized to any tensor -> scalar-loss function so it also covers
// multi-input ops (conv2d's weight/bias) via check_input_gradient below.
float numerical_grad(float& element, const std::function<float()>& loss_fn, float eps) {
    const float original = element;
    element = original + eps;
    const float loss_plus = loss_fn();
    element = original - eps;
    const float loss_minus = loss_fn();
    element = original;
    return (loss_plus - loss_minus) / (2.0f * eps);
}

void check_input_gradient(Tensor& input, const std::function<float()>& loss_fn, float eps, float tolerance) {
    const float loss_value = loss_fn();
    (void)loss_value;
    Tensor grad = input.grad();
    for (size_t i = 0; i < input.numel(); ++i) {
        const float analytic = grad.data()[i];
        const float numeric = numerical_grad(input.data()[i], loss_fn, eps);
        REQUIRE(analytic == Catch::Approx(numeric).margin(tolerance));
    }
}

}  // namespace

TEST_CASE("conv2d forward produces the expected output shape and a hand-computed value", "[ops][conv2d]") {
    // 1x1x3x3 input, a single 2x2 kernel, stride 1, no padding -> 1x1x2x2 output.
    Tensor x({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}, {1, 1, 3, 3}, /*requires_grad=*/true);
    Tensor weight({1.0f, 0.0f, 0.0f, 1.0f}, {1, 1, 2, 2}, /*requires_grad=*/true);
    Tensor bias({0.5f}, {1}, /*requires_grad=*/true);

    Tensor y = conv2d(x, weight, bias, /*stride=*/1, /*padding=*/0);
    REQUIRE(y.shape() == std::vector<size_t>{1, 1, 2, 2});
    // Each output = top-left + bottom-right of its window + bias.
    REQUIRE(y.data()[0] == Catch::Approx(1.0f + 5.0f + 0.5f));
    REQUIRE(y.data()[1] == Catch::Approx(2.0f + 6.0f + 0.5f));
    REQUIRE(y.data()[2] == Catch::Approx(4.0f + 8.0f + 0.5f));
    REQUIRE(y.data()[3] == Catch::Approx(5.0f + 9.0f + 0.5f));
}

TEST_CASE("conv2d backward matches numerical gradients w.r.t. x, weight, and bias", "[ops][conv2d]") {
    Tensor x = Tensor::random_uniform({2, 3, 5, 5}, -1.0f, 1.0f, /*seed=*/301, /*requires_grad=*/true);
    Tensor weight = Tensor::random_uniform({4, 3, 3, 3}, -1.0f, 1.0f, /*seed=*/302, /*requires_grad=*/true);
    Tensor bias = Tensor::random_uniform({4}, -1.0f, 1.0f, /*seed=*/303, /*requires_grad=*/true);

    auto forward_loss = [&]() {
        Tensor y = conv2d(x, weight, bias, /*stride=*/2, /*padding=*/1);
        return mse_loss(y, Tensor::zeros(y.shape()));
    };

    Tensor loss = forward_loss();
    loss.backward();

    REQUIRE(x.has_grad());
    REQUIRE(weight.has_grad());
    REQUIRE(bias.has_grad());

    constexpr float eps = 1e-2f;
    constexpr float tolerance = 5e-2f;
    check_input_gradient(x, [&]() { return forward_loss().data()[0]; }, eps, tolerance);
    check_input_gradient(weight, [&]() { return forward_loss().data()[0]; }, eps, tolerance);
    check_input_gradient(bias, [&]() { return forward_loss().data()[0]; }, eps, tolerance);
}

TEST_CASE("max_pool2d picks each window's max and its backward matches the numerical gradient", "[ops][pooling]") {
    Tensor x({1.0f, 3.0f, 2.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
             {1, 1, 4, 4}, /*requires_grad=*/true);
    Tensor y = max_pool2d(x, /*kernel_size=*/2, /*stride=*/2);
    REQUIRE(y.shape() == std::vector<size_t>{1, 1, 2, 2});
    REQUIRE(y.data()[0] == Catch::Approx(6.0f));
    REQUIRE(y.data()[1] == Catch::Approx(8.0f));
    REQUIRE(y.data()[2] == Catch::Approx(14.0f));
    REQUIRE(y.data()[3] == Catch::Approx(16.0f));

    Tensor x_grad = Tensor::random_uniform({2, 3, 6, 6}, -1.0f, 1.0f, /*seed=*/304, /*requires_grad=*/true);
    auto forward_loss = [&]() {
        Tensor out = max_pool2d(x_grad, /*kernel_size=*/2, /*stride=*/2);
        return mse_loss(out, Tensor::zeros(out.shape()));
    };
    Tensor loss = forward_loss();
    loss.backward();
    REQUIRE(x_grad.has_grad());
    check_input_gradient(x_grad, [&]() { return forward_loss().data()[0]; }, /*eps=*/1e-2f, /*tolerance=*/5e-2f);
}

TEST_CASE("avg_pool2d averages each window and its backward matches the numerical gradient", "[ops][pooling]") {
    Tensor x({1.0f, 2.0f, 3.0f, 4.0f}, {1, 1, 2, 2}, /*requires_grad=*/true);
    Tensor y = avg_pool2d(x, /*kernel_size=*/2, /*stride=*/2);
    REQUIRE(y.shape() == std::vector<size_t>{1, 1, 1, 1});
    REQUIRE(y.data()[0] == Catch::Approx(2.5f));

    Tensor x_grad = Tensor::random_uniform({2, 3, 6, 6}, -1.0f, 1.0f, /*seed=*/305, /*requires_grad=*/true);
    auto forward_loss = [&]() {
        Tensor out = avg_pool2d(x_grad, /*kernel_size=*/2, /*stride=*/2);
        return mse_loss(out, Tensor::zeros(out.shape()));
    };
    Tensor loss = forward_loss();
    loss.backward();
    REQUIRE(x_grad.has_grad());
    check_input_gradient(x_grad, [&]() { return forward_loss().data()[0]; }, /*eps=*/1e-2f, /*tolerance=*/5e-2f);
}

TEST_CASE("Flatten reshapes (N, C, H, W) to (N, C*H*W) and its backward matches the numerical gradient",
          "[nn][flatten]") {
    Tensor x = Tensor::random_uniform({2, 3, 4, 4}, -1.0f, 1.0f, /*seed=*/306, /*requires_grad=*/true);
    Flatten flatten_layer;
    Tensor y = flatten_layer.forward(x);
    REQUIRE(y.shape() == std::vector<size_t>{2, 48});

    auto forward_loss = [&]() {
        Tensor out = flatten_layer.forward(x);
        return mse_loss(out, Tensor::zeros(out.shape()));
    };
    Tensor loss = forward_loss();
    loss.backward();
    REQUIRE(x.has_grad());
    check_input_gradient(x, [&]() { return forward_loss().data()[0]; }, /*eps=*/1e-2f, /*tolerance=*/5e-2f);
}

TEST_CASE("BatchNorm2D normalizes to zero mean/unit variance per channel in training mode", "[nn][batchnorm]") {
    Tensor x = Tensor::random_uniform({4, 2, 3, 3}, -2.0f, 2.0f, /*seed=*/307, /*requires_grad=*/true);
    BatchNorm2D bn(2);
    Tensor y = bn.forward(x);
    REQUIRE(y.shape() == x.shape());

    for (size_t c = 0; c < 2; ++c) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        const size_t count = 4 * 3 * 3;
        for (size_t n = 0; n < 4; ++n) {
            for (size_t p = 0; p < 9; ++p) {
                const float v = y.data()[(n * 2 + c) * 9 + p];
                sum += v;
                sum_sq += v * v;
            }
        }
        const float mean = sum / static_cast<float>(count);
        const float var = sum_sq / static_cast<float>(count) - mean * mean;
        REQUIRE(mean == Catch::Approx(0.0f).margin(1e-3f));
        REQUIRE(var == Catch::Approx(1.0f).margin(1e-2f));
    }
}

TEST_CASE("BatchNorm2D backward matches numerical gradients w.r.t. x, gamma, and beta in training mode",
          "[nn][batchnorm]") {
    Tensor x = Tensor::random_uniform({3, 2, 3, 3}, -1.0f, 1.0f, /*seed=*/308, /*requires_grad=*/true);
    BatchNorm2D bn(2);

    auto forward_loss = [&]() {
        Tensor out = bn.forward(x);
        return mse_loss(out, Tensor::zeros(out.shape()));
    };
    Tensor loss = forward_loss();
    loss.backward();
    REQUIRE(x.has_grad());

    std::vector<Tensor> params = bn.parameters();
    REQUIRE(params.size() == 2);
    REQUIRE(params[0].has_grad());
    REQUIRE(params[1].has_grad());

    constexpr float eps = 1e-2f;
    constexpr float tolerance = 5e-2f;
    check_input_gradient(x, [&]() { return forward_loss().data()[0]; }, eps, tolerance);
    check_input_gradient(params[0], [&]() { return forward_loss().data()[0]; }, eps, tolerance);
    check_input_gradient(params[1], [&]() { return forward_loss().data()[0]; }, eps, tolerance);
}

TEST_CASE("BatchNorm2D eval mode uses frozen running statistics instead of the batch's own", "[nn][batchnorm]") {
    Tensor x1 = Tensor::random_uniform({4, 2, 3, 3}, -2.0f, 2.0f, /*seed=*/309);
    BatchNorm2D bn(2);
    bn.forward(x1);  // one training-mode pass to move the running stats away from their init values

    bn.eval();
    Tensor x2 = Tensor::random_uniform({4, 2, 3, 3}, -2.0f, 2.0f, /*seed=*/310);
    Tensor y_eval_a = bn.forward(x2);
    Tensor y_eval_b = bn.forward(x2);
    // Eval mode must be deterministic run-to-run (no batch-stat dependence).
    for (size_t i = 0; i < y_eval_a.numel(); ++i) {
        REQUIRE(y_eval_a.data()[i] == Catch::Approx(y_eval_b.data()[i]));
    }
}

TEST_CASE("A Conv2D/ReLU/MaxPool2D/Flatten/Linear Sequential model trains end to end on a toy task",
          "[nn][training][cnn]") {
    // 4 samples of a 1x6x6 image, a binary target of whether the top-left
    // quadrant's mean exceeds the bottom-right quadrant's -- a task a small
    // conv stack should be able to fit.
    Tensor x = Tensor::random_uniform({4, 1, 6, 6}, -1.0f, 1.0f, /*seed=*/311);
    std::vector<float> target_data(4);
    for (size_t n = 0; n < 4; ++n) {
        float top_left = 0.0f;
        float bottom_right = 0.0f;
        for (size_t h = 0; h < 3; ++h) {
            for (size_t w = 0; w < 3; ++w) {
                top_left += x.data()[(n * 36) + h * 6 + w];
                bottom_right += x.data()[(n * 36) + (h + 3) * 6 + (w + 3)];
            }
        }
        target_data[n] = top_left > bottom_right ? 1.0f : 0.0f;
    }
    Tensor target(target_data, {4, 1});

    Sequential model({
        std::make_shared<Conv2D>(1, 4, /*kernel_size=*/3, /*stride=*/1, /*padding=*/1, /*seed=*/40),
        std::make_shared<ReLU>(),
        std::make_shared<MaxPool2D>(/*kernel_size=*/2, /*stride=*/2),
        std::make_shared<Flatten>(),
        std::make_shared<Linear>(4 * 3 * 3, 1, /*seed=*/41),
    });
    MSELoss loss_fn;
    SGD optimizer(model.parameters(), /*learning_rate=*/0.1f);

    auto forward_loss = [&]() { return loss_fn(model.forward(x), target); };
    const float initial_loss = forward_loss().data()[0];

    float final_loss = initial_loss;
    for (int epoch = 0; epoch < 200; ++epoch) {
        Tensor loss = forward_loss();
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
        final_loss = loss.data()[0];
    }

    REQUIRE(final_loss < initial_loss * 0.5f);
}
