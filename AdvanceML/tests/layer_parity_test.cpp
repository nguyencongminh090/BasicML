#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <vector>

#include "advanceml/nn/batch_norm1d.hpp"
#include "advanceml/nn/dropout.hpp"
#include "advanceml/nn/global_avg_pool2d.hpp"
#include "advanceml/nn/global_max_pool2d.hpp"
#include "advanceml/ops.hpp"
#include "advanceml/tensor.hpp"

using namespace advanceml;

namespace {

// Same recipe as cnn_test.cpp's numerical_grad/check_input_gradient, kept
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

void check_input_gradient(Tensor& input, const std::function<float()>& loss_fn, float eps, float tolerance) {
    const float loss_value = loss_fn();
    (void)loss_value;
    Tensor grad = input.grad();
    for (size_t i = 0; i < input.numel(); ++i) {
        const float analytic = grad.data()[i];
        const float numeric = numerical_grad(input.mutable_data()[i], loss_fn, eps);
        REQUIRE(analytic == Catch::Approx(numeric).margin(tolerance));
    }
}

}  // namespace

TEST_CASE("GlobalAvgPool2D averages every channel's spatial plane and its backward matches the numerical gradient",
          "[nn][pooling]") {
    Tensor x({1.0f, 2.0f, 3.0f, 4.0f, 10.0f, 20.0f, 30.0f, 40.0f}, {1, 2, 2, 2}, /*requires_grad=*/true);
    GlobalAvgPool2D pool;
    Tensor y = pool.forward(x);
    REQUIRE(y.shape() == std::vector<size_t>{1, 2, 1, 1});
    REQUIRE(y.data()[0] == Catch::Approx(2.5f));
    REQUIRE(y.data()[1] == Catch::Approx(25.0f));

    Tensor x_grad = Tensor::random_uniform({2, 3, 4, 4}, -1.0f, 1.0f, /*seed=*/401, /*requires_grad=*/true);
    auto forward_loss = [&]() {
        Tensor out = pool.forward(x_grad);
        return mse_loss(out, Tensor::zeros(out.shape()));
    };
    Tensor loss = forward_loss();
    loss.backward();
    REQUIRE(x_grad.has_grad());
    check_input_gradient(x_grad, [&]() { return forward_loss().data()[0]; }, /*eps=*/1e-2f, /*tolerance=*/5e-2f);
}

TEST_CASE("GlobalMaxPool2D picks each channel's spatial max and its backward matches the numerical gradient",
          "[nn][pooling]") {
    Tensor x({1.0f, 3.0f, 2.0f, 4.0f, 40.0f, 10.0f, 30.0f, 20.0f}, {1, 2, 2, 2}, /*requires_grad=*/true);
    GlobalMaxPool2D pool;
    Tensor y = pool.forward(x);
    REQUIRE(y.shape() == std::vector<size_t>{1, 2, 1, 1});
    REQUIRE(y.data()[0] == Catch::Approx(4.0f));
    REQUIRE(y.data()[1] == Catch::Approx(40.0f));

    // Hand-crafted rather than random: each (n, c) plane's max must sit well
    // clear of its other three elements (margin >= 0.5) so a +-1e-2 central
    // difference can never flip the argmax and spuriously zero the gradient.
    std::vector<float> plane_data;
    plane_data.reserve(6 * 4);
    for (size_t plane = 0; plane < 6; ++plane) {
        const float base = 0.1f * static_cast<float>(plane);
        plane_data.insert(plane_data.end(), {base, base + 0.05f, base + 0.1f, base + 1.0f});
    }
    Tensor x_grad(plane_data, {2, 3, 2, 2}, /*requires_grad=*/true);
    auto forward_loss = [&]() {
        Tensor out = pool.forward(x_grad);
        return mse_loss(out, Tensor::zeros(out.shape()));
    };
    Tensor loss = forward_loss();
    loss.backward();
    REQUIRE(x_grad.has_grad());
    check_input_gradient(x_grad, [&]() { return forward_loss().data()[0]; }, /*eps=*/1e-2f, /*tolerance=*/5e-2f);
}

TEST_CASE("BatchNorm1D normalizes to zero mean/unit variance per feature in training mode", "[nn][batchnorm]") {
    Tensor x = Tensor::random_uniform({8, 3}, -2.0f, 2.0f, /*seed=*/403, /*requires_grad=*/true);
    BatchNorm1D bn(3);
    Tensor y = bn.forward(x);
    REQUIRE(y.shape() == x.shape());

    for (size_t c = 0; c < 3; ++c) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (size_t n = 0; n < 8; ++n) {
            const float v = y.data()[n * 3 + c];
            sum += v;
            sum_sq += v * v;
        }
        const float mean = sum / 8.0f;
        const float var = sum_sq / 8.0f - mean * mean;
        REQUIRE(mean == Catch::Approx(0.0f).margin(1e-3f));
        REQUIRE(var == Catch::Approx(1.0f).margin(1e-2f));
    }
}

TEST_CASE("BatchNorm1D backward matches numerical gradients w.r.t. x, gamma, and beta in training mode",
          "[nn][batchnorm]") {
    Tensor x = Tensor::random_uniform({6, 3}, -1.0f, 1.0f, /*seed=*/404, /*requires_grad=*/true);
    BatchNorm1D bn(3);

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

TEST_CASE("BatchNorm1D eval mode uses frozen running statistics instead of the batch's own", "[nn][batchnorm]") {
    Tensor x1 = Tensor::random_uniform({8, 3}, -2.0f, 2.0f, /*seed=*/405);
    BatchNorm1D bn(3);
    bn.forward(x1);  // one training-mode pass to move the running stats away from their init values

    bn.eval();
    Tensor x2 = Tensor::random_uniform({8, 3}, -2.0f, 2.0f, /*seed=*/406);
    Tensor y_eval_a = bn.forward(x2);
    Tensor y_eval_b = bn.forward(x2);
    for (size_t i = 0; i < y_eval_a.numel(); ++i) {
        REQUIRE(y_eval_a.data()[i] == Catch::Approx(y_eval_b.data()[i]));
    }
}

TEST_CASE("Dropout::eval passes input through unchanged", "[nn][dropout]") {
    Tensor x = Tensor::random_uniform({4, 5}, -1.0f, 1.0f, /*seed=*/407);
    Dropout layer(0.5f, /*seed=*/1);
    layer.eval();
    Tensor y = layer.forward(x);
    for (size_t i = 0; i < x.numel(); ++i) {
        REQUIRE(y.data()[i] == Catch::Approx(x.data()[i]));
    }
}

TEST_CASE("Dropout::train zeroes roughly p of elements and scales survivors by 1/(1-p)", "[nn][dropout]") {
    constexpr float p = 0.5f;
    Tensor x(std::vector<float>(2000, 1.0f), {2000});
    Dropout layer(p, /*seed=*/42);
    Tensor y = layer.forward(x);

    size_t zero_count = 0;
    for (size_t i = 0; i < y.numel(); ++i) {
        if (y.data()[i] == 0.0f) {
            ++zero_count;
        } else {
            REQUIRE(y.data()[i] == Catch::Approx(1.0f / (1.0f - p)));
        }
    }
    const float zero_fraction = static_cast<float>(zero_count) / static_cast<float>(y.numel());
    REQUIRE(zero_fraction == Catch::Approx(p).margin(0.05f));
}

TEST_CASE("Dropout::train backward routes gradient through the same mask used in forward", "[nn][dropout]") {
    Tensor x = Tensor::random_uniform({50}, -1.0f, 1.0f, /*seed=*/408, /*requires_grad=*/true);
    Dropout layer(0.3f, /*seed=*/7);
    Tensor y = layer.forward(x);
    Tensor loss = mse_loss(y, Tensor::zeros(y.shape()));
    loss.backward();

    REQUIRE(x.has_grad());
    Tensor grad = x.grad();
    for (size_t i = 0; i < x.numel(); ++i) {
        if (y.data()[i] == 0.0f) {
            REQUIRE(grad.data()[i] == Catch::Approx(0.0f));
        } else {
            const float scale = y.data()[i] / x.data()[i];
            const float expected = 2.0f * y.data()[i] / static_cast<float>(x.numel()) * scale;
            REQUIRE(grad.data()[i] == Catch::Approx(expected).margin(1e-4f));
        }
    }
}

TEST_CASE("Dropout with p == 0 is a no-op even in training mode", "[nn][dropout]") {
    Tensor x = Tensor::random_uniform({10}, -1.0f, 1.0f, /*seed=*/409);
    Dropout layer(0.0f, /*seed=*/3);
    Tensor y = layer.forward(x);
    for (size_t i = 0; i < x.numel(); ++i) {
        REQUIRE(y.data()[i] == Catch::Approx(x.data()[i]));
    }
}
