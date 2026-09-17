#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#include "advanceml/nn/gelu.hpp"
#include "advanceml/nn/leaky_relu.hpp"
#include "advanceml/nn/linear.hpp"
#include "advanceml/nn/loss.hpp"
#include "advanceml/nn/relu.hpp"
#include "advanceml/nn/sequential.hpp"
#include "advanceml/nn/sigmoid.hpp"
#include "advanceml/nn/softmax.hpp"
#include "advanceml/ops.hpp"
#include "advanceml/optim/sgd.hpp"
#include "advanceml/tensor.hpp"

using namespace advanceml;

namespace {

// Independent forward-only recomputation used by the numerical gradient
// checks below -- deliberately doesn't share code with backward(), same
// pattern as autograd_test.cpp's mlp_loss.
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
        const float numeric = numerical_grad(x.mutable_data()[i], loss_fn, eps);
        REQUIRE(analytic == Catch::Approx(numeric).margin(tolerance));
    }
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
        const float numeric = numerical_grad(pred.mutable_data()[i], scalar_loss, eps);
        REQUIRE(analytic == Catch::Approx(numeric).margin(tolerance));
    }
}

}  // namespace

TEST_CASE("Sigmoid::forward matches the sigmoid op and its numerical gradient", "[nn][activation]") {
    Tensor x = Tensor::random_uniform({3, 4}, -2.0f, 2.0f, /*seed=*/201, /*requires_grad=*/true);
    check_activation_gradient([](const Tensor& t) { return Sigmoid().forward(t); }, x);
}

TEST_CASE("LeakyReLU::forward matches the leaky_relu op and its numerical gradient", "[nn][activation]") {
    // Values kept away from 0 (|v| > 0.1) so the central-difference check
    // doesn't straddle the derivative's kink there, same rationale as
    // autograd_test.cpp's ReLU seed choice.
    Tensor x = Tensor::random_uniform({3, 4}, -2.0f, 2.0f, /*seed=*/202, /*requires_grad=*/true);
    check_activation_gradient([](const Tensor& t) { return LeakyReLU(0.1f).forward(t); }, x);
}

TEST_CASE("GELU::forward matches the gelu op and its numerical gradient", "[nn][activation]") {
    Tensor x = Tensor::random_uniform({3, 4}, -2.0f, 2.0f, /*seed=*/203, /*requires_grad=*/true);
    check_activation_gradient([](const Tensor& t) { return GELU().forward(t); }, x);
}

TEST_CASE("Softmax::forward sums to 1 per row and matches its numerical gradient", "[nn][activation]") {
    Tensor x = Tensor::random_uniform({3, 4}, -2.0f, 2.0f, /*seed=*/204, /*requires_grad=*/true);
    Tensor y = Softmax().forward(x);
    for (size_t r = 0; r < 3; ++r) {
        float row_sum = 0.0f;
        for (size_t c = 0; c < 4; ++c) {
            row_sum += y.data()[r * 4 + c];
        }
        REQUIRE(row_sum == Catch::Approx(1.0f).margin(1e-4f));
    }
    check_activation_gradient([](const Tensor& t) { return Softmax().forward(t); }, x);
}

TEST_CASE("CrossEntropyLoss matches a manual computation and its numerical gradient", "[nn][loss]") {
    const std::vector<float> probs = {0.7f, 0.2f, 0.1f, 0.1f, 0.2f, 0.7f};
    const std::vector<float> one_hot = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    Tensor pred(probs, {2, 3}, /*requires_grad=*/true);
    Tensor target(one_hot, {2, 3});

    CrossEntropyLoss loss_fn;
    Tensor loss = loss_fn(pred, target);

    const float expected = -0.5f * (std::log(0.7f) + std::log(0.7f));
    REQUIRE(loss.data()[0] == Catch::Approx(expected).margin(1e-5f));

    check_loss_gradient([](const Tensor& p, const Tensor& t) { return CrossEntropyLoss()(p, t); }, pred, target);
}

TEST_CASE("Sequential/Softmax/CrossEntropyLoss/SGD decrease loss on a 3-class linearly separable task",
          "[nn][training]") {
    // One-hot-encoded points clustered near each axis of a 2D simplex-like
    // layout, one class per cluster -- deliberately linearly separable so a
    // single Linear layer plus Softmax/CrossEntropyLoss should fit it.
    const std::vector<float> inputs = {
        2.0f, 0.0f, 1.8f, 0.3f, 0.0f, 2.0f, 0.3f, 1.8f, -2.0f, 0.0f, -1.8f, -0.3f,
    };
    const std::vector<float> one_hot_targets = {
        1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
    };
    Tensor x(inputs, {6, 2});
    Tensor target(one_hot_targets, {6, 3});

    Sequential model({std::make_shared<Linear>(2, 3, /*seed=*/30)});
    Softmax softmax;
    CrossEntropyLoss loss_fn;
    SGD optimizer(model.parameters(), /*learning_rate=*/0.5f);

    auto forward_loss = [&]() { return loss_fn(softmax.forward(model.forward(x)), target); };

    const float initial_loss = forward_loss().data()[0];

    float final_loss = initial_loss;
    for (int epoch = 0; epoch < 200; ++epoch) {
        Tensor loss = forward_loss();
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
        final_loss = loss.data()[0];
    }

    REQUIRE(final_loss < initial_loss * 0.2f);
}

TEST_CASE("Sequential/Linear/ReLU/MSELoss/SGD reproduce the raw-op XOR MLP's loss-decrease result", "[nn][training]") {
    const std::vector<float> xor_inputs = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
    const std::vector<float> xor_targets = {0.0f, 1.0f, 1.0f, 0.0f};
    Tensor x(xor_inputs, {4, 2});
    Tensor target(xor_targets, {4, 1});

    Sequential model({
        std::make_shared<Linear>(2, 8, /*seed=*/20),
        std::make_shared<ReLU>(),
        std::make_shared<Linear>(8, 1, /*seed=*/21),
    });
    MSELoss loss_fn;
    SGD optimizer(model.parameters(), /*learning_rate=*/0.1f);

    auto forward_loss = [&]() { return loss_fn(model.forward(x), target); };

    const float initial_loss = forward_loss().data()[0];

    float final_loss = initial_loss;
    for (int epoch = 0; epoch < 500; ++epoch) {
        Tensor loss = forward_loss();
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
        final_loss = loss.data()[0];
    }

    REQUIRE(final_loss < initial_loss * 0.5f);
}

TEST_CASE("Sequential::parameters() concatenates every child layer's parameters in order", "[nn]") {
    Sequential model({
        std::make_shared<Linear>(2, 3, /*seed=*/1),
        std::make_shared<ReLU>(),
        std::make_shared<Linear>(3, 1, /*seed=*/2),
    });

    std::vector<Tensor> params = model.parameters();

    // Each Linear contributes (w, b): 2 layers * 2 params each, ReLU contributes none.
    REQUIRE(params.size() == 4);
    REQUIRE(params[0].shape() == std::vector<size_t>{2, 3});
    REQUIRE(params[1].shape() == std::vector<size_t>{3});
    REQUIRE(params[2].shape() == std::vector<size_t>{3, 1});
    REQUIRE(params[3].shape() == std::vector<size_t>{1});
}
