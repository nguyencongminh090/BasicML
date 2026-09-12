#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <vector>

#include "advanceml/ops.hpp"
#include "advanceml/optim/sgd.hpp"
#include "advanceml/tensor.hpp"

using namespace advanceml;

namespace {

// Forward-only recomputation of a 2-layer MLP's MSE loss, used by the
// numerical gradient check -- deliberately independent of the graph
// built by backward() so it can't share a bug with it.
float mlp_loss(const Tensor& x, const Tensor& w1, const Tensor& b1, const Tensor& w2, const Tensor& b2, const Tensor& target) {
    Tensor hidden = relu(matmul(x, w1) + b1);
    Tensor pred = matmul(hidden, w2) + b2;
    Tensor loss = mse_loss(pred, target);
    return loss.data()[0];
}

float numerical_grad(float& element, const std::function<float()>& loss_fn, float eps) {
    const float original = element;
    element = original + eps;
    const float loss_plus = loss_fn();
    element = original - eps;
    const float loss_minus = loss_fn();
    element = original;
    return (loss_plus - loss_minus) / (2.0f * eps);
}

}  // namespace

TEST_CASE("backward() matches numerical gradient on a small MLP graph", "[autograd]") {
    // Seeds chosen so every ReLU pre-activation lands well away from
    // the kink at 0 (|preact| > 0.1) -- otherwise a central-difference
    // numerical gradient through the discontinuous derivative is
    // inherently unreliable near the boundary, independent of whether
    // backward() is correct.
    Tensor x = Tensor::random_uniform({2, 3}, -1.0f, 1.0f, /*seed=*/111, /*requires_grad=*/false);
    Tensor w1 = Tensor::random_uniform({3, 4}, -0.5f, 0.5f, /*seed=*/112, /*requires_grad=*/true);
    Tensor b1 = Tensor::random_uniform({4}, -0.5f, 0.5f, /*seed=*/113, /*requires_grad=*/true);
    Tensor w2 = Tensor::random_uniform({4, 1}, -0.5f, 0.5f, /*seed=*/4, /*requires_grad=*/true);
    Tensor b2 = Tensor::random_uniform({1}, -0.5f, 0.5f, /*seed=*/5, /*requires_grad=*/true);
    Tensor target = Tensor::random_uniform({2, 1}, -1.0f, 1.0f, /*seed=*/6, /*requires_grad=*/false);

    Tensor hidden = relu(matmul(x, w1) + b1);
    Tensor pred = matmul(hidden, w2) + b2;
    Tensor loss = mse_loss(pred, target);
    loss.backward();

    constexpr float eps = 1e-2f;
    constexpr float tolerance = 5e-2f;

    auto loss_fn = [&]() { return mlp_loss(x, w1, b1, w2, b2, target); };

    for (Tensor* param : {&w1, &b1, &w2, &b2}) {
        REQUIRE(param->has_grad());
        Tensor grad = param->grad();
        for (size_t i = 0; i < param->numel(); ++i) {
            const float analytic = grad.data()[i];
            const float numeric = numerical_grad(param->data()[i], loss_fn, eps);
            REQUIRE(analytic == Catch::Approx(numeric).margin(tolerance));
        }
    }
}

TEST_CASE("a tiny MLP trained with SGD reduces its XOR loss", "[autograd][training]") {
    const std::vector<float> xor_inputs = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
    const std::vector<float> xor_targets = {0.0f, 1.0f, 1.0f, 0.0f};
    Tensor x(xor_inputs, {4, 2});
    Tensor target(xor_targets, {4, 1});

    Tensor w1 = Tensor::random_uniform({2, 8}, -1.0f, 1.0f, /*seed=*/10, /*requires_grad=*/true);
    Tensor b1 = Tensor::zeros({8}, /*requires_grad=*/true);
    Tensor w2 = Tensor::random_uniform({8, 1}, -1.0f, 1.0f, /*seed=*/11, /*requires_grad=*/true);
    Tensor b2 = Tensor::zeros({1}, /*requires_grad=*/true);

    SGD optimizer({w1, b1, w2, b2}, /*learning_rate=*/0.1f);

    auto forward_loss = [&]() {
        Tensor hidden = relu(matmul(x, w1) + b1);
        Tensor pred = matmul(hidden, w2) + b2;
        return mse_loss(pred, target);
    };

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
