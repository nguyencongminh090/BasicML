#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <vector>

#include "advanceml/nn/linear.hpp"
#include "advanceml/nn/loss.hpp"
#include "advanceml/nn/relu.hpp"
#include "advanceml/nn/sequential.hpp"
#include "advanceml/optim/sgd.hpp"
#include "advanceml/tensor.hpp"

using namespace advanceml;

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
