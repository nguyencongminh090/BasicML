#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <vector>

#include "advanceml/ops.hpp"
#include "advanceml/tensor.hpp"

using namespace advanceml;

TEST_CASE("NoGradGuard records no nodes and restores the previous mode", "[autograd][no_grad]") {
    Tensor x = Tensor::random_uniform({4, 3}, -1.0f, 1.0f, /*seed=*/1);
    Tensor w = Tensor::random_uniform({3, 2}, -1.0f, 1.0f, /*seed=*/2, /*requires_grad=*/true);
    Tensor b = Tensor::zeros({2}, /*requires_grad=*/true);
    Tensor conv_x = Tensor::random_uniform({2, 1, 5, 5}, -1.0f, 1.0f, /*seed=*/3);
    Tensor conv_w = Tensor::random_uniform({2, 1, 3, 3}, -1.0f, 1.0f, /*seed=*/4, /*requires_grad=*/true);
    Tensor conv_b = Tensor::zeros({2}, /*requires_grad=*/true);

    REQUIRE(is_grad_enabled());
    {
        NoGradGuard outer;
        REQUIRE_FALSE(is_grad_enabled());
        {
            NoGradGuard inner;
            REQUIRE_FALSE(is_grad_enabled());
        }
        REQUIRE_FALSE(is_grad_enabled());

        Tensor out = relu(matmul(x, w) + b);
        REQUIRE_FALSE(out.requires_grad());
        REQUIRE(out.impl()->grad_fn == nullptr);

        Tensor pooled = max_pool2d(conv2d(conv_x, conv_w, conv_b, /*stride=*/1, /*padding=*/1), 2, 2);
        REQUIRE_FALSE(pooled.requires_grad());
        REQUIRE(pooled.impl()->grad_fn == nullptr);
    }
    REQUIRE(is_grad_enabled());

    Tensor recorded = matmul(x, w) + b;
    REQUIRE(recorded.requires_grad());
    REQUIRE(recorded.impl()->grad_fn != nullptr);
}

TEST_CASE("inference-mode conv2d/max_pool2d forward matches the recorded forward", "[autograd][no_grad]") {
    Tensor x = Tensor::random_uniform({3, 2, 6, 6}, -1.0f, 1.0f, /*seed=*/10);
    Tensor w = Tensor::random_uniform({4, 2, 3, 3}, -1.0f, 1.0f, /*seed=*/11, /*requires_grad=*/true);
    Tensor b = Tensor::random_uniform({4}, -1.0f, 1.0f, /*seed=*/12, /*requires_grad=*/true);

    Tensor recorded = max_pool2d(conv2d(x, w, b, /*stride=*/1, /*padding=*/1), 2, 2);
    NoGradGuard no_grad;
    Tensor inference = max_pool2d(conv2d(x, w, b, /*stride=*/1, /*padding=*/1), 2, 2);

    REQUIRE(recorded.shape() == inference.shape());
    for (size_t i = 0; i < recorded.numel(); ++i) {
        REQUIRE(inference.data()[i] == Catch::Approx(recorded.data()[i]).margin(1e-5f));
    }
}

TEST_CASE("inputs that do not require a gradient get none", "[autograd][skip_grad]") {
    Tensor x = Tensor::random_uniform({4, 3}, -1.0f, 1.0f, /*seed=*/20);
    Tensor w = Tensor::random_uniform({3, 2}, -1.0f, 1.0f, /*seed=*/21, /*requires_grad=*/true);
    Tensor target = Tensor::random_uniform({4, 2}, -1.0f, 1.0f, /*seed=*/22);

    Tensor pred = matmul(x, w);
    const Node& node = *pred.impl()->grad_fn;
    REQUIRE(node.num_inputs == 2);
    REQUIRE_FALSE(node.needs_input_grad[0]);
    REQUIRE(node.needs_input_grad[1]);

    Tensor loss = mse_loss(pred, target);
    loss.backward();
    REQUIRE(w.has_grad());
    REQUIRE_FALSE(x.has_grad());
    REQUIRE_FALSE(target.has_grad());
}

TEST_CASE("conv2d parameter gradients are unchanged when the input needs no gradient", "[autograd][skip_grad]") {
    const std::vector<float> x_values = Tensor::random_uniform({2, 3, 5, 5}, -1.0f, 1.0f, /*seed=*/30).data();
    Tensor w = Tensor::random_uniform({4, 3, 3, 3}, -1.0f, 1.0f, /*seed=*/31, /*requires_grad=*/true);
    Tensor b = Tensor::random_uniform({4}, -1.0f, 1.0f, /*seed=*/32, /*requires_grad=*/true);

    Tensor x_with_grad(x_values, {2, 3, 5, 5}, /*requires_grad=*/true);
    Tensor y1 = conv2d(x_with_grad, w, b, /*stride=*/1, /*padding=*/1);
    mse_loss(y1, Tensor::zeros(y1.shape())).backward();
    const std::vector<float> w_grad_full = w.grad().data();
    const std::vector<float> b_grad_full = b.grad().data();
    REQUIRE(x_with_grad.has_grad());

    w.zero_grad();
    b.zero_grad();
    Tensor x_data_only(x_values, {2, 3, 5, 5});
    Tensor y2 = conv2d(x_data_only, w, b, /*stride=*/1, /*padding=*/1);
    mse_loss(y2, Tensor::zeros(y2.shape())).backward();
    REQUIRE_FALSE(x_data_only.has_grad());
    REQUIRE(w.grad().data() == w_grad_full);
    REQUIRE(b.grad().data() == b_grad_full);
}

TEST_CASE("modifying a tensor saved for backward in place makes backward throw", "[autograd][version]") {
    SECTION("a saved input") {
        Tensor x = Tensor::random_uniform({5}, -1.0f, 1.0f, /*seed=*/40, /*requires_grad=*/true);
        Tensor loss = mse_loss(relu(x), Tensor::zeros({5}));
        const uint64_t before = x.version();
        x.mutable_data()[0] = 3.0f;
        REQUIRE(x.version() == before + 1);
        REQUIRE_THROWS_AS(loss.backward(), std::runtime_error);
    }
    SECTION("a saved output") {
        Tensor x = Tensor::random_uniform({5}, -1.0f, 1.0f, /*seed=*/41, /*requires_grad=*/true);
        Tensor y = sigmoid(x);
        Tensor loss = mse_loss(y + Tensor::zeros({5}), Tensor::zeros({5}));
        y.mutable_data()[0] = 0.5f;
        REQUIRE_THROWS_AS(loss.backward(), std::runtime_error);
    }
    SECTION("a write through a view sharing the saved storage") {
        Tensor x = Tensor::random_uniform({2, 3}, -1.0f, 1.0f, /*seed=*/42, /*requires_grad=*/true);
        Tensor loss = mse_loss(relu(x), Tensor::zeros({2, 3}));
        Tensor view = flatten(x);
        view.mutable_data()[0] = 3.0f;
        REQUIRE_THROWS_AS(loss.backward(), std::runtime_error);
    }
    SECTION("reads through data() are not writes") {
        Tensor x = Tensor::random_uniform({5}, -1.0f, 1.0f, /*seed=*/43, /*requires_grad=*/true);
        Tensor hidden = relu(x);
        Tensor loss = mse_loss(hidden, Tensor::zeros({5}));
        REQUIRE(hidden.data().size() == 5);
        REQUIRE(x.data().size() == 5);
        REQUIRE_NOTHROW(loss.backward());
    }
}

TEST_CASE("a 100k-op chain runs backward and tears down without overflowing the stack", "[autograd][deep]") {
    constexpr size_t kDepth = 100000;
    Tensor one({1.0f}, {1});

    SECTION("backward releases the graph as it goes") {
        Tensor x({0.0f}, {1}, /*requires_grad=*/true);
        Tensor y = x;
        for (size_t i = 0; i < kDepth; ++i) {
            y = y + one;
        }
        REQUIRE(y.data()[0] == Catch::Approx(static_cast<float>(kDepth)));
        y.backward();
        REQUIRE(x.grad().data()[0] == Catch::Approx(1.0f));
    }
    SECTION("a retained graph is destroyed iteratively") {
        Tensor x({0.0f}, {1}, /*requires_grad=*/true);
        {
            Tensor y = x;
            for (size_t i = 0; i < kDepth; ++i) {
                y = y + one;
            }
            y.backward(/*retain_graph=*/true);
        }
        REQUIRE(x.grad().data()[0] == Catch::Approx(1.0f));
    }
    SECTION("a graph that never runs backward is destroyed iteratively") {
        Tensor x({0.0f}, {1}, /*requires_grad=*/true);
        {
            Tensor y = x;
            for (size_t i = 0; i < kDepth; ++i) {
                y = relu(y + one);
            }
        }
        REQUIRE_FALSE(x.has_grad());
    }
}

TEST_CASE("retain_graph controls whether a graph can be backpropagated twice", "[autograd][retain_graph]") {
    Tensor w = Tensor::random_uniform({3, 2}, -1.0f, 1.0f, /*seed=*/50, /*requires_grad=*/true);
    Tensor x = Tensor::random_uniform({4, 3}, -1.0f, 1.0f, /*seed=*/51);
    Tensor target = Tensor::zeros({4, 2});

    Tensor retained = mse_loss(relu(matmul(x, w)), target);
    retained.backward(/*retain_graph=*/true);
    const std::vector<float> once = w.grad().data();
    retained.backward();
    for (size_t i = 0; i < once.size(); ++i) {
        REQUIRE(w.grad().data()[i] == Catch::Approx(2.0f * once[i]));
    }
    REQUIRE_THROWS_AS(retained.backward(), std::runtime_error);
}

TEST_CASE("gradients from several consumers accumulate without aliasing", "[autograd][accumulate]") {
    SECTION("a tensor used twice sums both contributions") {
        Tensor x({1.0f, -2.0f, 3.0f}, {3}, /*requires_grad=*/true);
        Tensor y = x * x + x;
        Tensor loss = mse_loss(y, Tensor::zeros({3}));
        loss.backward();
        for (size_t i = 0; i < 3; ++i) {
            const float xi = x.data()[i];
            const float expected = 2.0f * (xi * xi + xi) / 3.0f * (2.0f * xi + 1.0f);
            REQUIRE(x.grad().data()[i] == Catch::Approx(expected));
        }
    }
    SECTION("two leaves handed the same gradient buffer do not share it") {
        Tensor a({1.0f, 2.0f}, {2}, /*requires_grad=*/true);
        Tensor b({3.0f, 4.0f}, {2}, /*requires_grad=*/true);
        Tensor loss = mse_loss(a + b, Tensor::zeros({2}));
        loss.backward();
        const std::vector<float> b_grad = b.grad().data();
        a.grad().mutable_data()[0] = 100.0f;
        REQUIRE(b.grad().data() == b_grad);
    }
    SECTION("a second backward accumulates into an existing leaf gradient") {
        Tensor w({0.5f, -1.5f}, {2}, /*requires_grad=*/true);
        mse_loss(w, Tensor::zeros({2})).backward();
        const std::vector<float> once = w.grad().data();
        Tensor snapshot = w.grad();
        mse_loss(w, Tensor::zeros({2})).backward();
        for (size_t i = 0; i < 2; ++i) {
            REQUIRE(w.grad().data()[i] == Catch::Approx(2.0f * once[i]));
            REQUIRE(snapshot.data()[i] == Catch::Approx(once[i]));
        }
    }
}

TEST_CASE("reshape views share storage with their input", "[autograd][view]") {
    Tensor x = Tensor::random_uniform({2, 3, 2, 2}, -1.0f, 1.0f, /*seed=*/60, /*requires_grad=*/true);
    Tensor flat = flatten(x);
    Tensor same = identity(x);
    std::mt19937 rng(0);
    Tensor passthrough = dropout(x, 0.5f, /*training=*/false, rng);

    REQUIRE(flat.shape() == std::vector<size_t>{2, 12});
    REQUIRE(flat.impl()->storage == x.impl()->storage);
    REQUIRE(same.impl()->storage == x.impl()->storage);
    REQUIRE(passthrough.impl()->storage == x.impl()->storage);

    Tensor loss = mse_loss(flat, Tensor::zeros({2, 12}));
    loss.backward();
    REQUIRE(x.grad().shape() == x.shape());
    for (size_t i = 0; i < x.numel(); ++i) {
        REQUIRE(x.grad().data()[i] == Catch::Approx(2.0f * x.data()[i] / 24.0f));
    }
}
