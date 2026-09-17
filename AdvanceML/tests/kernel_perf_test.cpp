#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

#include "advanceml/ops.hpp"
#include "advanceml/tensor.hpp"

using namespace advanceml;

// Tests for TODO-0045's op/kernel changes: the fused softmax-cross-entropy op, transpose-free
// matmul/linear backward, aligned tensor buffers, and oneDNN-blocked layouts flowing between ops.

namespace {

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
    Tensor grad = input.grad();
    for (size_t i = 0; i < input.numel(); ++i) {
        const float analytic = grad.data()[i];
        const float numeric = numerical_grad(input.mutable_data()[i], loss_fn, eps);
        REQUIRE(analytic == Catch::Approx(numeric).margin(tolerance));
    }
}

void require_close(const Tensor& actual, const Tensor& expected, float margin) {
    REQUIRE(actual.shape() == expected.shape());
    for (size_t i = 0; i < expected.numel(); ++i) {
        REQUIRE(actual.data()[i] == Catch::Approx(expected.data()[i]).margin(margin));
    }
}

Tensor one_hot_rows(const std::vector<size_t>& labels, size_t num_classes) {
    FloatBuffer data(labels.size() * num_classes, 0.0f);
    for (size_t r = 0; r < labels.size(); ++r) {
        data[r * num_classes + labels[r]] = 1.0f;
    }
    return Tensor(std::move(data), {labels.size(), num_classes});
}

// Row-major transpose, used only to build reference values independently of the op under test.
Tensor transposed(const Tensor& t) {
    const size_t rows = t.shape()[0];
    const size_t cols = t.shape()[1];
    FloatBuffer out(rows * cols);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            out[c * rows + r] = t.data()[r * cols + c];
        }
    }
    return Tensor(std::move(out), {cols, rows});
}

}  // namespace

TEST_CASE("softmax_cross_entropy matches softmax followed by cross_entropy_loss", "[ops][loss]") {
    Tensor logits = Tensor::random_uniform({4, 5}, -2.0f, 2.0f, /*seed=*/450);
    Tensor target = one_hot_rows({0, 3, 4, 1}, 5);

    const float fused = softmax_cross_entropy(logits, target).data()[0];
    const float reference = cross_entropy_loss(softmax(logits), target).data()[0];
    REQUIRE(fused == Catch::Approx(reference).margin(1e-5f));
}

TEST_CASE("softmax_cross_entropy backward matches numerical gradients w.r.t. logits and target", "[ops][loss]") {
    Tensor logits = Tensor::random_uniform({3, 4}, -2.0f, 2.0f, /*seed=*/451, /*requires_grad=*/true);
    // A soft target that sums to 1 per row and needs a gradient, so both backward branches run.
    Tensor target({0.1f, 0.2f, 0.3f, 0.4f, 0.0f, 1.0f, 0.0f, 0.0f, 0.25f, 0.25f, 0.25f, 0.25f}, {3, 4},
                  /*requires_grad=*/true);

    auto forward_loss = [&]() { return softmax_cross_entropy(logits, target); };
    forward_loss().backward();

    constexpr float eps = 1e-2f;
    constexpr float tolerance = 2e-3f;
    check_input_gradient(logits, [&]() { return forward_loss().data()[0]; }, eps, tolerance);
    check_input_gradient(target, [&]() { return forward_loss().data()[0]; }, eps, tolerance);
}

TEST_CASE("softmax_cross_entropy stays finite for extreme logits and its gradient is (softmax - target) / N",
          "[ops][loss]") {
    const FloatBuffer values = {1000.0f, -1000.0f, 0.0f, -500.0f, 800.0f, 799.0f};
    Tensor logits(values, {2, 3}, /*requires_grad=*/true);
    Tensor target = one_hot_rows({1, 2}, 3);

    Tensor loss = softmax_cross_entropy(logits, target);
    REQUIRE(std::isfinite(loss.data()[0]));
    // Row 0's target logit sits 2000 below the max (softmax + clipped CE would cap this at
    // -log(1e-15) ~= 34.5); row 1's sits 1 below it.
    const float row1 = 1.0f + std::log1p(std::exp(-1.0f));
    REQUIRE(loss.data()[0] == Catch::Approx((2000.0f + row1) / 2.0f).margin(1e-2f));

    loss.backward();
    Tensor probs = softmax(Tensor(values, {2, 3}));
    for (size_t i = 0; i < 6; ++i) {
        REQUIRE(logits.grad().data()[i] == Catch::Approx((probs.data()[i] - target.data()[i]) / 2.0f).margin(1e-6f));
    }
}

TEST_CASE("matmul backward without transpose copies matches numerical gradients and the explicit formulas",
          "[ops][matmul]") {
    Tensor a = Tensor::random_uniform({3, 5}, -1.0f, 1.0f, /*seed=*/452, /*requires_grad=*/true);
    Tensor b = Tensor::random_uniform({5, 2}, -1.0f, 1.0f, /*seed=*/453, /*requires_grad=*/true);
    Tensor target = Tensor::random_uniform({3, 2}, -1.0f, 1.0f, /*seed=*/454);

    auto forward_loss = [&]() { return mse_loss(matmul(a, b), target); };
    forward_loss().backward();

    // mse_loss's dL/dC = 2 * (C - T) / numel; then dL/dA = dL/dC @ B^T and dL/dB = A^T @ dL/dC.
    NoGradGuard no_grad;
    Tensor c = matmul(a, b);
    FloatBuffer dc(c.numel());
    for (size_t i = 0; i < c.numel(); ++i) {
        dc[i] = 2.0f * (c.data()[i] - target.data()[i]) / static_cast<float>(c.numel());
    }
    Tensor grad_c(std::move(dc), c.shape());
    require_close(a.grad(), matmul(grad_c, transposed(b)), 1e-6f);
    require_close(b.grad(), matmul(transposed(a), grad_c), 1e-6f);

    constexpr float eps = 1e-2f;
    constexpr float tolerance = 2e-3f;
    check_input_gradient(a, [&]() { return forward_loss().data()[0]; }, eps, tolerance);
    check_input_gradient(b, [&]() { return forward_loss().data()[0]; }, eps, tolerance);
}

TEST_CASE("linear (fused bias) matches matmul + operator+ in value and gradients", "[ops][matmul]") {
    Tensor x = Tensor::random_uniform({4, 3}, -1.0f, 1.0f, /*seed=*/455, /*requires_grad=*/true);
    Tensor w = Tensor::random_uniform({3, 6}, -1.0f, 1.0f, /*seed=*/456, /*requires_grad=*/true);
    Tensor b = Tensor::random_uniform({6}, -1.0f, 1.0f, /*seed=*/457, /*requires_grad=*/true);
    Tensor target = Tensor::random_uniform({4, 6}, -1.0f, 1.0f, /*seed=*/458);

    Tensor fused = linear(x, w, b);
    mse_loss(fused, target).backward();
    const Tensor fused_x_grad = x.grad();
    const Tensor fused_w_grad = w.grad();
    const Tensor fused_b_grad = b.grad();

    x.zero_grad();
    w.zero_grad();
    b.zero_grad();
    Tensor reference = matmul(x, w) + b;
    mse_loss(reference, target).backward();

    require_close(fused, reference, 1e-6f);
    require_close(fused_x_grad, x.grad(), 1e-6f);
    require_close(fused_w_grad, w.grad(), 1e-6f);
    require_close(fused_b_grad, b.grad(), 1e-6f);
}

TEST_CASE("tensor buffers are 64-byte aligned", "[tensor][buffer]") {
    for (size_t n : {1, 3, 17, 1000}) {
        Tensor t = Tensor::zeros({n});
        REQUIRE(reinterpret_cast<std::uintptr_t>(t.data().data()) % 64 == 0);
    }
}

TEST_CASE("a conv -> batch_norm -> relu -> pool chain gives the same values and gradients blocked as plain",
          "[ops][layout]") {
    // 16 channels, so on AVX-512 machines oneDNN picks the blocked nChw16c layout for every op
    // in the chain; the reference run forces plain NCHW between ops by reading each output.
    Tensor x = Tensor::random_uniform({2, 3, 8, 8}, -1.0f, 1.0f, /*seed=*/460, /*requires_grad=*/true);
    Tensor w1 = Tensor::random_uniform({16, 3, 3, 3}, -0.5f, 0.5f, /*seed=*/461, /*requires_grad=*/true);
    Tensor b1 = Tensor::random_uniform({16}, -0.5f, 0.5f, /*seed=*/462, /*requires_grad=*/true);
    Tensor w2 = Tensor::random_uniform({16, 16, 3, 3}, -0.5f, 0.5f, /*seed=*/463, /*requires_grad=*/true);
    Tensor b2 = Tensor::random_uniform({16}, -0.5f, 0.5f, /*seed=*/464, /*requires_grad=*/true);
    Tensor gamma = Tensor::random_uniform({16}, 0.5f, 1.5f, /*seed=*/465, /*requires_grad=*/true);
    Tensor beta = Tensor::random_uniform({16}, -0.5f, 0.5f, /*seed=*/466, /*requires_grad=*/true);
    Tensor target = one_hot_rows({3, 7}, 16);
    const std::vector<Tensor> params = {x, w1, b1, w2, b2, gamma, beta};

    auto run = [&](bool force_plain, std::vector<Tensor>& grads) {
        auto step = [force_plain](Tensor t) {
            if (force_plain) {
                (void)t.data();
                REQUIRE(t.has_plain_layout());
            }
            return t;
        };
        for (Tensor p : params) {
            p.zero_grad();
        }
        std::vector<float> running_mean(16, 0.0f);
        std::vector<float> running_var(16, 1.0f);
        Tensor h = step(conv2d(x, w1, b1, /*stride=*/1, /*padding=*/1));
        h = step(batch_norm2d(h, gamma, beta, running_mean, running_var, /*training=*/true));
        h = step(relu(h));
        h = step(max_pool2d(h, /*kernel_size=*/2, /*stride=*/2));
        // h is used twice, so its gradient is the sum of a (possibly blocked) conv gradient and a
        // plain one: exercises accumulation across layouts.
        Tensor skip = global_avg_pool2d(h);
        h = step(conv2d(h, w2, b2, /*stride=*/1, /*padding=*/1));
        h = step(relu(h));
        h = step(global_avg_pool2d(h));
        Tensor logits = flatten(h) + flatten(skip);
        Tensor loss = softmax_cross_entropy(logits, target);
        loss.backward();
        grads.clear();
        for (const Tensor& p : params) {
            grads.push_back(Tensor(p.grad().data(), p.shape()));
        }
        return std::vector<float>{loss.data()[0], running_mean[5], running_var[5]};
    };

    std::vector<Tensor> blocked_grads;
    std::vector<Tensor> plain_grads;
    const std::vector<float> blocked = run(false, blocked_grads);
    const std::vector<float> plain = run(true, plain_grads);

    for (size_t i = 0; i < blocked.size(); ++i) {
        REQUIRE(blocked[i] == Catch::Approx(plain[i]).margin(1e-5f));
    }
    for (size_t i = 0; i < params.size(); ++i) {
        require_close(blocked_grads[i], plain_grads[i], 1e-5f);
    }
}

TEST_CASE("reading a blocked tensor converts it to plain in place without bumping its version", "[ops][layout]") {
    Tensor x = Tensor::random_uniform({1, 16, 4, 4}, -1.0f, 1.0f, /*seed=*/470);
    Tensor w = Tensor::random_uniform({16, 16, 1, 1}, -1.0f, 1.0f, /*seed=*/471);
    Tensor b = Tensor::zeros({16});
    Tensor y = conv2d(x, w, b, /*stride=*/1, /*padding=*/0);
    Tensor view = flatten(y);
    const uint64_t version = y.version();

    // 1x1 conv: y[n, o, i, j] = sum_c w[o, c] * x[n, c, i, j].
    const FloatBuffer& values = view.data();
    REQUIRE(y.has_plain_layout());
    REQUIRE(y.version() == version);
    REQUIRE(values.size() == 16 * 16);
    for (size_t o = 0; o < 16; ++o) {
        for (size_t p = 0; p < 16; ++p) {
            float expected = 0.0f;
            for (size_t c = 0; c < 16; ++c) {
                expected += w.data()[o * 16 + c] * x.data()[c * 16 + p];
            }
            REQUIRE(values[o * 16 + p] == Catch::Approx(expected).margin(1e-4f));
        }
    }
}

TEST_CASE("relu falls back to a scalar loop for tensors beyond oneDNN's 6D descriptor limit",
          "[ops][layout][relu]") {
    // oneDNN's plain_desc only covers 1D-6D; 7D exercises TODO-0047 P4's fallback instead of
    // relu throwing.
    Tensor x = Tensor::random_uniform({2, 1, 1, 1, 1, 1, 3}, -1.0f, 1.0f, /*seed=*/490, /*requires_grad=*/true);

    Tensor y = relu(x);
    REQUIRE(y.shape() == x.shape());
    for (size_t i = 0; i < x.numel(); ++i) {
        const float expected = x.data()[i] > 0.0f ? x.data()[i] : 0.0f;
        REQUIRE(y.data()[i] == Catch::Approx(expected).margin(1e-6f));
    }

    auto forward_loss = [&]() { return mse_loss(relu(x), Tensor::zeros(x.shape())); };
    forward_loss().backward();

    constexpr float eps = 1e-2f;
    constexpr float tolerance = 2e-3f;
    check_input_gradient(x, [&]() { return forward_loss().data()[0]; }, eps, tolerance);
}

TEST_CASE("batch_norm2d in eval mode backward treats running statistics as constants", "[ops][batchnorm]") {
    Tensor x = Tensor::random_uniform({2, 3, 4, 4}, -1.0f, 1.0f, /*seed=*/480, /*requires_grad=*/true);
    Tensor gamma = Tensor::random_uniform({3}, 0.5f, 1.5f, /*seed=*/481, /*requires_grad=*/true);
    Tensor beta = Tensor::random_uniform({3}, -0.5f, 0.5f, /*seed=*/482, /*requires_grad=*/true);
    std::vector<float> running_mean = {0.1f, -0.2f, 0.3f};
    std::vector<float> running_var = {0.5f, 1.5f, 2.0f};

    auto forward_loss = [&]() {
        Tensor y = batch_norm2d(x, gamma, beta, running_mean, running_var, /*training=*/false);
        return mse_loss(y, Tensor::zeros(y.shape()));
    };
    forward_loss().backward();

    constexpr float eps = 1e-2f;
    constexpr float tolerance = 2e-3f;
    check_input_gradient(x, [&]() { return forward_loss().data()[0]; }, eps, tolerance);
    check_input_gradient(gamma, [&]() { return forward_loss().data()[0]; }, eps, tolerance);
    check_input_gradient(beta, [&]() { return forward_loss().data()[0]; }, eps, tolerance);
}
