#pragma once

#include "advanceml/tensor.hpp"

namespace advanceml {

/**
 * 2D matrix multiplication: `a` is `(M, K)`, `b` is `(K, N)`, result is `(M, N)`.
 *
 * Forward is computed via oneDNN's `dnnl::matmul` primitive (exercises
 * this CPU's AVX-512 VNNI). Backward, for upstream gradient `dL/dC`:
 * `dL/dA = dL/dC @ B^T`, `dL/dB = A^T @ dL/dC` -- the same two-matmul
 * shape as BasicML's `Linear.backward`.
 *
 * @throws std::runtime_error if shapes are not 2D or the inner dimensions disagree.
 */
Tensor matmul(const Tensor& a, const Tensor& b);

/**
 * Rectified linear unit, elementwise: `relu(x) = max(0, x)`.
 * Backward: `dL/dx = dL/dy` where `x > 0`, else `0`.
 */
Tensor relu(const Tensor& x);

/**
 * Mean squared error: `loss = mean((pred - target)^2)`, reduced to a scalar.
 * Backward: `dL/d(pred) = 2 * (pred - target) / N`.
 *
 * @throws std::runtime_error if `pred.shape() != target.shape()`.
 */
Tensor mse_loss(const Tensor& pred, const Tensor& target);

/**
 * Logistic sigmoid, elementwise: `sigmoid(x) = 1 / (1 + exp(-x))`.
 * Backward: `dL/dx = dL/dy * y * (1 - y)`, reusing the forward output `y`.
 */
Tensor sigmoid(const Tensor& x);

/**
 * Leaky rectified linear unit, elementwise:
 * `leaky_relu(x) = x` where `x > 0`, else `negative_slope * x`.
 * Backward: `dL/dx = dL/dy` where `x > 0`, else `negative_slope * dL/dy`.
 */
Tensor leaky_relu(const Tensor& x, float negative_slope = 0.01f);

/**
 * Gaussian Error Linear Unit, elementwise, exact (erf-based) form:
 * `gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))`.
 * Backward: `dL/dx = dL/dy * (0.5 * (1 + erf(x / sqrt(2))) + x * phi(x))`,
 * where `phi` is the standard normal density.
 */
Tensor gelu(const Tensor& x);

/**
 * Softmax over the last axis: for a 1D tensor, over the whole vector; for a
 * 2D `(N, D)` tensor, independently per row. `softmax(x)_i = exp(x_i) /
 * sum_j(exp(x_j))`, computed with a max-subtraction for numerical stability.
 * Backward (per row): `dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))`.
 *
 * @throws std::runtime_error if `x` is neither 1D nor 2D.
 */
Tensor softmax(const Tensor& x);

/**
 * Cross-entropy loss for already-normalized probabilities, matching
 * `basicml.nn.loss.CrossEntropyLoss`'s contract: unlike a fused
 * softmax-cross-entropy, `pred` is expected to already be a probability
 * distribution per row (e.g. the output of `softmax`), and `target` a
 * one-hot (or otherwise normalized) distribution of the same shape.
 * `loss = -mean_i(sum_j(target_ij * log(clip(pred_ij))))`, where the mean
 * is over the batch dimension `i` and `clip` keeps `pred` in
 * `[1e-15, 1 - 1e-15]` to avoid `log(0)`.
 * Backward: `dL/d(pred_ij) = -(1/N) * target_ij / clip(pred_ij)`.
 *
 * @throws std::runtime_error if `pred.shape() != target.shape()`, or the shape is neither 1D nor 2D.
 */
Tensor cross_entropy_loss(const Tensor& pred, const Tensor& target);

}  // namespace advanceml
