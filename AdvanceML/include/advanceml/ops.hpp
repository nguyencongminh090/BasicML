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

}  // namespace advanceml
