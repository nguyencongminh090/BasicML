#pragma once

#include "advanceml/tensor.hpp"

#include <random>
#include <vector>

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
 * Mean absolute error: `loss = mean(|pred - target|)`, reduced to a scalar.
 * Backward: `dL/d(pred) = sign(pred - target) / N`, `dL/d(target) = -sign(pred - target) / N`,
 * where `sign(0) = 0` (a subgradient choice at the non-differentiable point).
 *
 * @throws std::runtime_error if `pred.shape() != target.shape()`.
 */
Tensor abs_loss(const Tensor& pred, const Tensor& target);

/**
 * Binary cross-entropy loss for already-normalized probabilities, matching
 * `basicml.nn.loss.BinaryCrossEntropy`'s contract: `pred` is expected to
 * already be a probability per element (e.g. the output of `sigmoid`), and
 * `target` a `{0, 1}`-valued tensor of the same shape.
 * `loss = -mean_i(sum_j(target_ij * log(clip(pred_ij)) + (1 - target_ij) *
 * log(1 - clip(pred_ij))))`, where the mean is over the batch dimension `i`
 * and `clip` keeps `pred` in `[1e-15, 1 - 1e-15]` to avoid `log(0)`.
 * Backward: `dL/d(pred_ij) = -(1/N) * (target_ij / clip(pred_ij) - (1 -
 * target_ij) / (1 - clip(pred_ij)))`.
 *
 * @throws std::runtime_error if `pred.shape() != target.shape()`, or the shape is neither 1D nor 2D.
 */
Tensor binary_cross_entropy(const Tensor& pred, const Tensor& target);

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
 * Hyperbolic tangent, elementwise: `tanh(x) = (e^x - e^-x) / (e^x + e^-x)`.
 * Backward: `dL/dx = dL/dy * (1 - y^2)`, reusing the forward output `y`.
 */
Tensor tanh(const Tensor& x);

/**
 * Identity, elementwise: `identity(x) = x`. Backward: `dL/dx = dL/dy`.
 */
Tensor identity(const Tensor& x);

/**
 * Parametric rectified linear unit, elementwise: `prelu(x) = x` where `x >
 * 0`, else `a_c * x`, where `c` is `0` if `a` has one entry (a single
 * shared slope), else `i % a.numel()` for flat index `i` (broadcasting
 * `a` over `x`'s trailing axis, matching `basicml.nn.activation.PReLU`).
 * Backward: `dL/dx = dL/dy` where `x > 0`, else `a_c * dL/dy`; `dL/da_c =
 * sum` over elements sharing slope `c`, of `dL/dy * x` where `x <= 0`.
 *
 * @throws std::runtime_error if `a.numel()` is neither `1` nor `x.shape().back()`.
 */
Tensor prelu(const Tensor& x, const Tensor& a);

/**
 * Exponential linear unit, elementwise: `elu(x) = x` where `x > 0`, else
 * `alpha * (exp(x) - 1)`.
 * Backward: `dL/dx = dL/dy` where `x > 0`, else `dL/dy * alpha * exp(x)`.
 */
Tensor elu(const Tensor& x, float alpha = 1.0f);

/**
 * Scaled exponential linear unit, elementwise, with the fixed constants
 * from Klambauer et al.: `selu(x) = scale * x` where `x > 0`, else `scale
 * * alpha * (exp(x) - 1)`, `alpha ~= 1.6732632423543772`, `scale ~=
 * 1.0507009873554805`.
 * Backward: `dL/dx = dL/dy * scale` where `x > 0`, else `dL/dy * scale *
 * alpha * exp(x)`.
 */
Tensor selu(const Tensor& x);

/**
 * Softplus, elementwise, numerically stable: `softplus(x) = log(1 +
 * exp(x))`, computed as `max(x, 0) + log1p(exp(-|x|))`.
 * Backward: `dL/dx = dL/dy * sigmoid(x)`.
 */
Tensor softplus(const Tensor& x);

/**
 * Swish (SiLU when `beta = 1`), elementwise: `swish(x) = x * sigmoid(beta *
 * x)`.
 * Backward: `dL/dx = dL/dy * (sig + beta * x * sig * (1 - sig))`, where
 * `sig = sigmoid(beta * x)`.
 */
Tensor swish(const Tensor& x, float beta = 1.0f);

/**
 * Mish, elementwise: `mish(x) = x * tanh(softplus(x))`.
 * Backward: `dL/dx = dL/dy * (t + x * (1 - t^2) * sigmoid(x))`, where `t =
 * tanh(softplus(x))`.
 */
Tensor mish(const Tensor& x);

/**
 * Hard tanh, elementwise: `hardtanh(x) = clip(x, min_val, max_val)`.
 * Backward: `dL/dx = dL/dy` where `min_val < x < max_val`, else `0`.
 *
 * @throws std::runtime_error if `max_val <= min_val`.
 */
Tensor hardtanh(const Tensor& x, float min_val = -1.0f, float max_val = 1.0f);

/**
 * Hard sigmoid, elementwise, the piecewise-linear sigmoid approximation:
 * `hardsigmoid(x) = clip(x / 6 + 0.5, 0, 1)`.
 * Backward: `dL/dx = dL/dy / 6` where `-3 < x < 3`, else `0`.
 */
Tensor hardsigmoid(const Tensor& x);

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

/**
 * 2D convolution: `x` is `(N, in_channels, H, W)`, `weight` is
 * `(out_channels, in_channels, kh, kw)`, `bias` is `(out_channels)`,
 * output is `(N, out_channels, Hout, Wout)` with
 * `Hout = (H + 2*padding - kh) / stride + 1` (`Wout` analogous).
 *
 * Forward runs through oneDNN's `dnnl::convolution_forward` primitive;
 * backward (w.r.t. `x`, `weight`, and `bias`) through
 * `dnnl::convolution_backward_data` and `convolution_backward_weights`.
 *
 * @param stride Stride applied to both spatial dimensions.
 * @param padding Zero-padding applied to both spatial dimensions, both sides.
 * @throws std::runtime_error if `x`/`weight` are not 4D, `bias` is not 1D
 * with `out_channels` entries, `x`'s channel count doesn't match
 * `weight`'s `in_channels`, or the padded input is smaller than the kernel.
 */
Tensor conv2d(const Tensor& x, const Tensor& weight, const Tensor& bias, size_t stride, size_t padding);

/**
 * 2D max pooling over a `(N, C, H, W)` tensor: each output element is the
 * max over its `kernel_size x kernel_size` window, strided by `stride`.
 * Forward/backward run through oneDNN's `dnnl::pooling_forward` /
 * `pooling_backward` (`pooling_max` algorithm); backward routes the
 * upstream gradient to each window's argmax via oneDNN's workspace.
 *
 * @throws std::runtime_error if `x` is not 4D or `kernel_size` exceeds `H` or `W`.
 */
Tensor max_pool2d(const Tensor& x, size_t kernel_size, size_t stride);

/**
 * 2D average pooling over a `(N, C, H, W)` tensor: each output element is
 * the mean over its `kernel_size x kernel_size` window, strided by
 * `stride`. Forward/backward run through oneDNN's `dnnl::pooling_forward`
 * / `pooling_backward` (`pooling_avg_exclude_padding` algorithm).
 *
 * @throws std::runtime_error if `x` is not 4D or `kernel_size` exceeds `H` or `W`.
 */
Tensor avg_pool2d(const Tensor& x, size_t kernel_size, size_t stride);

/**
 * Flattens every dimension but the first (batch) one: `(N, ...)` becomes
 * `(N, prod(...))`. A pure reshape: the underlying data is copied
 * unchanged, and backward reshapes the upstream gradient back to `x`'s
 * original shape.
 *
 * @throws std::runtime_error if `x` has no dimensions.
 */
Tensor flatten(const Tensor& x);

/**
 * 2D batch normalization over a `(N, C, H, W)` tensor, normalizing each
 * channel across the `N`, `H`, `W` axes:
 * `y = gamma * (x - mean) / sqrt(var + eps) + beta`.
 *
 * When `training` is true, `mean`/`var` are the current batch's
 * per-channel statistics, and `running_mean`/`running_var` are updated in
 * place: `running = (1 - momentum) * running + momentum * batch`. When
 * `training` is false (inference), `running_mean`/`running_var` are used
 * as `mean`/`var` directly and left untouched. Backward differs between
 * the two modes accordingly: in training mode it accounts for `mean`/`var`
 * depending on `x`; in inference mode it treats them as constants.
 *
 * @throws std::runtime_error if `x` is not 4D, `gamma`/`beta` are not 1D
 * with `C` entries, or `running_mean`/`running_var` do not have `C` entries.
 */
Tensor batch_norm2d(const Tensor& x, const Tensor& gamma, const Tensor& beta, std::vector<float>& running_mean,
                     std::vector<float>& running_var, bool training, float momentum = 0.1f, float eps = 1e-5f);

/**
 * 1D batch normalization over a `(N, C)` tensor, normalizing each feature
 * across the batch axis `N`: `y = gamma * (x - mean) / sqrt(var + eps) +
 * beta`. Same semantics as `batch_norm2d` with `H = W = 1`.
 *
 * @throws std::runtime_error if `x` is not 2D, `gamma`/`beta` are not 1D
 * with `C` entries, or `running_mean`/`running_var` do not have `C` entries.
 */
Tensor batch_norm1d(const Tensor& x, const Tensor& gamma, const Tensor& beta, std::vector<float>& running_mean,
                     std::vector<float>& running_var, bool training, float momentum = 0.1f, float eps = 1e-5f);

/**
 * Global average pooling over a `(N, C, H, W)` tensor's spatial dimensions:
 * output is `(N, C, 1, 1)`, `out[n, c] = mean_{h, w}(x[n, c, h, w])`.
 * Backward broadcasts the upstream gradient back evenly: `dL/dx[n, c, h, w]
 * = dL/d(out[n, c]) / (H * W)`.
 *
 * @throws std::runtime_error if `x` is not 4D.
 */
Tensor global_avg_pool2d(const Tensor& x);

/**
 * Global max pooling over a `(N, C, H, W)` tensor's spatial dimensions:
 * output is `(N, C, 1, 1)`, `out[n, c] = max_{h, w}(x[n, c, h, w])`.
 * Backward routes the whole upstream gradient to each `(n, c)` slice's
 * argmax location, zero elsewhere.
 *
 * @throws std::runtime_error if `x` is not 4D.
 */
Tensor global_max_pool2d(const Tensor& x);

/**
 * Inverted dropout, elementwise, over any shape: in training mode, each
 * element is independently zeroed with probability `p` (`rng` draws the
 * per-element decisions) and the survivors are scaled by `1 / (1 - p)` so
 * the output's expectation matches `x`; in inference mode (`training ==
 * false`) or when `p == 0`, `x` passes through unchanged. Backward applies
 * the same mask/scale used in forward (or the identity, in the unchanged
 * case).
 *
 * @throws std::runtime_error if `p` is not in `[0, 1)`.
 */
Tensor dropout(const Tensor& x, float p, bool training, std::mt19937& rng);

}  // namespace advanceml
