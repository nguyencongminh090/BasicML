#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace advanceml {

class Tensor;

/**
 * One recorded operation in the autograd graph.
 *
 * `inputs` are the tensors the op read; `backward_fn` maps the
 * upstream gradient (w.r.t. this node's output) to one gradient per
 * entry in `inputs`, in the same order. Mirrors BasicML's
 * "forward() caches what backward() needs" convention, except the
 * caching and chaining here happen automatically per op rather than
 * being hand-written per layer.
 */
struct Node {
    std::vector<std::shared_ptr<class TensorImpl>> inputs;
    std::function<std::vector<Tensor>(const Tensor& grad_output)> backward_fn;
};

/**
 * Storage + graph-node bookkeeping backing a Tensor value handle.
 *
 * `grad_fn` is null for leaves (parameters, inputs) and set by
 * whichever op produced this tensor. `grad` is allocated lazily, on
 * first accumulation during backward().
 */
class TensorImpl {
public:
    std::vector<float> data;
    std::vector<size_t> shape;
    std::shared_ptr<TensorImpl> grad;
    bool requires_grad = false;
    std::shared_ptr<Node> grad_fn;
};

/**
 * An autograd-tracked, dense float32 tensor (row-major, 1D or 2D in
 * this milestone).
 *
 * Ops (`operator+`, `matmul`, `relu`, `mse_loss`, ...) build a
 * define-by-run computation graph as they execute: each produces a
 * new Tensor whose `TensorImpl::grad_fn` records how to route the
 * upstream gradient back to its inputs. Calling `backward()` on a
 * scalar output tensor walks that graph via the chain rule and
 * accumulates into each leaf parameter's `.grad`, in contrast to
 * BasicML's hand-written per-layer `backward()`.
 */
class Tensor {
public:
    /** Constructs a tensor from row-major `data` with the given `shape`. */
    Tensor(std::vector<float> data, std::vector<size_t> shape, bool requires_grad = false);

    /** Returns a `shape`-shaped tensor of zeros. */
    static Tensor zeros(std::vector<size_t> shape, bool requires_grad = false);

    /** Returns a `shape`-shaped tensor with entries drawn uniformly from `[low, high)`. */
    static Tensor random_uniform(std::vector<size_t> shape, float low, float high, unsigned seed, bool requires_grad = false);

    /** Wraps an existing `TensorImpl` (used by op implementations to attach a `grad_fn`). */
    static Tensor from_impl(std::shared_ptr<TensorImpl> impl);

    /** @returns The total number of elements (product of `shape()`). */
    [[nodiscard]] size_t numel() const noexcept;

    [[nodiscard]] const std::vector<size_t>& shape() const noexcept;
    [[nodiscard]] std::vector<float>& data() noexcept;
    [[nodiscard]] const std::vector<float>& data() const noexcept;
    [[nodiscard]] bool requires_grad() const noexcept;

    /** @returns Whether backward() has accumulated a gradient into this (leaf) tensor. */
    [[nodiscard]] bool has_grad() const noexcept;

    /** @returns The accumulated gradient. @throws std::runtime_error if `has_grad()` is false. */
    [[nodiscard]] Tensor grad() const;

    /** Resets the accumulated gradient to unset (equivalent to BasicML's `zero_grad()`). */
    void zero_grad();

    /**
     * Runs backward from this (scalar) tensor, accumulating dL/dx
     * into `.grad` for every leaf tensor with `requires_grad() ==
     * true` reachable through the graph.
     *
     * @throws std::runtime_error if this tensor is not a scalar (numel() != 1).
     */
    void backward();

    [[nodiscard]] const std::shared_ptr<TensorImpl>& impl() const noexcept;

private:
    Tensor() = default;

    std::shared_ptr<TensorImpl> impl_;
};

/**
 * Elementwise or bias-broadcast addition.
 *
 * Either `a.shape() == b.shape()` (elementwise), or `a` is 2D
 * `(N, D)` and `b` is 1D `(D)` (row-broadcast, the bias-add case).
 * Backward: `grad_a = grad_output`; `grad_b` is `grad_output` itself
 * in the elementwise case, or its column-sum in the broadcast case.
 */
Tensor operator+(const Tensor& a, const Tensor& b);

/**
 * Elementwise (Hadamard) product; requires `a.shape() == b.shape()`.
 * Backward: `grad_a = grad_output * b`, `grad_b = grad_output * a`.
 */
Tensor operator*(const Tensor& a, const Tensor& b);

}  // namespace advanceml
