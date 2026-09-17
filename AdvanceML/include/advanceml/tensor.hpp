#pragma once

#include "advanceml/buffer.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace advanceml {

class Tensor;
class TensorImpl;

namespace detail {
/** An interned oneDNN memory layout (defined in the implementation; opaque to users). */
struct Layout;
}  // namespace detail

/** The largest number of inputs (and of saved tensors) any single op records. */
inline constexpr size_t kMaxNodeInputs = 3;

/** One flag per recorded input: whether that input needs a gradient. */
using NeedsInputGrad = std::array<bool, kMaxNodeInputs>;

/**
 * One optional gradient per recorded input, in input order.
 * `std::nullopt` means "not computed"; a backward function leaves an entry
 * empty when the matching `NeedsInputGrad` flag is false.
 */
using InputGrads = std::array<std::optional<Tensor>, kMaxNodeInputs>;

/**
 * The flat float buffer behind one or more tensors.
 *
 * Several `TensorImpl`s may share one `Storage` (a reshape view such as
 * `flatten` shares its input's buffer). `version` counts in-place writes made
 * through `Tensor::mutable_data()`, so backward can detect that a tensor it
 * saved was modified after being saved.
 *
 * `layout` is null when `data` is plain row-major (NCHW for 4D tensors).
 * oneDNN-backed ops (`conv2d`, `batch_norm2d`, `relu`, pooling) may instead
 * return tensors whose `data` is in a oneDNN-optimized layout such as
 * `nChw16c` (channels grouped in blocks of 16, possibly zero-padded, so
 * `data.size()` can exceed the element count). Chains of those ops pass the
 * blocked buffer along without converting it. The first plain read through
 * `Tensor::data()` or `Tensor::mutable_data()` converts the storage to plain
 * row-major in place and clears `layout`; the logical values, and `version`,
 * do not change.
 */
struct Storage {
    FloatBuffer data;
    uint64_t version = 0;
    std::shared_ptr<const detail::Layout> layout;
};

/** A storage an op saved for backward, plus its version at save time. */
struct SavedVersion {
    std::shared_ptr<Storage> storage;
    uint64_t version = 0;
};

/**
 * One recorded operation in the autograd graph.
 *
 * `inputs[0..num_inputs)` are the tensors the op read, and
 * `needs_input_grad[i]` records whether `inputs[i]` required a gradient when
 * the op ran; backward never computes or routes gradients to inputs with the
 * flag cleared. `apply()` maps the upstream gradient (w.r.t. this node's
 * output) to one optional gradient per input, in the same order.
 *
 * Concrete nodes are created by `detail::record()`, which stores the op's
 * backward lambda and its saved state in the same allocation as the node.
 * After a non-retaining `Tensor::backward()` runs a node, `release()` drops
 * its saved state and input edges; running it again then throws.
 */
class Node {
public:
    Node() = default;
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    /** Tears the upstream graph down iteratively, so very deep graphs cannot overflow the call stack. */
    virtual ~Node();

    std::array<std::shared_ptr<TensorImpl>, kMaxNodeInputs> inputs{};
    NeedsInputGrad needs_input_grad{};
    size_t num_inputs = 0;
    std::array<SavedVersion, kMaxNodeInputs> saved{};
    size_t num_saved = 0;

    /**
     * Computes the gradient w.r.t. each input from `grad_output`, the gradient
     * w.r.t. this node's output.
     *
     * @return One optional gradient per input; entries whose `needs_input_grad` flag is false stay empty.
     */
    [[nodiscard]] virtual InputGrads apply(const Tensor& grad_output) = 0;

    /** Drops the backward state this node saved (closure captures and `saved`), keeping its input edges. */
    virtual void release_saved() noexcept;

    /** Drops saved state and input edges, and marks the node as released. */
    void release() noexcept;

    /** @return Whether `release()` has run. */
    [[nodiscard]] bool released() const noexcept;

    /**
     * @throws std::runtime_error if the node was released, or if any saved
     * storage's version changed since it was saved (an in-place write through
     * `Tensor::mutable_data()` would make the gradient silently wrong).
     */
    void check_saved_versions() const;

private:
    bool released_ = false;
};

/**
 * Shape, storage and graph bookkeeping behind a `Tensor` value handle.
 *
 * `grad_fn` is null for leaves (parameters, inputs) and for any tensor
 * produced while recording was off; otherwise it is the node of the op that
 * produced this tensor. `grad` is a leaf's accumulated gradient, set on first
 * accumulation during backward. `visit_mark` and `pending_grad` are
 * `Tensor::backward()`'s per-pass scratch space (a visited marker for the
 * topological sort and the gradient accumulated so far for this tensor), which
 * replace hash maps keyed on tensor addresses.
 */
class TensorImpl {
public:
    std::shared_ptr<Storage> storage;
    std::vector<size_t> shape;
    std::shared_ptr<TensorImpl> grad;
    bool requires_grad = false;
    std::shared_ptr<Node> grad_fn;
    uint64_t visit_mark = 0;
    std::shared_ptr<TensorImpl> pending_grad;
};

/**
 * @return Whether ops currently record autograd nodes on this thread (true
 * unless a `NoGradGuard` is alive).
 */
[[nodiscard]] bool is_grad_enabled() noexcept;

/**
 * RAII switch that turns graph recording off on the current thread for its
 * lifetime, restoring the previous mode on destruction (guards nest).
 *
 * While it is alive, every op returns a plain tensor with
 * `requires_grad() == false` and no `grad_fn`, so no saved tensors are kept
 * alive, and oneDNN-backed ops use `prop_kind::forward_inference`. Use it for
 * evaluation loops.
 */
class NoGradGuard {
public:
    NoGradGuard() noexcept;
    ~NoGradGuard();
    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    bool previous_;
};

/**
 * An autograd-tracked, dense float32 tensor (row-major).
 *
 * Ops (`operator+`, `matmul`, `relu`, `mse_loss`, ...) build a define-by-run
 * computation graph as they execute: each produces a new Tensor whose
 * `TensorImpl::grad_fn` records how to route the upstream gradient back to its
 * inputs. Calling `backward()` on a scalar output tensor walks that graph via
 * the chain rule and accumulates into each leaf parameter's `.grad`, in
 * contrast to BasicML's hand-written per-layer `backward()`.
 *
 * Copying a Tensor copies the handle, not the data. Reshape views
 * (`flatten`, `identity`, inference-mode `dropout`) share storage with their
 * input, so a write through one is visible through the other.
 *
 * Tensors are not thread-safe: `data()` may convert a oneDNN-blocked storage
 * to plain layout in place (see `Storage`), and ops share process-wide
 * oneDNN primitive caches that are neither locked nor size-bounded (one entry
 * per distinct op shape and layout).
 */
class Tensor {
public:
    /** Constructs a tensor from row-major `data` with the given `shape`. @throws std::runtime_error if the sizes disagree. */
    Tensor(FloatBuffer data, std::vector<size_t> shape, bool requires_grad = false);

    /**
     * Constructs a tensor by copying a standard-allocator `std::vector<float>`
     * into an aligned `FloatBuffer`. (A template, so that a braced list such as
     * `Tensor({1.0f, 2.0f}, {2})` unambiguously picks the `FloatBuffer` overload.)
     *
     * @throws std::runtime_error if the sizes disagree.
     */
    template <std::same_as<std::vector<float>> Vector>
    Tensor(const Vector& data, std::vector<size_t> shape, bool requires_grad = false)
        : Tensor(FloatBuffer(data.begin(), data.end()), std::move(shape), requires_grad) {}

    /** Returns a `shape`-shaped tensor of zeros. */
    static Tensor zeros(std::vector<size_t> shape, bool requires_grad = false);

    /** Returns a `shape`-shaped tensor with entries drawn uniformly from `[low, high)`. */
    static Tensor random_uniform(std::vector<size_t> shape, float low, float high, unsigned seed, bool requires_grad = false);

    /** Wraps an existing `TensorImpl` (used by op implementations). */
    static Tensor from_impl(std::shared_ptr<TensorImpl> impl);

    /** @return The total number of elements (product of `shape()`). */
    [[nodiscard]] size_t numel() const noexcept;

    [[nodiscard]] const std::vector<size_t>& shape() const noexcept;

    /**
     * @return Read-only access to the row-major element buffer.
     *
     * If the storage holds a oneDNN-blocked layout, it is first converted to
     * plain row-major in place (a logical no-op; see `Storage`).
     */
    [[nodiscard]] const FloatBuffer& data() const;

    /**
     * @return Writable access to the row-major element buffer.
     *
     * Counts as an in-place write: bumps the storage version, so a later
     * `backward()` through an op that saved this tensor (or a view sharing its
     * storage) throws instead of computing a wrong gradient. Use `data()` for
     * reads. Converts a oneDNN-blocked storage to plain first, like `data()`.
     */
    [[nodiscard]] FloatBuffer& mutable_data();

    /**
     * @return Whether the storage currently holds plain row-major data (true),
     * or a oneDNN-optimized layout that `data()` would convert first (false).
     */
    [[nodiscard]] bool has_plain_layout() const noexcept;

    /** @return How many times `mutable_data()` was called on this tensor's storage. */
    [[nodiscard]] uint64_t version() const noexcept;

    [[nodiscard]] bool requires_grad() const noexcept;

    /** @return Whether backward() has accumulated a gradient into this (leaf) tensor. */
    [[nodiscard]] bool has_grad() const noexcept;

    /** @return The accumulated gradient. @throws std::runtime_error if `has_grad()` is false. */
    [[nodiscard]] Tensor grad() const;

    /**
     * Resets the accumulated gradient to unset (equivalent to BasicML's
     * `zero_grad()`). The next backward adopts the incoming gradient buffer
     * instead of allocating and copying into a new one.
     */
    void zero_grad();

    /**
     * Runs backward from this (scalar) tensor, accumulating dL/dx into `.grad`
     * for every leaf tensor with `requires_grad() == true` reachable through
     * the graph.
     *
     * Gradients are only computed for inputs that required one when the op
     * ran; each intermediate gradient is freed as soon as the node consuming
     * it has run. Unless `retain_graph` is true, each node's saved tensors and
     * input edges are released right after it runs, so activation memory is
     * freed during the pass and a second backward through the same graph
     * throws.
     *
     * Double backward (gradients of gradients) is not supported: backward
     * functions run with recording off, so the gradients they return carry no
     * graph.
     *
     * @param retain_graph Keep saved tensors and edges so the graph can be backpropagated again.
     * @throws std::runtime_error if this tensor is not a scalar (numel() != 1),
     * if the graph was already released by an earlier backward, or if a tensor
     * saved for backward was modified in place via `mutable_data()`.
     */
    void backward(bool retain_graph = false);

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
