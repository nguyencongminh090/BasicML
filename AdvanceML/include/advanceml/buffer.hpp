#pragma once

#include <cstddef>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__linux__) && defined(__GLIBC__)
#include <malloc.h>
#endif

namespace advanceml {

namespace detail {

/**
 * Fixes glibc's mmap threshold instead of leaving it on the default dynamic policy, where every
 * large free raises the threshold so later same-sized allocations move from mmap (returned to the
 * OS on free) onto the growing heap (freed blocks mid-heap are not). A training loop that
 * repeatedly allocates and frees multi-megabyte activation buffers hits this every epoch, so RSS
 * creeps up over the run even though the live working set is constant. Called once, lazily, from
 * the first buffer allocation.
 *
 * 8 MiB, not a smaller value: a threshold below the largest buffer a hot loop allocates and frees
 * every call (here, an mb128/16-channel/28x28 f32 activation, ~6.1 MiB) routes that recurring
 * allocation through mmap/munmap instead of the heap. That trades a real RSS win for a much larger
 * one: on `train_cnn_mnist`, pinning the threshold at 128 KiB (small enough to force it) cut peak
 * RSS from ~205 MiB to ~148 MiB but made epoch time ~62% worse (16.1s -> 26.0s for 10 epochs,
 * `OMP_NUM_THREADS=4`) -- the mmap/page-fault overhead on every allocate/free of that ~6 MiB buffer
 * dominates the op cost it's part of. 8 MiB sits above that buffer, so training speed is unaffected
 * (16.6s, matching the untuned baseline) while peak RSS still drops to ~185 MiB, since a fixed
 * threshold (any fixed value) stops the default dynamic-threshold growth that caused the creep.
 */
inline void tune_malloc_once() {
#if defined(__linux__) && defined(__GLIBC__)
    static const bool tuned = [] {
        mallopt(M_MMAP_THRESHOLD, 8 * 1024 * 1024);
        return true;
    }();
    (void)tuned;
#endif
}

}  // namespace detail

/**
 * A standard allocator for tensor buffers that
 *
 * - aligns every allocation to `kAlignment` (64) bytes, the width of an
 *   AVX-512 register, so vectorized loops and oneDNN kernels can use aligned
 *   loads on any buffer, and
 * - *default*-initializes elements instead of value-initializing them, so
 *   `FloatBuffer(n)` / `resize(n)` leave the elements uninitialized rather than
 *   zero-filling memory that an op is about to overwrite anyway.
 *
 * Constructing with an explicit value (`FloatBuffer(n, 0.0f)`) or from a
 * range still initializes every element as usual.
 *
 * @tparam T Element type.
 */
template <typename T>
class AlignedDefaultInitAllocator {
public:
    using value_type = T;

    /** Alignment of every allocation, in bytes. */
    static constexpr std::align_val_t kAlignment{64};

    template <typename U>
    struct rebind {
        using other = AlignedDefaultInitAllocator<U>;
    };

    AlignedDefaultInitAllocator() noexcept = default;

    template <typename U>
    AlignedDefaultInitAllocator(const AlignedDefaultInitAllocator<U>&) noexcept {}

    /**
     * @return Storage for `n` elements of `T`, aligned to `kAlignment`.
     * @throws std::bad_array_new_length if `n * sizeof(T)` overflows.
     * @throws std::bad_alloc if the allocation fails.
     */
    [[nodiscard]] T* allocate(size_t n) {
        if (n > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        detail::tune_malloc_once();
        return static_cast<T*>(::operator new(n * sizeof(T), kAlignment));
    }

    /** Releases storage obtained from `allocate(n)`. */
    void deallocate(T* p, size_t n) noexcept {
        ::operator delete(p, n * sizeof(T), kAlignment);
    }

    /** Default-initializes (for `float`: leaves uninitialized) the element at `p`. */
    template <typename U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) {
        ::new (static_cast<void*>(p)) U;
    }

    /** Constructs the element at `p` from `args`. */
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    friend bool operator==(const AlignedDefaultInitAllocator&, const AlignedDefaultInitAllocator&) noexcept {
        return true;
    }
};

/**
 * The element buffer type behind every `Tensor`: a `std::vector<float>` with
 * 64-byte alignment and no zero-fill on sized construction (see
 * `AlignedDefaultInitAllocator`). `FloatBuffer(n)` holds `n` *uninitialized*
 * floats; use `FloatBuffer(n, 0.0f)` when zeros are needed.
 */
using FloatBuffer = std::vector<float, AlignedDefaultInitAllocator<float>>;

}  // namespace advanceml
