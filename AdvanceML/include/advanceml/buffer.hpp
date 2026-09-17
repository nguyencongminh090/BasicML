#pragma once

#include <cstddef>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace advanceml {

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
