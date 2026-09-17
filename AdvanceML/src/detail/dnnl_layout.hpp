#pragma once

#include "advanceml/tensor.hpp"

#include <dnnl.hpp>

#include <memory>
#include <vector>

/**
 * Private glue between `Tensor` storage and oneDNN memory objects: the shared
 * CPU engine/stream, layout interning, cached reorders, and plain <-> blocked
 * conversions. Not installed as a public header, so users never need
 * `dnnl.hpp` to use `Tensor`.
 */
namespace advanceml::detail {

/**
 * A oneDNN memory descriptor, interned: every distinct descriptor maps to one
 * `Layout` object for the life of the process, so its address is a stable,
 * cheap-to-compare cache key.
 */
struct Layout {
    dnnl::memory::desc desc;
};

/** @return The process-wide oneDNN CPU engine. */
dnnl::engine& cpu_engine();

/** @return The process-wide oneDNN CPU stream (synchronous use: execute, then `wait()`). */
dnnl::stream& cpu_stream();

/** @return `shape` as oneDNN dims; an empty (scalar) shape maps to `{1}`. */
dnnl::memory::dims to_dims(const std::vector<size_t>& shape);

/** @return The plain row-major (`a`, `ab`, `abc`, ... up to 6D) f32 descriptor for `dims`. */
dnnl::memory::desc plain_desc(const dnnl::memory::dims& dims);

/** @return The interned `Layout` equal to `desc` (plain descriptors included). */
const Layout* layout_id(const dnnl::memory::desc& desc);

/** @return The interned `Layout` for `desc`, or null when `desc` is plain row-major for its dims. */
std::shared_ptr<const Layout> storage_layout(const dnnl::memory::desc& desc);

/** Copies `src` into `dst` (same dims, any layouts) with a reorder primitive cached per layout pair. */
void reorder(const dnnl::memory& src, const dnnl::memory& dst);

/** Converts a blocked `storage` to plain row-major in place and clears its layout; no-op when already plain. */
void materialize_plain(Storage& storage);

/**
 * @return The descriptor of `t`'s storage as seen through `t.shape()`. If
 * the storage is blocked but `t` views it with different dims (e.g.
 * `flatten` of a blocked tensor), the storage is materialized to plain first.
 */
dnnl::memory::desc desc_of(const Tensor& t);

/** `desc_of` for a bare storage viewed with `dims` (used by backward functions that capture storage, not a tensor). */
dnnl::memory::desc desc_of(Storage& storage, const dnnl::memory::dims& dims);

/** @return A non-owning oneDNN memory over `storage`, described by `desc_of(storage, dims)`. */
dnnl::memory memory_of(Storage& storage, const dnnl::memory::dims& dims);

/** @return A non-owning oneDNN memory over `t`'s storage, described by `desc_of(t)`. */
dnnl::memory memory_of(const Tensor& t);

/**
 * @return `m` itself when it already has descriptor `want`, else a new
 * (oneDNN-owned) memory holding `m` reordered into `want`.
 */
dnnl::memory convert(const dnnl::memory& m, const dnnl::memory::desc& want);

/** @return `convert(memory_of(t), want)`. */
dnnl::memory memory_as(const Tensor& t, const dnnl::memory::desc& want);

/** @return An uninitialized buffer large enough for `desc` (including any block padding). */
FloatBuffer buffer_for(const dnnl::memory::desc& desc);

/** @return A tensor of `shape` over `data`, whose layout is `layout` (null = plain). */
Tensor make_tensor(FloatBuffer data, std::vector<size_t> shape, std::shared_ptr<const Layout> layout);

}  // namespace advanceml::detail
