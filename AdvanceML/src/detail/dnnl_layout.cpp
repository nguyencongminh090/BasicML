#include "detail/dnnl_layout.hpp"

#include "advanceml/detail/autograd.hpp"

#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace advanceml::detail {

namespace {

size_t hash_combine(size_t seed, size_t h) {
    return seed ^ (h + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

template <typename T>
size_t hash_dims(size_t seed, const std::vector<T>& dims) {
    for (T d : dims) {
        seed = hash_combine(seed, std::hash<T>{}(d));
    }
    return seed;
}

// A memory::desc has equality (dnnl_memory_desc_equal) but no hash, so this approximates one from
// the fields that actually vary between descriptors seen in practice (dims, dtype, strides, and
// the blocked-format inner block/index lists). DescEqual below falls back to the real `==`, so a
// hash collision (or a field this misses) only costs an extra bucket compare, never correctness.
struct DescHash {
    size_t operator()(const dnnl::memory::desc& desc) const {
        size_t seed = hash_combine(0, std::hash<int>{}(static_cast<int>(desc.get_data_type())));
        seed = hash_dims(seed, desc.get_dims());
        seed = hash_dims(seed, desc.get_strides());
        seed = hash_dims(seed, desc.get_inner_blks());
        seed = hash_dims(seed, desc.get_inner_idxs());
        return seed;
    }
};

struct DescEqual {
    bool operator()(const dnnl::memory::desc& a, const dnnl::memory::desc& b) const { return a == b; }
};

// Hashed on (dims, dtype, strides, inner blocks/indices): lookup and insert cost don't grow with
// the number of distinct descriptors seen, which the old linear scan's cost did. Node-based map,
// so `intern()`'s returned reference (and the raw `Layout*` callers derive from it) stays valid
// for the life of the process regardless of later inserts/rehashes.
std::unordered_map<dnnl::memory::desc, std::shared_ptr<const Layout>, DescHash, DescEqual>& layout_registry() {
    static std::unordered_map<dnnl::memory::desc, std::shared_ptr<const Layout>, DescHash, DescEqual> registry;
    return registry;
}

const std::shared_ptr<const Layout>& intern(const dnnl::memory::desc& desc) {
    auto& registry = layout_registry();
    auto it = registry.find(desc);
    if (it == registry.end()) {
        it = registry.emplace(desc, std::make_shared<const Layout>(Layout{desc})).first;
    }
    return it->second;
}

struct LayoutPairHash {
    size_t operator()(const std::pair<const Layout*, const Layout*>& key) const {
        const size_t h = std::hash<const Layout*>{}(key.first);
        return h ^ (std::hash<const Layout*>{}(key.second) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    }
};

}  // namespace

dnnl::engine& cpu_engine() {
    static dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    return engine;
}

dnnl::stream& cpu_stream() {
    static dnnl::stream stream(cpu_engine());
    return stream;
}

dnnl::memory::dims to_dims(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return {1};
    }
    dnnl::memory::dims dims(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        dims[i] = static_cast<dnnl::memory::dim>(shape[i]);
    }
    return dims;
}

dnnl::memory::desc plain_desc(const dnnl::memory::dims& dims) {
    using tag = dnnl::memory::format_tag;
    static constexpr tag kTags[] = {tag::a, tag::ab, tag::abc, tag::abcd, tag::abcde, tag::abcdef};
    if (dims.empty() || dims.size() > std::size(kTags)) {
        throw std::runtime_error("plain_desc: only 1D to 6D tensors can be passed to oneDNN");
    }
    return dnnl::memory::desc(dims, dnnl::memory::data_type::f32, kTags[dims.size() - 1]);
}

const Layout* layout_id(const dnnl::memory::desc& desc) {
    return intern(desc).get();
}

std::shared_ptr<const Layout> storage_layout(const dnnl::memory::desc& desc) {
    if (desc == plain_desc(desc.get_dims())) {
        return nullptr;
    }
    return intern(desc);
}

void reorder(const dnnl::memory& src, const dnnl::memory& dst) {
    static std::unordered_map<std::pair<const Layout*, const Layout*>, dnnl::reorder, LayoutPairHash> cache;
    const std::pair<const Layout*, const Layout*> key{layout_id(src.get_desc()), layout_id(dst.get_desc())};
    auto it = cache.find(key);
    if (it == cache.end()) {
        it = cache.emplace(key, dnnl::reorder(src, dst)).first;
    }
    dnnl::memory src_handle = src;
    dnnl::memory dst_handle = dst;
    it->second.execute(cpu_stream(), src_handle, dst_handle);
    cpu_stream().wait();
}

void materialize_plain(Storage& storage) {
    if (!storage.layout) {
        return;
    }
    const dnnl::memory::desc& blocked = storage.layout->desc;
    const dnnl::memory::desc plain = plain_desc(blocked.get_dims());
    FloatBuffer data = buffer_for(plain);
    reorder(dnnl::memory(blocked, cpu_engine(), storage.data.data()), dnnl::memory(plain, cpu_engine(), data.data()));
    storage.data = std::move(data);
    storage.layout.reset();
}

dnnl::memory::desc desc_of(Storage& storage, const dnnl::memory::dims& dims) {
    if (storage.layout) {
        if (storage.layout->desc.get_dims() == dims) {
            return storage.layout->desc;
        }
        materialize_plain(storage);
    }
    return plain_desc(dims);
}

dnnl::memory::desc desc_of(const Tensor& t) {
    return desc_of(*t.impl()->storage, to_dims(t.shape()));
}

dnnl::memory memory_of(Storage& storage, const dnnl::memory::dims& dims) {
    const dnnl::memory::desc desc = desc_of(storage, dims);
    return dnnl::memory(desc, cpu_engine(), storage.data.data());
}

dnnl::memory memory_of(const Tensor& t) {
    return memory_of(*t.impl()->storage, to_dims(t.shape()));
}

dnnl::memory convert(const dnnl::memory& m, const dnnl::memory::desc& want) {
    if (m.get_desc() == want) {
        return m;
    }
    dnnl::memory out(want, cpu_engine());
    reorder(m, out);
    return out;
}

dnnl::memory memory_as(const Tensor& t, const dnnl::memory::desc& want) {
    return convert(memory_of(t), want);
}

FloatBuffer buffer_for(const dnnl::memory::desc& desc) {
    return FloatBuffer(desc.get_size() / sizeof(float));
}

Tensor make_tensor(FloatBuffer data, std::vector<size_t> shape, std::shared_ptr<const Layout> layout) {
    auto impl = std::make_shared<TensorImpl>();
    impl->storage = std::make_shared<Storage>(Storage{std::move(data), 0, std::move(layout)});
    impl->shape = std::move(shape);
    return Tensor::from_impl(std::move(impl));
}

}  // namespace advanceml::detail
