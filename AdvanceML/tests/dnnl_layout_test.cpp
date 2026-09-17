#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstddef>
#include <vector>

#include "detail/dnnl_layout.hpp"

using namespace advanceml;

// TODO-0047 P1: the layout registry used to be a linear scan over every descriptor ever seen
// (AdvanceML/src/detail/dnnl_layout.cpp `intern()`), so lookup cost grew with the number of
// distinct shapes interned. It's now a hash map keyed on (dims, dtype, strides, inner blocks/
// indices), with `==` as the tiebreaker on collisions. This checks that a lookup against a probe
// set doesn't get measurably slower once 1,000 other distinct shapes have been interned around it.

namespace {

dnnl::memory::desc desc_for(size_t last_dim) {
    return dnnl::memory::desc({1, 1, 1, static_cast<dnnl::memory::dim>(last_dim + 1)}, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::abcd);
}

double time_lookups_us(const std::vector<dnnl::memory::desc>& probes, int repeats) {
    const auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r) {
        for (const dnnl::memory::desc& d : probes) {
            detail::layout_id(d);
        }
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / (repeats * probes.size());
}

}  // namespace

TEST_CASE("layout registry lookup cost does not grow with the number of distinct shapes interned",
          "[detail][layout][perf]") {
    // A small probe set, interned first so it occupies the map regardless of insertion order.
    std::vector<dnnl::memory::desc> probes;
    for (size_t i = 0; i < 20; ++i) {
        probes.push_back(desc_for(i));
        detail::layout_id(probes.back());
    }

    const double before_us = time_lookups_us(probes, /*repeats=*/20000);

    // Intern 1,000 further distinct descriptors so the registry holds ~1,020 entries.
    for (size_t i = 20; i < 1020; ++i) {
        dnnl::memory::desc d = desc_for(i);
        detail::layout_id(d);
    }

    const double after_us = time_lookups_us(probes, /*repeats=*/20000);

    // A linear scan over ~1,020 entries instead of ~20 would be roughly two orders of magnitude
    // slower; a hash map stays flat. 8x is generous headroom for scheduling/cache noise while
    // still clearly rejecting linear growth.
    REQUIRE(after_us < before_us * 8.0 + 1.0);
}
