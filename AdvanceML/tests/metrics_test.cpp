#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <vector>

#include "advanceml/metrics/accuracy.hpp"
#include "advanceml/tensor.hpp"

using namespace advanceml;
using namespace advanceml::metrics;

TEST_CASE("Accuracy::update/compute compares row-wise argmax of pred against a one-hot target", "[metrics]") {
    Accuracy acc;
    Tensor pred({0.1f, 0.9f, 0.8f, 0.2f, 0.3f, 0.7f}, {3, 2});
    Tensor target({0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f}, {3, 2});

    acc.update(pred, target);

    REQUIRE(acc.compute() == Catch::Approx(2.0f / 3.0f));
}

TEST_CASE("Accuracy::update accumulates across multiple batches", "[metrics]") {
    Accuracy acc;
    Tensor pred1({0.9f, 0.1f}, {1, 2});
    Tensor target1({1.0f, 0.0f}, {1, 2});
    Tensor pred2({0.1f, 0.9f}, {1, 2});
    Tensor target2({1.0f, 0.0f}, {1, 2});

    acc.update(pred1, target1);
    acc.update(pred2, target2);

    REQUIRE(acc.compute() == Catch::Approx(0.5f));
}

TEST_CASE("Accuracy::update accepts raw integer labels in place of a one-hot target", "[metrics]") {
    Accuracy acc;
    Tensor pred({0.1f, 0.9f, 0.8f, 0.2f}, {2, 2});

    acc.update(pred, std::vector<int>{1, 0});

    REQUIRE(acc.compute() == Catch::Approx(1.0f));
}

TEST_CASE("Accuracy::reset clears accumulated counts", "[metrics]") {
    Accuracy acc;
    Tensor pred({0.9f, 0.1f}, {1, 2});
    Tensor target({1.0f, 0.0f}, {1, 2});
    acc.update(pred, target);

    acc.reset();

    REQUIRE_THROWS_AS(acc.compute(), std::runtime_error);
}
