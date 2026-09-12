#include <catch2/catch_test_macros.hpp>

#include "advanceml/version.hpp"

TEST_CASE("advanceml scaffold builds and links", "[smoke]") {
    REQUIRE(advanceml::version() == "0.1.0");
}
