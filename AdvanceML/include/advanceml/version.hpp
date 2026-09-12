#pragma once

#include <string_view>

namespace advanceml {

/** Returns the AdvanceML library version string (semantic-versioning, e.g. "0.1.0"). */
[[nodiscard]] std::string_view version() noexcept;

}  // namespace advanceml
