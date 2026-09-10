// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "sl2cfoamxx/export.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sl2cfoamxx {

// Stable machine-readable reasons for operational sl2cfoam++ failures.  The
// explicit values are part of the public API: bindings may expose them as
// integers, and existing values will not be renumbered in later releases.
//
// Programmer/input errors continue to use the corresponding standard C++
// exceptions (invalid_argument, out_of_range, overflow_error, length_error,
// bad_alloc, and filesystem_error). Error is reserved for failures for which
// std::runtime_error alone would otherwise require parsing what().
enum class ErrorCode : std::uint16_t {
    backend_unavailable = 1,
    backend_failure = 2,
    invalid_tensor_file = 3,
    cache_failure = 4,
    resource_limit = 5,
    numerical_failure = 6,
    internal_failure = 7,
    unsupported_operation = 8
};

[[nodiscard]] constexpr std::string_view to_string(ErrorCode code) noexcept
{
    switch (code) {
    case ErrorCode::backend_unavailable:
        return "backend_unavailable";
    case ErrorCode::backend_failure:
        return "backend_failure";
    case ErrorCode::invalid_tensor_file:
        return "invalid_tensor_file";
    case ErrorCode::cache_failure:
        return "cache_failure";
    case ErrorCode::resource_limit:
        return "resource_limit";
    case ErrorCode::numerical_failure:
        return "numerical_failure";
    case ErrorCode::internal_failure:
        return "internal_failure";
    case ErrorCode::unsupported_operation:
        return "unsupported_operation";
    }
    return "unknown";
}

// A source-compatible refinement of std::runtime_error. Existing callers that
// catch std::runtime_error keep working, while bindings can catch Error and
// branch on code() without depending on diagnostic text.
class SL2CFOAMXX_API Error final : public std::runtime_error {
public:
    Error(ErrorCode code, const std::string& message)
        : std::runtime_error(message), code_(code)
    {}

    Error(ErrorCode code, const char* message)
        : std::runtime_error(message), code_(code)
    {}

    [[nodiscard]] ErrorCode code() const noexcept { return code_; }

private:
    ErrorCode code_;
};

} // namespace sl2cfoamxx
