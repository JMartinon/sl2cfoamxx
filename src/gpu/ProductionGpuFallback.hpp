// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "core/Failure.hpp"

#include <array>
#include <exception>
#include <new>
#include <stdexcept>
#include <string_view>

namespace sl2cfoam::gpu::production::detail {

[[nodiscard]] constexpr char ascii_lower(char value) noexcept {
  return value >= 'A' && value <= 'Z'
             ? static_cast<char>(value - 'A' + 'a')
             : value;
}

[[nodiscard]] constexpr bool contains_ascii_case_insensitive(
    std::string_view text, std::string_view needle) noexcept {
  if (needle.empty())
    return true;
  if (needle.size() > text.size())
    return false;
  for (std::size_t begin = 0; begin + needle.size() <= text.size(); ++begin) {
    bool equal = true;
    for (std::size_t i = 0; i < needle.size(); ++i) {
      if (ascii_lower(text[begin + i]) != ascii_lower(needle[i])) {
        equal = false;
        break;
      }
    }
    if (equal)
      return true;
  }
  return false;
}

// Fallbacks may change only bounded-memory configuration. Keep this predicate
// deliberately conservative: an unrecognized CUDA, recurrence, validation, or
// numerical failure must be reported immediately instead of being retried.
[[nodiscard]] constexpr bool
is_memory_fallback_failure(std::string_view message) noexcept {
  constexpr std::array<std::string_view, 15> memory_markers{{
      "out of memory",
      "cudaerrormemoryallocation",
      "memory allocation",
      "allocation failed",
      "failed to allocate",
      "insufficient memory",
      "not enough memory",
      "vram",
      "memory budget",
      "device budget",
      "workspace",
      "row capacity",
      "cannot fit",
      "does not fit",
      "std::bad_alloc",
  }};
  for (const std::string_view marker : memory_markers) {
    if (contains_ascii_case_insensitive(message, marker))
      return true;
  }
  return false;
}

[[nodiscard]] inline bool
is_memory_fallback_failure(const std::exception &error) noexcept {
  if (dynamic_cast<const b4utils::ResourceLimitFailure *>(&error) != nullptr)
    return true;
  if (dynamic_cast<const std::bad_alloc *>(&error) != nullptr)
    return true;
  if (dynamic_cast<const std::logic_error *>(&error) != nullptr ||
      dynamic_cast<const std::overflow_error *>(&error) != nullptr ||
      dynamic_cast<const std::range_error *>(&error) != nullptr)
    return false;
  return is_memory_fallback_failure(std::string_view(error.what()));
}

} // namespace sl2cfoam::gpu::production::detail
