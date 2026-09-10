// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "sl2cfoamxx/configuration.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace sl2cfoamxx::detail {

// Human-facing product name. Keep technical identifiers such as namespaces,
// package names, environment variables, and cache paths as sl2cfoamxx.
inline constexpr std::string_view product_display_name = "sl2cfoam++";

// Interactive transport selection is deliberately private and automatic. A
// caller-supplied std::ostream has no portable file descriptor, so it always
// uses plain records even when it happens to wrap a terminal. PrintingStyle
// selects the glyph set after that transport check.
struct OutputPresentation {
    bool interactive = false;
    bool utf8 = false;
    bool terminal_control = false;
    std::size_t columns = 0;
};

[[nodiscard]] OutputPresentation detect_output_presentation(
    bool has_custom_stream,
    PrintingStyle printing_style,
    int file_descriptor = 2) noexcept;

[[nodiscard]] std::string format_human_duration(
    double seconds,
    bool utf8);

[[nodiscard]] std::string format_human_bytes(std::uint64_t bytes);

struct ProgressRenderState {
    std::string_view label{};
    std::string_view unit_label{};
    std::uint64_t completed = 0;
    std::uint64_t total = 0;
    double elapsed_seconds = 0.0;
    double eta_seconds = 0.0;
    bool eta_available = false;
    bool completed_successfully = false;
    bool stopped = false;
};

// Neither renderer includes cursor movement or a trailing newline. The caller
// owns those transport details while holding the shared informational lock.
[[nodiscard]] std::string render_interactive_progress(
    const ProgressRenderState& state,
    const OutputPresentation& presentation,
    std::size_t maximum_bar_width = 22);

[[nodiscard]] std::string render_plain_progress(
    const ProgressRenderState& state);

} // namespace sl2cfoamxx::detail
