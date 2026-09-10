// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/OutputPresentation.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>

#include <sys/ioctl.h>
#include <unistd.h>

namespace sl2cfoamxx::detail {
namespace {

[[nodiscard]] std::string lowercase(std::string value)
{
    std::transform(
        value.begin(), value.end(), value.begin(), [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    return value;
}

[[nodiscard]] const char* active_locale_name() noexcept
{
    for (const char* name : {"LC_ALL", "LC_CTYPE", "LANG"}) {
        if (const char* value = std::getenv(name);
            value != nullptr && *value != '\0') {
            return value;
        }
    }
    return nullptr;
}

[[nodiscard]] bool utf8_is_plausible() noexcept
{
    try {
        const char* locale_name = active_locale_name();
        if (locale_name == nullptr) {
            // Contemporary macOS/Linux terminals are UTF-8 by default. Only
            // an explicit C/POSIX locale disables Unicode presentation.
            return true;
        }
        const std::string locale = lowercase(locale_name);
        if (locale == "c" || locale == "posix") return false;
        return locale.find("utf-8") != std::string::npos ||
               locale.find("utf8") != std::string::npos;
    } catch (...) {
        return false;
    }
}

[[nodiscard]] std::string trimmed_fixed(double value, int precision)
{
    std::ostringstream output;
    output << std::fixed << std::setprecision(precision) << value;
    std::string text = output.str();
    if (const std::size_t point = text.find('.'); point != std::string::npos) {
        while (!text.empty() && text.back() == '0') text.pop_back();
        if (!text.empty() && text.back() == '.') text.pop_back();
    }
    return text;
}

[[nodiscard]] std::string quoted_plain(std::string_view text)
{
    std::string result;
    result.reserve(text.size() + 2);
    result.push_back('"');
    for (const char character : text) {
        if (character == '\\' || character == '"') result.push_back('\\');
        if (character == '\n' || character == '\r') {
            result.push_back(' ');
        } else {
            result.push_back(character);
        }
    }
    result.push_back('"');
    return result;
}

[[nodiscard]] std::string progress_phase(std::string_view label)
{
    if (label.find("sj-stream") != std::string_view::npos) {
        return "6j_stream";
    }
    return "contract";
}

[[nodiscard]] std::string make_bar(
    double fraction,
    std::size_t width,
    bool utf8)
{
    const double clamped = std::clamp(fraction, 0.0, 1.0);
    const std::size_t filled = std::min<std::size_t>(
        width,
        static_cast<std::size_t>(std::floor(
            clamped * static_cast<double>(width) + 1.0e-12)));
    std::string result;
    if (utf8) {
        result.reserve(width * 3);
        for (std::size_t index = 0; index < width; ++index) {
            result += index < filled ? "█" : "░";
        }
    } else {
        result.reserve(width + 2);
        result.push_back('[');
        for (std::size_t index = 0; index < width; ++index) {
            result.push_back(index < filled ? '#' : '-');
        }
        result.push_back(']');
    }
    return result;
}

[[nodiscard]] std::size_t display_width(std::string_view text) noexcept
{
    std::size_t width = 0;
    for (const unsigned char character : text) {
        if ((character & 0xc0u) != 0x80u) ++width;
    }
    return width;
}

void remove_last_codepoint(std::string& text) noexcept
{
    if (text.empty()) return;
    std::size_t begin = text.size() - 1;
    while (begin > 0 &&
           (static_cast<unsigned char>(text[begin]) & 0xc0u) == 0x80u) {
        --begin;
    }
    text.erase(begin);
}

[[nodiscard]] std::string progress_percent(double percent)
{
    std::ostringstream output;
    output << std::fixed << std::setprecision(1) << percent << '%';
    return output.str();
}

} // namespace

OutputPresentation detect_output_presentation(
    bool has_custom_stream,
    PrintingStyle printing_style,
    int file_descriptor) noexcept
{
    OutputPresentation presentation;
    if (has_custom_stream || file_descriptor < 0 ||
        ::isatty(file_descriptor) != 1) {
        return presentation;
    }

    if (const char* term = std::getenv("TERM"); term != nullptr) {
        try {
            if (lowercase(term) == "dumb") return presentation;
        } catch (...) {
            return presentation;
        }
    }

    winsize size{};
    if (::ioctl(file_descriptor, TIOCGWINSZ, &size) == 0 && size.ws_col != 0) {
        presentation.columns = size.ws_col;
        if (presentation.columns < 48) return {};
    }
    presentation.interactive = true;
    presentation.utf8 = printing_style == PrintingStyle::UTF8 &&
                        utf8_is_plausible();
    presentation.terminal_control = true;
    return presentation;
}

std::string format_human_duration(double seconds, bool utf8)
{
    if (!std::isfinite(seconds) || seconds < 0.0) {
        return utf8 ? "—" : "--";
    }
    if (seconds < 1.0e-6) {
        return trimmed_fixed(seconds * 1.0e9, 0) + " ns";
    }
    if (seconds < 1.0e-3) {
        return trimmed_fixed(seconds * 1.0e6, 0) +
               (utf8 ? " µs" : " us");
    }
    if (seconds < 1.0) {
        return trimmed_fixed(seconds * 1.0e3, seconds < 0.01 ? 2 : 1) +
               " ms";
    }
    if (seconds < 60.0) {
        return trimmed_fixed(seconds, 3) + " s";
    }

    const std::uint64_t rounded = static_cast<std::uint64_t>(seconds + 0.5);
    const std::uint64_t hours = rounded / 3600;
    const std::uint64_t minutes = (rounded % 3600) / 60;
    const std::uint64_t remaining_seconds = rounded % 60;
    std::ostringstream output;
    if (hours != 0) output << hours << " h ";
    if (minutes != 0 || hours != 0) output << minutes << " min ";
    output << remaining_seconds << " s";
    return output.str();
}

std::string format_human_bytes(std::uint64_t bytes)
{
    static constexpr std::array<const char*, 5> units{
        "B", "KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    std::size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < units.size()) {
        value /= 1024.0;
        ++unit;
    }
    const int precision = unit == 0 ? 0 : (value < 10.0 ? 2 : 1);
    return trimmed_fixed(value, precision) + ' ' + units[unit];
}

std::string render_interactive_progress(
    const ProgressRenderState& state,
    const OutputPresentation& presentation,
    std::size_t maximum_bar_width)
{
    double fraction = state.total == 0
        ? 1.0
        : std::clamp(
              static_cast<double>(state.completed) /
                  static_cast<double>(state.total),
              0.0,
              1.0);
    if (!state.completed_successfully) {
        fraction = std::min(fraction, 0.999);
    }
    const double percent = 100.0 * fraction;

    const std::string separator = presentation.utf8 ? " · " : " ; ";
    const std::string percent_text = progress_percent(percent);
    const std::string elapsed_text =
        separator + "elapsed " +
        format_human_duration(state.elapsed_seconds, presentation.utf8);
    const std::string eta_text =
        separator + "ETA " +
        (state.eta_available
             ? format_human_duration(state.eta_seconds, presentation.utf8)
             : (presentation.utf8 ? "—" : "--"));
    std::size_t indent_width =
        presentation.columns != 0 && presentation.columns < 60 ? 2 : 13;

    if (state.stopped) {
        std::string output = std::string(indent_width, ' ') +
            (presentation.utf8 ? "× stopped at " : "x stopped at ") +
            percent_text;
        if (presentation.columns == 0 ||
            display_width(output) + display_width(elapsed_text) <=
                presentation.columns) {
            output += elapsed_text;
        }
        return output;
    }

    bool show_elapsed = presentation.columns == 0 ||
                        presentation.columns >= 45;
    // ETA is the primary reason for a live progress display. Keep its place
    // visible from startup (as an em dash until the first reliable sample),
    // and sacrifice elapsed time first on constrained terminals.
    bool show_eta = !state.completed_successfully;
    const std::size_t bar_decoration_width = presentation.utf8 ? 0 : 2;
    constexpr std::size_t minimum_bar_width = 10;
    const auto required_width = [&](bool elapsed, bool eta) {
        return indent_width + minimum_bar_width + bar_decoration_width + 1 +
               display_width(percent_text) +
               (elapsed ? display_width(elapsed_text) : 0) +
               (eta ? display_width(eta_text) : 0);
    };
    if (presentation.columns != 0 &&
        required_width(show_elapsed, show_eta) > presentation.columns) {
        show_elapsed = false;
    }
    if (presentation.columns != 0 &&
        required_width(show_elapsed, show_eta) > presentation.columns) {
        indent_width = 0;
    }
    if (presentation.columns != 0 &&
        required_width(show_elapsed, show_eta) > presentation.columns) {
        show_eta = false;
    }

    const std::size_t fixed_width = indent_width + bar_decoration_width + 1 +
        display_width(percent_text) +
        (show_elapsed ? display_width(elapsed_text) : 0) +
        (show_eta ? display_width(eta_text) : 0);
    const std::size_t available_bar_width = presentation.columns == 0
        ? maximum_bar_width
        : presentation.columns > fixed_width
            ? presentation.columns - fixed_width
            : 1;
    const std::size_t width = std::max<std::size_t>(
        1, std::min(maximum_bar_width, available_bar_width));

    std::string output = std::string(indent_width, ' ') +
        make_bar(fraction, width, presentation.utf8) + ' ' + percent_text;
    if (show_elapsed) output += elapsed_text;
    if (!state.completed_successfully && show_eta) output += eta_text;
    if (presentation.columns != 0 &&
        display_width(output) > presentation.columns) {
        // Interactive mode is only selected for usable terminals, but keep
        // forced/test presentations safe as well.
        while (!output.empty() && display_width(output) > presentation.columns) {
            remove_last_codepoint(output);
        }
    }
    return output;
}

std::string render_plain_progress(const ProgressRenderState& state)
{
    double fraction = state.total == 0
        ? 1.0
        : std::clamp(
              static_cast<double>(state.completed) /
                  static_cast<double>(state.total),
              0.0,
              1.0);
    if (!state.completed_successfully) {
        fraction = std::min(fraction, 0.999);
    }
    const char* status = state.stopped
        ? "stopped"
        : (state.completed_successfully ? "complete" : "running");

    std::ostringstream output;
    output << '[' << product_display_name << ": progress] label="
           << quoted_plain(state.label)
           << " phase=" << progress_phase(state.label)
           << " status=" << status
           << " completed=" << state.completed
           << " total=" << state.total
           << " unit=" << quoted_plain(state.unit_label)
           << " percent=" << std::fixed << std::setprecision(1)
           << 100.0 * fraction
           << " elapsed_seconds=" << std::setprecision(6)
           << state.elapsed_seconds;
    if (!state.completed_successfully && !state.stopped &&
        state.eta_available) {
        output << " eta_seconds=" << state.eta_seconds;
    }
    return output.str();
}

} // namespace sl2cfoamxx::detail
