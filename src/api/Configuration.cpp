// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "sl2cfoamxx/configuration.hpp"
#include "sl2cfoamxx/error.hpp"

#include "api/ConfigurationInternal.hpp"
#include "core/InformationalOutput.hpp"
#include "core/LowLevelTuning.hpp"
#include "core/OpenMp.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#if !defined(SL2CFOAMXX_COMPILED_WITH_CUDA)
#define SL2CFOAMXX_COMPILED_WITH_CUDA 0
#endif

namespace sl2cfoamxx {
namespace {

namespace fs = std::filesystem;

std::mutex configuration_mutex;
std::shared_ptr<const LibraryConfig> active_configuration;

[[nodiscard]] fs::path environment_path(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? fs::path(value) : fs::path{};
}

[[nodiscard]] fs::path default_cache_root()
{
    if (fs::path path = environment_path("SL2CFOAMXX_CACHE_DIR"); !path.empty()) {
        return path;
    }
    return fs::current_path() / "data_sl2cfoamxx";
}

[[nodiscard]] fs::path normalize_directory(
    const fs::path& configured,
    const fs::path& fallback)
{
    fs::path path = configured.empty() ? fallback : configured;
    if (path.is_relative()) path = fs::absolute(path);
    return path.lexically_normal();
}

[[nodiscard]] bool path_is_within(
    const fs::path& candidate,
    const fs::path& directory)
{
    auto candidate_component = candidate.begin();
    for (auto directory_component = directory.begin();
         directory_component != directory.end();
         ++directory_component, ++candidate_component) {
        if (candidate_component == candidate.end() ||
            *candidate_component != *directory_component) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] LibraryConfig normalize_and_validate(LibraryConfig config)
{
    const fs::path root = normalize_directory(default_cache_root(), fs::current_path());
    config.tensors.directory = normalize_directory(
        config.tensors.directory, root);

    if (config.cpu_threads >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("cpu_threads does not fit the OpenMP API");
    }
    if (!std::isfinite(config.coherent_amplitude.pruning_ratio) ||
        config.coherent_amplitude.pruning_ratio < 0.0 ||
        config.coherent_amplitude.pruning_ratio > 1.0) {
        throw std::invalid_argument(
            "coherent_amplitude.pruning_ratio must be finite and lie in "
            "[0,1]");
    }
    if (!detail::is_supported_booster_rho_tile(
            config.low_level.booster_rho_tile)) {
        throw std::invalid_argument(
            "low_level.booster_rho_tile must be 0, 4, 8, or 16");
    }
    if (config.low_level.full_tensor_gpu_row_capacity != 0 &&
        (config.low_level.full_tensor_gpu_row_capacity < 1024 ||
         config.low_level.full_tensor_gpu_row_capacity >
             static_cast<std::uint64_t>(std::numeric_limits<int>::max()))) {
        throw std::invalid_argument(
            "low_level.full_tensor_gpu_row_capacity must be zero or fit in "
            "[1024, INT_MAX]");
    }
    if (!detail::is_supported_coherent_gpu_pipeline_slots(
            config.low_level.coherent_gpu_pipeline_slots)) {
        throw std::invalid_argument(
            "low_level.coherent_gpu_pipeline_slots must be 0, 1, 2, or 3");
    }
    if (!detail::is_supported_full_tensor_gpu_producer_waves_per_sm(
            config.low_level.full_tensor_gpu_producer_waves_per_sm)) {
        throw std::invalid_argument(
            "low_level.full_tensor_gpu_producer_waves_per_sm must be zero "
            "or fit in [1,64]");
    }
    if (!detail::is_supported_full_tensor_cpu_p_batch(
            config.low_level.full_tensor_cpu_p_batch)) {
        throw std::invalid_argument(
            "low_level.full_tensor_cpu_p_batch must be 0, 128, 256, 512, "
            "1024, or 2048");
    }
    if (!detail::is_supported_full_tensor_cpu_final_tile(
            config.low_level.full_tensor_cpu_final_tile)) {
        throw std::invalid_argument(
            "unknown low_level.full_tensor_cpu_final_tile");
    }
    switch (config.simplicity_constraint) {
    case SimplicityConstraint::gamma_j:
    case SimplicityConstraint::gamma_j_p1:
        break;
    default:
        throw std::invalid_argument("unknown simplicity constraint");
    }
    switch (config.verbosity) {
    case Verbosity::silent:
    case Verbosity::summary:
    case Verbosity::timing:
    case Verbosity::detailed:
        break;
    default:
        throw std::invalid_argument("unknown library verbosity level");
    }
    switch (config.printing_style) {
    case PrintingStyle::UTF8:
    case PrintingStyle::ASCII:
        break;
    default:
        throw std::invalid_argument("unknown library printing style");
    }

    switch (config.backend) {
    case Backend::CPU:
    case Backend::GPU:
        break;
    default:
        throw std::invalid_argument("unknown library backend");
    }
    switch (config.tensors.storage) {
    case TensorStorage::in_memory_sym:
    case TensorStorage::in_memory_dense:
        if (!config.tensors.output_path.empty()) {
            throw std::invalid_argument(
                "tensors.output_path requires an on-disk storage mode");
        }
        break;
    case TensorStorage::on_disk_sym:
    case TensorStorage::on_disk_dense:
        break;
    default:
        throw std::invalid_argument("unknown full-tensor storage mode");
    }

    std::error_code error;
    fs::create_directories(config.tensors.directory, error);
    if (error) {
        throw fs::filesystem_error(
            "cannot create the tensor cache directory",
            config.tensors.directory,
            error);
    }

    config.tensors.directory =
        fs::weakly_canonical(config.tensors.directory);

    if (!config.tensors.output_path.empty()) {
        fs::path output = config.tensors.output_path;
        if (output.is_relative()) output = fs::absolute(output);
        output = fs::weakly_canonical(output);

        const fs::path scratch = fs::weakly_canonical(
            config.tensors.directory / ".scratch");
        if (path_is_within(output, scratch)) {
            throw std::invalid_argument(
                "tensors.output_path cannot be inside the reserved tensor "
                ".scratch directory");
        }
        config.tensors.output_path = std::move(output);
    }
    return config;
}

void apply_openmp_configuration(const LibraryConfig& config)
{
    struct CallerOpenMpState {
        int automatic_thread_count = std::max(1, omp_get_max_threads());
        int last_library_thread_count = 0;
    };

    // OpenMP internal-control variables belong to the calling task. Preserve
    // that caller's pre-library ceiling, but notice a later omp_set_num_threads
    // made by the application: a value different from our last applied value
    // becomes the new automatic baseline. This avoids both cross-thread state
    // leakage and permanently freezing the first value observed on a thread.
    thread_local CallerOpenMpState state;
    const int current_thread_count = std::max(1, omp_get_max_threads());
    if (state.last_library_thread_count != 0 &&
        current_thread_count != state.last_library_thread_count) {
        state.automatic_thread_count = current_thread_count;
    }
    omp_set_dynamic(0);
    const int selected_thread_count = config.cpu_threads != 0
        ? static_cast<int>(config.cpu_threads)
        : state.automatic_thread_count;
    omp_set_num_threads(selected_thread_count);
    state.last_library_thread_count = selected_thread_count;
}

[[nodiscard]] std::shared_ptr<const LibraryConfig> active_snapshot()
{
    std::lock_guard lock(configuration_mutex);
    if (!active_configuration) {
        active_configuration = std::make_shared<const LibraryConfig>(
            normalize_and_validate(default_config()));
    }
    return active_configuration;
}

} // namespace

LibraryConfig default_config()
{
    const fs::path root = default_cache_root();
    LibraryConfig config;
    config.tensors.directory = root;
#if SL2CFOAMXX_COMPILED_WITH_CUDA
    config.backend = Backend::GPU;
#else
    config.backend = Backend::CPU;
#endif
    config.simplicity_constraint = SimplicityConstraint::gamma_j;
    return config;
}

void configure(const LibraryConfig& config)
{
    auto normalized = std::make_shared<const LibraryConfig>(
        normalize_and_validate(config));
    apply_openmp_configuration(*normalized);
    std::lock_guard lock(configuration_mutex);
    active_configuration = std::move(normalized);
}

LibraryConfig current_config()
{
    return *active_snapshot();
}

namespace detail {

namespace {

struct ParsedField {
    std::string key;
    std::string value;
};

struct ParsedDetail {
    std::string prefix;
    std::vector<ParsedField> fields;
};

[[nodiscard]] const char* plain_operation_name(
    DiagnosticOperation operation) noexcept
{
    switch (operation) {
    case DiagnosticOperation::amplitude:
        return "compute_amplitude";
    case DiagnosticOperation::coherent_amplitude:
        return "compute_coherent_amplitude";
    case DiagnosticOperation::full_tensor:
        return "compute_full_tensor";
    }
    return "operation";
}

[[nodiscard]] const char* pretty_operation_name(
    DiagnosticOperation operation) noexcept
{
    switch (operation) {
    case DiagnosticOperation::amplitude:
        return "amplitude";
    case DiagnosticOperation::coherent_amplitude:
        return "coherent amplitude";
    case DiagnosticOperation::full_tensor:
        return "full tensor";
    }
    return "operation";
}

[[nodiscard]] const char* diagnostic_backend_name(Backend backend) noexcept
{
    return backend == Backend::GPU ? "GPU" : "CPU";
}

[[nodiscard]] std::string_view trim_view(std::string_view text) noexcept
{
    while (!text.empty() &&
           std::isspace(static_cast<unsigned char>(text.front())) != 0) {
        text.remove_prefix(1);
    }
    while (!text.empty() &&
           std::isspace(static_cast<unsigned char>(text.back())) != 0) {
        text.remove_suffix(1);
    }
    return text;
}

[[nodiscard]] bool is_key_character(unsigned char character) noexcept
{
    return std::isalnum(character) != 0 || character == '_' ||
           character == '-' || character == '+';
}

struct KeyOccurrence {
    std::size_t start = 0;
    std::size_t equals = 0;
};

[[nodiscard]] std::optional<KeyOccurrence> find_key(
    std::string_view text,
    std::size_t begin) noexcept
{
    for (std::size_t index = begin; index < text.size(); ++index) {
        if (index != 0 &&
            std::isspace(static_cast<unsigned char>(text[index - 1])) == 0) {
            continue;
        }
        if (!is_key_character(static_cast<unsigned char>(text[index]))) {
            continue;
        }
        std::size_t end = index + 1;
        while (end < text.size() &&
               is_key_character(static_cast<unsigned char>(text[end]))) {
            ++end;
        }
        if (end < text.size() && text[end] == '=') {
            return KeyOccurrence{index, end};
        }
    }
    return std::nullopt;
}

[[nodiscard]] ParsedDetail parse_detail(std::string_view text)
{
    ParsedDetail parsed;
    text = trim_view(text);
    const std::optional<KeyOccurrence> first = find_key(text, 0);
    if (!first) {
        parsed.prefix = std::string(text);
        return parsed;
    }
    parsed.prefix = std::string(trim_view(text.substr(0, first->start)));

    std::optional<KeyOccurrence> current = first;
    while (current) {
        const std::size_t value_begin = current->equals + 1;
        const std::optional<KeyOccurrence> next = find_key(text, value_begin);
        const std::size_t value_end = next ? next->start : text.size();
        parsed.fields.push_back({
            std::string(text.substr(
                current->start, current->equals - current->start)),
            std::string(trim_view(text.substr(
                value_begin, value_end - value_begin)))});
        current = next;
    }
    return parsed;
}

[[nodiscard]] bool ends_with(
    std::string_view text,
    std::string_view suffix) noexcept
{
    return text.size() >= suffix.size() &&
           text.substr(text.size() - suffix.size()) == suffix;
}

[[nodiscard]] std::optional<std::uint64_t> parse_u64(
    std::string_view text) noexcept
{
    text = trim_view(text);
    std::uint64_t value = 0;
    const auto result = std::from_chars(
        text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} ||
        result.ptr != text.data() + text.size()) {
        return std::nullopt;
    }
    return value;
}

[[nodiscard]] std::optional<double> parse_double(std::string_view text) noexcept
{
    try {
        const std::string owned(trim_view(text));
        std::size_t consumed = 0;
        const double value = std::stod(owned, &consumed);
        if (consumed != owned.size()) return std::nullopt;
        return value;
    } catch (...) {
        return std::nullopt;
    }
}

[[nodiscard]] std::string grouped_integer(std::uint64_t value)
{
    const std::string digits = std::to_string(value);
    std::string result;
    result.reserve(digits.size() + digits.size() / 3);
    for (std::size_t index = 0; index < digits.size(); ++index) {
        if (index != 0 && (digits.size() - index) % 3 == 0) {
            result.push_back(',');
        }
        result.push_back(digits[index]);
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

[[nodiscard]] std::string pretty_key(std::string_view key, bool utf8)
{
    if (key == "Dl") return "Dl";
    if (key == "gamma") return utf8 ? "γ" : "gamma";
    if (key == "dimensions") return "shape";
    if (key == "simplicity_constraint") return "simplicity";
    if (key == "cpu_threads") return "CPU threads";
    if (key == "requested_storage") return "requested storage";
    if (key == "actual_storage" || key == "effective_storage") {
        return "storage";
    }
    if (key == "logical_values") return "logical";
    if (key == "stored_values") return "stored";
    if (key == "logical_bytes") return "logical size";
    if (key == "stored_bytes") return "stored size";
    if (key == "memory_budget_bytes") return "budget";
    if (key == "estimated_peak_bytes") return "estimated peak";
    if (key == "resident_fast_peak_bytes") return "resident-fast peak";
    if (key == "resident_boosters_only_peak_bytes") {
        return "boosters-only peak";
    }
    if (key == "peak_device_bytes") return "device peak";
    if (key == "support_metadata_bytes") return "support metadata";
    if (key == "sj_payload_bytes") return "6j payload";
    if (key == "booster_payload_bytes") return "booster payload";

    std::string result(key);
    if (ends_with(result, "_seconds")) {
        result.resize(result.size() - std::string_view("_seconds").size());
    } else if (ends_with(result, "_bytes")) {
        result.resize(result.size() - std::string_view("_bytes").size());
    }
    std::replace(result.begin(), result.end(), '_', ' ');
    return result;
}

[[nodiscard]] std::string pretty_value(
    const ParsedField& field,
    bool utf8)
{
    if (field.key == "simplicity" ||
        field.key == "simplicity_constraint") {
        if (field.value == "gamma_j") {
            return utf8 ? "γj" : "gamma*j";
        }
        if (field.value == "gamma_j_p1") {
            return utf8 ? "γ(j+1)" : "gamma*(j+1)";
        }
    }
    if (field.key == "dimensions" && field.value.size() >= 2 &&
        field.value.front() == '[' && field.value.back() == ']') {
        std::string shape = field.value.substr(1, field.value.size() - 2);
        const std::string multiplication = utf8 ? "×" : "x";
        std::string result;
        for (const char character : shape) {
            if (character == ',') {
                result += multiplication;
            } else {
                result.push_back(character);
            }
        }
        return result;
    }
    if (ends_with(field.key, "_bytes")) {
        if (const auto bytes = parse_u64(field.value)) {
            return format_human_bytes(*bytes);
        }
    }
    if (ends_with(field.key, "_seconds")) {
        if (const auto seconds = parse_double(field.value)) {
            return format_human_duration(*seconds, utf8);
        }
    }
    if (ends_with(field.key, "_values")) {
        if (const auto value = parse_u64(field.value)) {
            return grouped_integer(*value);
        }
    }
    return field.value;
}

[[nodiscard]] std::string render_fields(
    const std::vector<ParsedField>& fields,
    const OutputPresentation& presentation)
{
    const std::string indent(13, ' ');
    const std::string separator = presentation.utf8 ? " · " : " ; ";
    const std::size_t columns = presentation.columns != 0
        ? presentation.columns
        : 100;

    std::vector<std::string> chunks;
    std::vector<std::string> own_lines;
    for (const ParsedField& field : fields) {
        if (field.key == "reason") {
            own_lines.push_back(field.value);
            continue;
        }
        if (field.key == "path" || field.key == "output_path") {
            own_lines.push_back(field.value);
            continue;
        }
        chunks.push_back(
            pretty_key(field.key, presentation.utf8) + '=' +
            pretty_value(field, presentation.utf8));
    }

    std::ostringstream output;
    std::size_t line_size = indent.size();
    if (!chunks.empty()) output << indent;
    for (std::size_t index = 0; index < chunks.size(); ++index) {
        const std::size_t added =
            (index == 0 ? 0 : display_width(separator)) +
            display_width(chunks[index]);
        if (index != 0 && line_size + added > columns) {
            output << '\n' << indent;
            line_size = indent.size();
        } else if (index != 0) {
            output << separator;
            line_size += display_width(separator);
        }
        output << chunks[index];
        line_size += display_width(chunks[index]);
    }
    for (const std::string& line : own_lines) {
        if (output.tellp() > 0) output << '\n';
        output << indent << line;
    }
    return output.str();
}

[[nodiscard]] std::string detail_topic(std::string_view prefix)
{
    if (prefix == "input" || prefix == "inputs") return "input";
    if (prefix == "output") return "output";
    if (prefix == "cache" || prefix == "full-tensor cache") return "cache";
    if (prefix == "planner") return "plan";
    if (prefix == "execution") return "execution";
    if (prefix == "coherent coefficient timing") return "coefficients";
    if (prefix == "GPU host_plan") return "host plan";
    if (prefix == "GPU profile") return "GPU profile";
    return prefix.empty() ? "detail" : std::string(prefix);
}

[[nodiscard]] std::string interactive_header(
    DiagnosticOperation operation,
    Backend backend,
    std::string_view state,
    bool utf8)
{
    const std::string separator = utf8 ? " │ " : " | ";
    return std::string(product_display_name) + separator +
           pretty_operation_name(operation) + separator +
           diagnostic_backend_name(backend) + separator + std::string(state);
}

[[nodiscard]] OutputPresentation forced_presentation(
    DiagnosticPresentation selection) noexcept
{
    OutputPresentation presentation;
    switch (selection) {
    case DiagnosticPresentation::terminal_utf8:
        presentation.interactive = true;
        presentation.utf8 = true;
        presentation.terminal_control = true;
        presentation.columns = 120;
        break;
    case DiagnosticPresentation::terminal_ascii:
        presentation.interactive = true;
        presentation.utf8 = false;
        presentation.terminal_control = true;
        presentation.columns = 120;
        break;
    case DiagnosticPresentation::automatic:
    case DiagnosticPresentation::plain:
        break;
    }
    return presentation;
}

} // namespace

Diagnostics::Diagnostics(
    const LibraryConfig& config,
    DiagnosticOperation operation,
    Backend backend,
    DiagnosticPresentation presentation) noexcept
    : verbosity_(config.verbosity),
      stream_(config.log_stream),
      operation_(operation),
      backend_(backend)
{
    if (presentation == DiagnosticPresentation::automatic) {
        if (verbosity_ != Verbosity::silent ||
            (operation_ == DiagnosticOperation::coherent_amplitude &&
             config.coherent_amplitude.check_closure)) {
            presentation_ = detect_output_presentation(
                stream_ != nullptr, config.printing_style);
        }
    } else if (presentation != DiagnosticPresentation::plain) {
        presentation_ = forced_presentation(presentation);
    }
}

bool Diagnostics::enabled(Verbosity level) const noexcept
{
    return static_cast<std::uint8_t>(verbosity_) >=
           static_cast<std::uint8_t>(level);
}

void Diagnostics::emit_plain(const std::string& message) const noexcept
{
    try {
        std::lock_guard lock(informational_output_mutex());
        std::ostream& output = stream_ != nullptr ? *stream_ : std::clog;
        output << '[' << product_display_name << ": "
               << plain_operation_name(operation_) << "] " << message;
        if (message.empty() || message.back() != '\n') output << '\n';
        output.flush();
    } catch (...) {
        // Diagnostics must never turn a successful numerical computation into
        // a failure, including when a caller-supplied stream enables exceptions.
    }
}

void Diagnostics::emit_interactive(const std::string& message) const noexcept
{
    try {
        std::lock_guard lock(informational_output_mutex());
        std::ostream& output = stream_ != nullptr ? *stream_ : std::clog;
        if (presentation_.terminal_control &&
            interactive_progress_is_active()) {
            const std::size_t columns = presentation_.columns != 0
                ? presentation_.columns
                : 120;
            output << '\r' << std::string(columns, ' ') << '\r';
        }
        output << message;
        if (message.empty() || message.back() != '\n') output << '\n';
        output.flush();
    } catch (...) {
        // Interactive diagnostics remain best-effort.
    }
}

void Diagnostics::emit_warning(
    std::string_view subject,
    const std::string& payload) const noexcept
{
    try {
        if (!presentation_.interactive) {
            std::ostringstream message;
            message << "warning backend=" << diagnostic_backend_name(backend_)
                    << " subject=" << subject;
            if (!payload.empty()) message << payload;
            emit_plain(message.str());
            return;
        }

        std::string state;
        if (presentation_.utf8) state = "\u26a0 ";
        else state = "! ";
        state += subject;
        state += " warning";

        std::ostringstream message;
        message << interactive_header(
            operation_, backend_, state, presentation_.utf8);
        if (!payload.empty()) {
            const ParsedDetail parsed = parse_detail(payload);
            const std::string fields = render_fields(
                parsed.fields, presentation_);
            if (!fields.empty()) message << '\n' << fields;
            if (parsed.fields.empty() && !parsed.prefix.empty()) {
                message << '\n' << std::string(13, ' ') << parsed.prefix;
            }
        }
        emit_interactive(message.str());
    } catch (...) {
        // Warnings are best-effort and never alter the computation.
    }
}

void Diagnostics::emit_event(
    Event event,
    double elapsed_seconds,
    const std::string& payload) const noexcept
{
    try {
        if (!presentation_.interactive) {
            std::ostringstream message;
            switch (event) {
            case Event::started:
                message << "started backend="
                        << diagnostic_backend_name(backend_) << payload;
                break;
            case Event::completed:
                message << "completed backend="
                        << diagnostic_backend_name(backend_) << payload;
                if (operation_ == DiagnosticOperation::full_tensor) {
                    message << " total_seconds=" << elapsed_seconds;
                } else {
                    message << " total=" << std::fixed
                            << std::setprecision(6) << elapsed_seconds << " s";
                }
                break;
            case Event::failed:
                message << "failed backend="
                        << diagnostic_backend_name(backend_);
                if (operation_ == DiagnosticOperation::full_tensor) {
                    message << " elapsed_seconds=" << elapsed_seconds
                            << " reason=" << payload;
                } else {
                    message << " reason=" << payload << " total="
                            << std::fixed << std::setprecision(6)
                            << elapsed_seconds << " s";
                }
                break;
            }
            emit_plain(message.str());
            return;
        }

        std::ostringstream message;
        switch (event) {
        case Event::started:
            message << interactive_header(
                operation_, backend_, "start", presentation_.utf8);
            break;
        case Event::completed:
            message << interactive_header(
                operation_, backend_,
                std::string(presentation_.utf8 ? "✓ done in " : "done in ") +
                    format_human_duration(
                        elapsed_seconds, presentation_.utf8),
                presentation_.utf8);
            break;
        case Event::failed:
            message << interactive_header(
                operation_, backend_,
                std::string(presentation_.utf8 ? "× failed after "
                                               : "x failed after ") +
                    format_human_duration(
                        elapsed_seconds, presentation_.utf8),
                presentation_.utf8);
            break;
        }

        if (event == Event::failed) {
            message << '\n' << std::string(13, ' ') << payload;
        } else if (!payload.empty()) {
            const ParsedDetail parsed = parse_detail(payload);
            const std::string fields = render_fields(
                parsed.fields, presentation_);
            if (!fields.empty()) message << '\n' << fields;
            if (parsed.fields.empty() && !parsed.prefix.empty()) {
                message << '\n' << std::string(13, ' ') << parsed.prefix;
            }
        }
        emit_interactive(message.str());
    } catch (...) {
        // Rendering must never alter the computation.
    }
}

void Diagnostics::failed(
    double elapsed_seconds,
    std::string_view reason) const noexcept
{
    if (!enabled(Verbosity::summary)) return;
    try {
        emit_event(
            Event::failed, elapsed_seconds, std::string(reason));
    } catch (...) {
        // Reporting is best-effort.
    }
}

void Diagnostics::timing(
    std::initializer_list<DiagnosticTiming> phases,
    std::string_view cache_status) const noexcept
{
    if (!enabled(Verbosity::timing)) return;
    try {
        if (!presentation_.interactive) {
            std::ostringstream message;
            if (operation_ == DiagnosticOperation::full_tensor) {
                message << "timing backend="
                        << diagnostic_backend_name(backend_);
                if (!cache_status.empty()) {
                    message << " cache=" << cache_status;
                }
                for (const DiagnosticTiming& phase : phases) {
                    message << ' ' << phase.plain_key << '=' << phase.seconds;
                }
            } else {
                message << "timing";
                for (const DiagnosticTiming& phase : phases) {
                    message << "\n  " << phase.plain_key << " = "
                            << std::fixed << std::setprecision(6)
                            << phase.seconds << " s";
                }
            }
            emit_plain(message.str());
            return;
        }

        std::vector<const DiagnosticTiming*> visible_phases;
        visible_phases.reserve(phases.size());

        // Timing mode is meant to answer where the wall time went at a
        // glance. When a full-tensor contraction already accounts for nearly
        // all of it, retain the phases that describe its execution plan and
        // numerical preparation even when each is below the materiality
        // threshold. Zero-valued full-tensor phases are non-applicable
        // placeholders and add no useful information to the interactive tree.
        // Detailed mode deliberately keeps the complete phase inventory, and
        // plain records retain every stable key for log processors.
        constexpr double dominant_contraction_fraction = 0.90;
        constexpr double material_phase_fraction = 0.01;
        const bool concise_full_tensor_timing =
            operation_ == DiagnosticOperation::full_tensor &&
            verbosity_ == Verbosity::timing;
        const DiagnosticTiming* contraction = nullptr;
        const DiagnosticTiming* total = nullptr;
        if (concise_full_tensor_timing) {
            for (const DiagnosticTiming& phase : phases) {
                if (phase.plain_key == "contraction_seconds") {
                    contraction = &phase;
                } else if (phase.plain_key == "backend_total_seconds") {
                    total = &phase;
                }
            }
        }
        const bool compact_dominant_timing =
            contraction != nullptr && total != nullptr &&
            std::isfinite(contraction->seconds) &&
            std::isfinite(total->seconds) && total->seconds > 0.0 &&
            contraction->seconds >= 0.0 &&
            contraction->seconds / total->seconds >=
                dominant_contraction_fraction;
        for (const DiagnosticTiming& phase : phases) {
            if (concise_full_tensor_timing && phase.seconds == 0.0 &&
                &phase != total) {
                continue;
            }
            const bool retained_full_tensor_phase =
                concise_full_tensor_timing &&
                (phase.plain_key == "planning_seconds" ||
                 phase.plain_key == "sj_fill_seconds" ||
                 phase.plain_key == "booster_fill_seconds" ||
                 phase.plain_key == "contraction_seconds" ||
                 phase.plain_key == "backend_total_seconds");
            const bool material = compact_dominant_timing &&
                phase.seconds > 0.0 && std::isfinite(phase.seconds) &&
                phase.seconds / total->seconds >= material_phase_fraction;
            if (!compact_dominant_timing || retained_full_tensor_phase ||
                material) {
                visible_phases.push_back(&phase);
            }
        }

        const std::string indent(13, ' ');
        const std::string branch = presentation_.utf8 ? "├─ " : "+- ";
        const std::string last_branch = presentation_.utf8 ? "└─ " : "`- ";
        std::size_t label_width = 0;
        for (const DiagnosticTiming* phase : visible_phases) {
            label_width = std::max(label_width, phase->label.size());
        }

        std::string timing_state = "timing";
        if (!cache_status.empty()) {
            timing_state += presentation_.utf8 ? " · cache " : " ; cache ";
            timing_state += cache_status;
        }
        std::ostringstream message;
        message << interactive_header(
            operation_, backend_, timing_state, presentation_.utf8);
        std::size_t index = 0;
        for (const DiagnosticTiming* phase : visible_phases) {
            message << '\n' << indent
                    << (++index == visible_phases.size()
                            ? last_branch
                            : branch)
                    << std::left << std::setw(static_cast<int>(label_width + 2))
                    << phase->label << std::right
                    << format_human_duration(
                           phase->seconds, presentation_.utf8);
        }
        emit_interactive(message.str());
    } catch (...) {
        // Timing presentation is informational only.
    }
}

void Diagnostics::emit(const std::string& message) const noexcept
{
    if (!presentation_.interactive) {
        emit_plain(message);
        return;
    }
    try {
        const ParsedDetail parsed = parse_detail(message);
        std::ostringstream rendered;
        rendered << interactive_header(
            operation_, backend_,
            parsed.fields.empty() ? std::string("detail")
                                  : detail_topic(parsed.prefix),
            presentation_.utf8);
        const std::string fields = render_fields(
            parsed.fields, presentation_);
        if (!fields.empty()) rendered << '\n' << fields;
        if (parsed.fields.empty() && !parsed.prefix.empty()) {
            rendered << '\n' << std::string(13, ' ') << parsed.prefix;
        }
        emit_interactive(rendered.str());
    } catch (...) {
        emit_plain(message);
    }
}

LibraryConfig configuration_snapshot()
{
    LibraryConfig snapshot = *active_snapshot();
    // OpenMP control variables are task-local. Apply the configured cap on the
    // application thread that is about to enter this computation.
    apply_openmp_configuration(snapshot);
    return snapshot;
}

namespace {

std::recursive_mutex tensor_output_process_mutex;

} // namespace

struct TensorOutputLease::Impl {
    explicit Impl(const fs::path& output_path)
        : process_lock(tensor_output_process_mutex)
    {
        std::error_code error;
        fs::create_directories(output_path.parent_path(), error);
        if (error) {
            throw fs::filesystem_error(
                "cannot create the full-tensor output directory",
                output_path.parent_path(),
                error);
        }

        const fs::path lock_directory = output_path.parent_path() / ".locks";
        fs::create_directories(lock_directory, error);
        if (error) {
            throw fs::filesystem_error(
                "cannot create the tensor-lock directory",
                lock_directory,
                error);
        }
        const fs::path lock_file =
            lock_directory / (output_path.filename().string() + ".lock");
        descriptor = ::open(
            lock_file.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0666);
        if (descriptor == -1) {
            throw fs::filesystem_error(
                "cannot open the full-tensor output lock",
                lock_file,
                std::error_code(errno, std::generic_category()));
        }
        if (::flock(descriptor, LOCK_EX) != 0) {
            const std::error_code lock_error(
                errno, std::generic_category());
            ::close(descriptor);
            descriptor = -1;
            throw fs::filesystem_error(
                "cannot lock the full-tensor output",
                lock_file,
                lock_error);
        }

        const std::string stale_prefix =
            output_path.filename().string() + ".part.";
        for (fs::directory_iterator iterator(output_path.parent_path(), error), end;
             !error && iterator != end;
             iterator.increment(error)) {
            const fs::directory_entry& entry = *iterator;
            if (!entry.is_regular_file(error)) {
                if (error) break;
                continue;
            }
            if (!entry.path().filename().string().starts_with(stale_prefix)) {
                continue;
            }
            fs::remove(entry.path(), error);
            if (error) break;
        }
        if (error) {
            const fs::path directory = output_path.parent_path();
            ::flock(descriptor, LOCK_UN);
            ::close(descriptor);
            descriptor = -1;
            throw fs::filesystem_error(
                "cannot clean stale full-tensor temporary output",
                directory,
                error);
        }
    }

    ~Impl()
    {
        if (descriptor != -1) {
            ::flock(descriptor, LOCK_UN);
            ::close(descriptor);
        }
    }

    std::unique_lock<std::recursive_mutex> process_lock;
    int descriptor = -1;
};

TensorOutputLease::TensorOutputLease(const fs::path& output_path)
    : impl_(std::make_unique<Impl>(output_path))
{
}

TensorOutputLease::~TensorOutputLease() = default;

struct AtomicTensorOutput::Impl {
    explicit Impl(fs::path destination)
        : final_path(std::move(destination))
    {
        static std::atomic<std::uint64_t> sequence{0};
        const auto stamp = std::chrono::high_resolution_clock::now()
                               .time_since_epoch()
                               .count();
        temporary = final_path.parent_path() /
            (final_path.filename().string() + ".part." +
             std::to_string(::getpid()) + "." +
             std::to_string(stamp) + "." +
             std::to_string(sequence.fetch_add(1)));
    }

    ~Impl()
    {
        if (!published) {
            std::error_code ignored;
            fs::remove(temporary, ignored);
        }
    }

    fs::path final_path;
    fs::path temporary;
    bool published = false;
};

AtomicTensorOutput::AtomicTensorOutput(const fs::path& final_path)
    : impl_(std::make_unique<Impl>(final_path))
{
}

AtomicTensorOutput::~AtomicTensorOutput() = default;

const fs::path& AtomicTensorOutput::temporary_path() const noexcept
{
    return impl_->temporary;
}

void AtomicTensorOutput::publish()
{
    if (impl_->published) {
        throw Error(
            ErrorCode::internal_failure,
            "full-tensor output was already published");
    }
    if (::rename(impl_->temporary.c_str(), impl_->final_path.c_str()) != 0) {
        throw fs::filesystem_error(
            "cannot atomically publish the full-tensor output",
            impl_->temporary,
            impl_->final_path,
            std::error_code(errno, std::generic_category()));
    }
    impl_->published = true;
}

} // namespace detail
} // namespace sl2cfoamxx
