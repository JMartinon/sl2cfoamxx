// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "sl2cfoamxx/configuration.hpp"
#include "core/OutputPresentation.hpp"

#include <filesystem>
#include <initializer_list>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace sl2cfoamxx::detail {

[[nodiscard]] LibraryConfig configuration_snapshot();

enum class DiagnosticOperation {
    amplitude,
    coherent_amplitude,
    full_tensor
};

// automatic is the production policy. The forced modes are private test hooks
// for deterministic rendering checks and never enter LibraryConfig.
enum class DiagnosticPresentation {
    automatic,
    plain,
    terminal_utf8,
    terminal_ascii
};

struct DiagnosticTiming {
    std::string_view label;
    std::string_view plain_key;
    double seconds = 0.0;
};

// Builds each diagnostic record privately, then emits it under one process-wide
// lock. This keeps records intact when independent API calls run concurrently.
class Diagnostics {
public:
    Diagnostics(
        const LibraryConfig& config,
        DiagnosticOperation operation,
        Backend backend,
        DiagnosticPresentation presentation =
            DiagnosticPresentation::automatic) noexcept;

    [[nodiscard]] bool enabled(Verbosity level) const noexcept;

    template <class... Values>
    void started(Values&&... values) const noexcept
    {
        if (!enabled(Verbosity::summary)) return;
        try {
            std::ostringstream details;
            (details << ... << std::forward<Values>(values));
            emit_event(Event::started, 0.0, details.str());
        } catch (...) {
            // Reporting is best-effort and must not change computation results.
        }
    }

    template <class... Values>
    void completed(double elapsed_seconds, Values&&... values) const noexcept
    {
        if (!enabled(Verbosity::summary)) return;
        try {
            std::ostringstream details;
            (details << ... << std::forward<Values>(values));
            emit_event(Event::completed, elapsed_seconds, details.str());
        } catch (...) {
            // Reporting is best-effort and must not change computation results.
        }
    }

    void failed(double elapsed_seconds, std::string_view reason) const noexcept;

    // Warnings are explicit policy output, so they are emitted independently
    // of the informational verbosity level.
    template <class... Values>
    void warning(std::string_view subject, Values&&... values) const noexcept
    {
        try {
            std::ostringstream details;
            (details << ... << std::forward<Values>(values));
            emit_warning(subject, details.str());
        } catch (...) {
            // Reporting is best-effort and must not change computation results.
        }
    }

    void timing(
        std::initializer_list<DiagnosticTiming> phases,
        std::string_view cache_status = {}) const noexcept;

    template <class... Values>
    void log(Verbosity level, Values&&... values) const noexcept
    {
        if (!enabled(level)) return;
        try {
            std::ostringstream message;
            (message << ... << std::forward<Values>(values));
            emit(message.str());
        } catch (...) {
            // Reporting is best-effort and must not change computation results.
        }
    }

private:
    enum class Event {
        started,
        completed,
        failed
    };

    void emit_event(
        Event event,
        double elapsed_seconds,
        const std::string& payload) const noexcept;
    void emit_warning(
        std::string_view subject,
        const std::string& payload) const noexcept;
    void emit_plain(const std::string& message) const noexcept;
    void emit_interactive(const std::string& message) const noexcept;
    void emit(const std::string& message) const noexcept;

    Verbosity verbosity_ = Verbosity::silent;
    std::ostream* stream_ = nullptr;
    DiagnosticOperation operation_ = DiagnosticOperation::amplitude;
    Backend backend_ = Backend::CPU;
    OutputPresentation presentation_{};
};

// Serializes lookup/publication for one configured tensor path. The lock file
// is stable and intentionally survives individual computations.
class TensorOutputLease {
public:
    explicit TensorOutputLease(const std::filesystem::path& output_path);
    TensorOutputLease(const TensorOutputLease&) = delete;
    TensorOutputLease& operator=(const TensorOutputLease&) = delete;
    ~TensorOutputLease();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Creates a unique sibling path and publishes it with one atomic rename. A
// failed or interrupted computation can therefore leave only an unreferenced
// temporary, never a metadata-valid partial tensor at the final cache path.
class AtomicTensorOutput {
public:
    explicit AtomicTensorOutput(const std::filesystem::path& final_path);
    AtomicTensorOutput(const AtomicTensorOutput&) = delete;
    AtomicTensorOutput& operator=(const AtomicTensorOutput&) = delete;
    ~AtomicTensorOutput();

    [[nodiscard]] const std::filesystem::path& temporary_path() const noexcept;
    void publish();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sl2cfoamxx::detail
