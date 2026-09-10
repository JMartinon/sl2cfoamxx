// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <locale>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <string>
#include <array>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <filesystem>

#include "full_tensor/FullTensorInputSupport_HalfSjC.hpp"
#include "full_tensor/CpuContraction.hpp"
#include "full_tensor/FullTensorPackedSJTensors_HalfSjC.hpp"
#include "full_tensor/FullTensorPackedBoosters_HalfSjC.hpp"
#include "core/Failure.hpp"
#include "core/HostMemoryBudget.hpp"
#include "core/tensors.hpp"

namespace b4support {

// ============================================================================
// Full-tensor execution/memory planner
// ============================================================================
//
// This planner deliberately does NOT allocate tensors.  It classifies whether
// the current in-core full-tensor algorithm is admissible, and whether the dense
// canonical output is already too large to be a sane target.
//
// Two entry points are provided:
//
//   1. preflight_full_tensor_output_from_channels(...)
//      Cheap, runs before build_full_tensor_input_support_plan(...).  Use this
//      to reject impossible 100-TB canonical outputs before support metadata is
//      materialized.  It cannot know packed payload sizes.
//
//   2. make_full_tensor_execution_plan_from_support(...)
//      Accurate resident-path estimate using the materialized support plan.
//      Use this after build_full_tensor_input_support_plan(...), before filling
//      packed sj/booster payloads.
// ============================================================================

enum class FullTensorPlannerMode {
    ResidentFast,
    ResidentFastNoCanonicalScatter,
    NeedsSjStreaming,
    NeedsBoosterStreamingOrImplicitSupport,
    // The caller supplied a memory budget smaller than the fixed resident
    // costs (safety margin + other external bytes).  No amount of
    // sj/booster streaming can make this budget meaningful; reduce the safety
    // margin for artificial tests or increase the budget.
    BudgetBelowFixedCost,
    RefuseOutputTooLarge
};

const char* full_tensor_planner_mode_name(FullTensorPlannerMode mode) noexcept;

struct FullTensorPlannerOptions {
    // If zero, use ram_fraction times the detected physical/job hard capacity,
    // additionally capped by current system/cgroup allocation headroom.
    std::uint64_t memory_budget_bytes = 0;
    double ram_fraction = 0.80;

    // If zero, use a conservative automatic safety margin.
    std::uint64_t safety_margin_bytes = 0;

    // External resident memory that the planner should include for other
    // caller-owned long-lived workspaces.
    std::uint64_t extra_fixed_bytes = 0;

    // Dense canonical output guard.  This is deliberately independent of RAM:
    // even if a filesystem exists somewhere, writing a 100-TB dense tensor is
    // probably a cry for help.
    std::uint64_t max_canonical_output_bytes = 100ull * 1024ull * 1024ull * 1024ull;

    // Current resident backend allocates the reassociated by-i1 output buffer.
    bool include_reassociated_output = true;

    // Set true only if this run will also allocate/scatter the canonical
    // Tensor<double,5> in memory.  For large runs this should be false; use a
    // disk/tile sink instead.
    bool include_canonical_output_in_peak = false;

    // If true, canonical output > max_canonical_output_bytes selects
    // RefuseOutputTooLarge even if one only asked for reassociated output.
    // This is useful for a public API named compute_full_tensor(...), where the
    // user likely expects the actual dense full tensor eventually.
    bool refuse_large_canonical_output = true;

    // Additional caller-known contraction scratch not covered by the CPU
    // batched L/R workspace estimate below.
    std::uint64_t contraction_scratch_bytes = 0;

    // Filled by the CPU auto dispatcher from the same worker ceiling, padded
    // dimensions, and effective P_batch used by the contraction.  Keeping this
    // separate prevents caller-provided scratch from being counted twice.
    std::uint64_t cpu_batched_workspace_bytes = 0;
};

struct FullTensorPreflightOutputPlan {
    std::size_t I1 = 0;
    std::size_t I2 = 0;
    std::size_t I3 = 0;
    std::size_t I4 = 0;
    std::size_t I5 = 0;

    std::uint64_t canonical_values = 0;
    std::uint64_t canonical_output_bytes = 0;
    bool canonical_output_overflows_u64 = false;
    bool canonical_output_exceeds_guard = false;

    void print(std::ostream& os) const;
};

struct FullTensorExecutionPlan {
    FullTensorPlannerMode mode = FullTensorPlannerMode::ResidentFast;

    std::uint64_t physical_ram_bytes = 0;
    std::uint64_t memory_budget_bytes = 0;
    std::uint64_t safety_margin_bytes = 0;
    std::uint64_t external_fixed_bytes = 0;
    bool budget_below_fixed_cost = false;

    std::uint64_t sj_payload_bytes = 0;
    std::uint64_t booster_payload_bytes = 0;
    std::uint64_t support_metadata_bytes = 0;
    std::uint64_t support_total_bytes = 0;

    std::uint64_t reassociated_values = 0;
    std::uint64_t reassociated_output_bytes = 0;
    std::uint64_t canonical_values = 0;
    std::uint64_t canonical_output_bytes = 0;
    bool canonical_output_overflows_u64 = false;
    bool canonical_output_exceeds_guard = false;

    std::uint64_t resident_fast_peak_bytes = 0;
    std::uint64_t resident_boosters_only_peak_bytes = 0;
    std::uint64_t contraction_scratch_bytes = 0;
    std::uint64_t cpu_batched_workspace_bytes = 0;

    bool resident_fast_fits = false;
    bool resident_boosters_fit = false;

    FullTensorReassociatedAutoOutputPlan output_plan;

    void print(std::ostream& os) const;
};

inline std::uint64_t full_tensor_planner_sat_add_u64(std::uint64_t a,
                                                     std::uint64_t b) noexcept {
    const std::uint64_t maxv = std::numeric_limits<std::uint64_t>::max();
    return (maxv - a < b) ? maxv : a + b;
}

inline std::uint64_t full_tensor_planner_sat_mul_u64(std::uint64_t a,
                                                     std::uint64_t b,
                                                     bool* overflow = nullptr) noexcept {
    const std::uint64_t maxv = std::numeric_limits<std::uint64_t>::max();
    if (a != 0 && b > maxv / a) {
        if (overflow) *overflow = true;
        return maxv;
    }
    return a * b;
}

inline std::uint64_t full_tensor_planner_total_contraction_scratch_bytes(
    const FullTensorPlannerOptions& opt) noexcept
{
    return full_tensor_planner_sat_add_u64(
        opt.contraction_scratch_bytes,
        opt.cpu_batched_workspace_bytes);
}

std::uint64_t full_tensor_planner_physical_ram_bytes() noexcept;

std::uint64_t full_tensor_planner_default_safety_margin_bytes(std::uint64_t ram) noexcept;

std::uint64_t full_tensor_planner_memory_budget_bytes(const FullTensorPlannerOptions& opt,
                                                             std::uint64_t physical_ram) noexcept;

std::uint64_t full_tensor_planner_output_values_from_channels(const FullTensorChannels& ch,
                                                                     bool* overflow = nullptr) noexcept;

FullTensorPreflightOutputPlan
preflight_full_tensor_output_from_channels(const FullTensorChannels& ch,
                                           const FullTensorPlannerOptions& opt = FullTensorPlannerOptions{});

inline std::uint64_t full_tensor_support_sj_payload_bytes(const FullTensorInputSupportPlan& s) noexcept {
    return full_tensor_planner_sat_add_u64(
        full_tensor_planner_sat_add_u64(s.sjA.payload_bytes(), s.sjB.payload_bytes()),
        full_tensor_planner_sat_add_u64(
            s.sjC.payload_bytes(),
            full_tensor_planner_sat_add_u64(s.sjD.payload_bytes(), s.sjE.payload_bytes())));
}

inline std::uint64_t full_tensor_support_booster_payload_bytes(const FullTensorInputSupportPlan& s) noexcept {
    return full_tensor_planner_sat_add_u64(
        full_tensor_planner_sat_add_u64(s.B2.payload_bytes(), s.B3.payload_bytes()),
        full_tensor_planner_sat_add_u64(s.B4.payload_bytes(), s.B5.payload_bytes()));
}

FullTensorExecutionPlan
make_full_tensor_execution_plan_from_support(const FullTensorInputSupportPlan& support,
                                             const FullTensorPlannerOptions& opt = FullTensorPlannerOptions{});

void full_tensor_planner_print_bytes(std::ostream& os, std::uint64_t bytes);


// ============================================================================
// Full-tensor auto dispatcher, resident backend first
// ============================================================================

// The dispatcher is intentionally conservative.  It selects among the resident,
// x-slabbed 6j, and fully boundary intertwiner/booster-streamed backends only after their host
// working sets have passed the host memory planner.  The native CUDA route
// performs a second, exact live-VRAM admission from its resident packed inputs,
// seed caches, output panel, and bounded q-slab workspace immediately before
// device allocation.

enum class FullTensorOutputMode {
    // Automatic production mode.  Resident/SJ-streamed backends write the
    // reassociated reflection-compact output in RAM when it fits the selected
    // memory budget, otherwise they use Tensor's mapped-file storage.
    // The fully-streamed boundary intertwiner-tiled backend cannot currently produce a global
    // reassociated layout, so Auto falls back to canonical disk output there.
    ReassociatedAuto,

    // Force the reassociated by-i1 output to be returned in RAM.
    ReassociatedOnly,

    // Force the reassociated by-i1 output to be written to a mapped Tensor file
    // tree.  The returned std::vector is empty; use result.reassociated_disk_root.
    ReassociatedOnDisk,

    // Allocate a dense canonical column-major vector with dimensions
    // (i1,i2,i3,i4,i5) and scatter into it.  Guarded by the planner.
    CanonicalDenseInMemory,

    // Write the dense canonical tensor to a mapped Tensor file.  This is the
    // natural disk output for the fully-streamed boundary intertwiner-tiled backend.
    CanonicalDenseOnDisk,

    // Run the selected backend for timing/profiling.  A scratch output buffer is
    // still needed by the current contractors; when it would exceed the memory
    // budget, it is backed by a mapped Tensor and discarded after return.
    BenchmarkOnly
};

const char* full_tensor_output_mode_name(FullTensorOutputMode mode) noexcept;

struct FullTensorAutoTimings {
    double total_seconds = 0.0;
    double channels_seconds = 0.0;
    double preflight_seconds = 0.0;
    double support_seconds = 0.0;
    double streamed_support_seconds = 0.0;
    double planning_seconds = 0.0;
    double sj_fill_seconds = 0.0;
    double booster_fill_seconds = 0.0;
    double output_allocate_seconds = 0.0;
    double contraction_seconds = 0.0;
    double canonical_scatter_seconds = 0.0;

    void print(std::ostream& os) const;
};

struct FullTensorAutoOptions {
    FullTensorPlannerOptions planner{};

    sl2cfoamxx::SimplicityConstraint simplicity_constraint =
        sl2cfoamxx::SimplicityConstraint::gamma_j;
    std::uint32_t booster_rho_tile = 0;

    // Default library behavior: compute a reusable full tensor cache.  In
    // ReassociatedAuto mode the resident/SJ-streamed backends write the compact
    // reassociated output to disk by default, while the fully-streamed backend
    // falls back to canonical disk output because global reassociated tile
    // scattering is not implemented yet.  Use ReassociatedOnly for a pure
    // in-memory result, or BenchmarkOnly when you only want timings.
    FullTensorOutputMode output_mode = FullTensorOutputMode::ReassociatedAuto;

    // In ReassociatedAuto mode, write the compact output to disk even when it
    // would fit in RAM.  This matches the common library/cache workflow: compute
    // once, reuse later.  Set false for RAM-if-fits/disk-if-needed behavior.
    bool save_reassociated_auto_to_disk = true;

    // Reuse an existing on-disk tensor cache when the binary header/tag matches
    // the requested (gamma, Dl, doubled spins, layout, dimensions).  This avoids
    // recomputing an identical request, while still detecting a file whose
    // embedded metadata does not match its requested tensor. Set
    // force_recompute_existing_output=true to overwrite.
    bool reuse_existing_disk_output = true;
    bool force_recompute_existing_output = false;
    bool print_existing_output_status = true;

    // Disk-backed output.  Default storage is a single .sl2cxx binary
    // container produced by Tensor<T,N>::create_mapped(...).  The file
    // contains both the binary header and the raw values payload; no sidecar
    // metadata or values file is required.
    //
    // By default, output_root is empty and the dispatcher builds:
    //   <output_base_dir>/<full_tensor_output_subdir>/gamma_j/gamma_1.200/
    //       Dspins_20_..._20/Dl_4.sl2cxx
    //
    // If output_root is non-empty, it is used as an exact file path override
    // for the default reassociated output.  To choose only the base data
    // directory while keeping the automatic tree, set output_base_dir instead.
    std::filesystem::path output_base_dir = "data";
    std::filesystem::path full_tensor_output_subdir = "full_tensor";
    std::filesystem::path output_root{};
    std::string output_extension = ".sl2cxx";
    std::string canonical_output_suffix = "_canonical";

    // Guard disk outputs as well as resident allocations. Zero disables the
    // limit.
    std::uint64_t max_reassociated_output_disk_bytes =
        100ull * 1024ull * 1024ull * 1024ull;
    std::uint64_t max_canonical_output_disk_bytes =
        100ull * 1024ull * 1024ull * 1024ull;

    // Convenience override for tests.  If nonzero, compute_full_tensor_auto()
    // copies this into planner.memory_budget_bytes before planning.  This lets
    // tests to reduce the apparent memory budget and exercise streamed dispatch
    // paths without changing the lower-level planner object.
    std::uint64_t memory_budget_bytes = 0;

    // Streamed backend knobs.  A value of zero means auto.
    // For the fully-streamed backend, auto chooses the largest common boundary intertwiner-panel
    // size that passes a conservative tile-memory estimate. Panel size 1 is a
    // debugging fallback because it can create millions of tiny contractions.
    std::size_t streamed_x_per_slab = 0;
    std::size_t fully_streamed_boundary_panel_size = 0;
    // Boundary-independent sjB/sjC/sjD slabs may be reused by every boundary intertwiner tile.
    // Zero opportunistically uses otherwise-unused planner headroom, capped at
    // 1 GiB.  A nonzero value reserves up to this many bytes before choosing
    // the boundary intertwiner-panel size, which is useful when recurrence generation is more
    // expensive than the extra boundary intertwiner tiling on a particular machine.
    std::uint64_t fully_streamed_common_sj_cache_bytes = 0;
    bool print_streaming_plan = false;

    // Dispatcher permissions.  Backend selection is automatic from the memory
    // budget.  These flags are escape hatches for benchmarking/debugging only:
    // disabling one forces a clean throw if the planner selects it.
    bool allow_resident_fast = true;
    bool allow_sj_streaming = true;
    bool allow_booster_streaming = true;

    bool print_preflight = false;
    bool print_execution_plan = false;
    bool print_timing = false;
    bool print_result_summary = false;
    std::ostream* log = &std::cout;

    // Existing resident-contractor knobs, with production-silent defaults.
    FullTensorPackedContractionL25ScalarOptions contraction_options{};
    FullTensorReflectionCompactTriangularBatchedOptions triangular_options{};
    FullTensorAdaptiveScheduleOptions schedule_options{};
    FullTensorReflectionCompactWeightedProgressOptions progress_options{};

    // Keep the reassociated buffer even when canonical output is requested.
    // ReassociatedOnly always keeps it; BenchmarkOnly always discards it.
    bool keep_reassociated_output = false;

    FullTensorAutoOptions();
};

struct FullTensorAutoResult {
    FullTensorOutputMode output_mode = FullTensorOutputMode::ReassociatedAuto;
    FullTensorPreflightOutputPlan preflight{};
    FullTensorExecutionPlan execution_plan{};
    FullTensorReassociatedAutoOutputPlan output_plan{};

    FullTensorPackedContractionL25ScalarProfile contraction_profile{};
    FullTensorScheduledBalanceProfile schedule_balance{};
    FullTensorAutoTimings timings{};
    double value_checksum = 0.0;
    std::uint64_t boundary_tiles_completed = 0;
    std::uint64_t booster_leg_builds = 0;
    std::uint64_t sj_slab_builds = 0;
    std::uint64_t sj_common_cache_hits = 0;
    std::uint64_t sj_common_cache_bytes = 0;
    std::uint64_t sj_common_cache_slabs = 0;

    // Use the aligned tensor storage internally as well as at the public
    // boundary. Besides avoiding a second multi-gigabyte copy, this lets its
    // parallel zero path establish NUMA first-touch placement for resident
    // outputs before the OpenMP contraction starts.
    Tensor<double, 1> reassociated_output;
    Tensor<double, 5> canonical_output;
    std::array<std::size_t, 5> canonical_dims{{0,0,0,0,0}};

    bool ran_resident_backend = false;
    bool reassociated_materialized = false;
    bool reassociated_materialized_on_disk = false;
    bool canonical_materialized = false;
    bool canonical_materialized_on_disk = false;
    std::filesystem::path reassociated_disk_root{};
    std::filesystem::path canonical_disk_root{};

    bool reused_existing_disk_output = false;
    std::filesystem::path reused_existing_disk_path{};
    std::string reused_existing_disk_layout{};

    void print_summary(std::ostream& os) const;
};

class FullTensorAutoScopedTimer {
public:
    explicit FullTensorAutoScopedTimer(double& dst)
        : dst_(dst), start_(clock::now()) {}

    ~FullTensorAutoScopedTimer() {
        dst_ += std::chrono::duration<double>(clock::now() - start_).count();
    }

private:
    using clock = std::chrono::high_resolution_clock;
    double& dst_;
    clock::time_point start_;
};

double full_tensor_value_checksum(const double* values, std::size_t count) noexcept;

std::size_t full_tensor_checked_size_t_from_u64(std::uint64_t n,
                                                       const char* what);

std::string full_tensor_format_bytes_string(std::uint64_t bytes);

[[noreturn]] void full_tensor_throw_unavailable_backend(
    const FullTensorExecutionPlan& plan);


std::vector<int> full_tensor_slice_axis(const std::vector<int>& v,
                                               std::size_t begin,
                                               std::size_t end);

FullTensorChannels full_tensor_channels_with_x_slice(const FullTensorChannels& ch,
                                                            std::size_t x_begin,
                                                            std::size_t x_end);

FullTensorChannels full_tensor_channels_with_boundary_slices(
    const FullTensorChannels& ch,
    std::size_t i1_begin, std::size_t i1_end,
    std::size_t i2_begin, std::size_t i2_end,
    std::size_t i3_begin, std::size_t i3_end,
    std::size_t i4_begin, std::size_t i4_end,
    std::size_t i5_begin, std::size_t i5_end);

inline bool full_tensor_output_mode_forces_reassoc_memory(FullTensorOutputMode mode) noexcept {
    return mode == FullTensorOutputMode::ReassociatedOnly ||
           mode == FullTensorOutputMode::CanonicalDenseInMemory;
}

inline bool full_tensor_output_mode_needs_canonical_memory(FullTensorOutputMode mode) noexcept {
    return mode == FullTensorOutputMode::CanonicalDenseInMemory;
}

inline bool full_tensor_output_mode_needs_canonical_disk(FullTensorOutputMode mode) noexcept {
    return mode == FullTensorOutputMode::CanonicalDenseOnDisk;
}

std::uint64_t full_tensor_output_peak_bytes_for_mode(
    const FullTensorExecutionPlan& plan,
    const FullTensorAutoOptions& options) noexcept;

std::uint64_t full_tensor_estimate_sj_streamed_peak_bytes(
    const FullTensorExecutionPlan& plan,
    const FullTensorInputSupportPlan& slab_support,
    const FullTensorAutoOptions& options) noexcept;

std::size_t full_tensor_choose_next_x_slab_end(
    const B4Spins& spins,
    int Dl,
    const FullTensorChannels& full_channels,
    const FullTensorExecutionPlan& plan,
    const FullTensorAutoOptions& options,
    std::size_t x_begin,
    FullTensorInputSupportPlan& chosen_support,
    std::uint64_t* chosen_peak_bytes = nullptr);

inline std::size_t full_tensor_canonical_flat_index(std::size_t ii1,
                                                    std::size_t ii2,
                                                    std::size_t ii3,
                                                    std::size_t ii4,
                                                    std::size_t ii5,
                                                    std::size_t I1,
                                                    std::size_t I2,
                                                    std::size_t I3,
                                                    std::size_t I4) noexcept
{
    return ii1 + I1 * (ii2 + I2 * (ii3 + I3 * (ii4 + I4 * ii5)));
}

void full_tensor_add_local_canonical_tile_to_global(
    const std::vector<double>& local,
    Tensor<double, 5>& global,
    const std::array<std::size_t,5>& global_dims,
    std::size_t i1_begin,
    std::size_t i2_begin,
    std::size_t i3_begin,
    std::size_t i4_begin,
    std::size_t i5_begin,
    std::size_t I1t,
    std::size_t I2t,
    std::size_t I3t,
    std::size_t I4t,
    std::size_t I5t);

inline void full_tensor_auto_add_profile(FullTensorPackedContractionL25ScalarProfile& dst,
                                         const FullTensorPackedContractionL25ScalarProfile& src)
{
    dst.add(src);
}

inline FullTensorReassociatedStorageOrder
full_tensor_reassociated_storage_order_for_output_plan(
    const FullTensorReassociatedAutoOutputPlan& plan) noexcept
{
    return plan.uses_reflection_compact()
        ? FullTensorReassociatedStorageOrder::ReflectionTriangularByI1
        : FullTensorReassociatedStorageOrder::RectangularByI1;
}

// The reassociated output layout depends only on the canonical boundary intertwiner channels
// and on whether reflection 1,5,4,3,2 preserves the boundary spins.  Build the
// expected plan without materializing the much larger support metadata so the
// disk-cache fast path can still validate both physical count and storage
// order before accepting a file.
FullTensorReassociatedAutoOutputPlan
full_tensor_expected_reassociated_output_plan_from_channels(
    const std::array<int, 10>& spins,
    const FullTensorChannels& channels);


inline std::array<int, 10> full_tensor_spins_array_from_b4spins(const B4Spins& s) noexcept {
    return {{s.Dj12, s.Dj13, s.Dj14, s.Dj15,
             s.Dj23, s.Dj24, s.Dj25,
             s.Dj34, s.Dj35, s.Dj45}};
}

std::string full_tensor_gamma_path_component(double gamma);

std::string full_tensor_dspins_path_component(const std::array<int, 10>& spins);

std::string full_tensor_Dl_path_component(std::size_t Dl);

inline std::size_t full_tensor_Dl_from_support(const FullTensorInputSupportPlan& support) noexcept {
    // The full tensor support is built with l23..l45 ranges of length Dl+1.
    // If a caller ever supplies an empty support, use zero as the least harmful
    // metadata value. The computation itself will fail elsewhere for empty
    // physics support, as it should.
    return support.l23.empty() ? std::size_t{0} : support.l23.size() - 1;
}

const char* full_tensor_simplicity_path_component(
    sl2cfoamxx::SimplicityConstraint prescription) noexcept;

std::filesystem::path full_tensor_run_output_directory(const FullTensorAutoOptions& options,
                                                              const std::array<int, 10>& spins,
                                                              double gamma);

std::filesystem::path full_tensor_reassociated_disk_root(const FullTensorAutoOptions& options,
                                                                const std::array<int, 10>& spins,
                                                                std::size_t Dl,
                                                                double gamma);

std::filesystem::path full_tensor_canonical_disk_root(const FullTensorAutoOptions& options,
                                                             const std::array<int, 10>& spins,
                                                             std::size_t Dl,
                                                             double gamma);

std::filesystem::path full_tensor_canonical_reassociated_scratch_root(
    const FullTensorAutoOptions& options,
    const std::array<int, 10>& spins,
    std::size_t Dl,
    double gamma);

// Declared before the mapped Tensor at each call site so stack unwinding first
// closes the mapping and then removes the scratch file.  remove_now() is used
// on the success path to surface cleanup failures; the destructor is the
// no-throw fallback for every exceptional path.
class FullTensorScratchFileCleanup {
public:
    FullTensorScratchFileCleanup() = default;
    FullTensorScratchFileCleanup(const FullTensorScratchFileCleanup&) = delete;
    FullTensorScratchFileCleanup& operator=(
        const FullTensorScratchFileCleanup&) = delete;

    ~FullTensorScratchFileCleanup() noexcept
    {
        remove_noexcept();
    }

    void arm(std::filesystem::path file)
    {
        if (!file_.empty()) {
            throw std::logic_error(
                "full-tensor scratch cleanup was armed more than once");
        }
        file_ = std::move(file);
    }

    [[nodiscard]] bool armed() const noexcept
    {
        return !file_.empty();
    }

    void remove_now();

private:
    void remove_noexcept() noexcept
    {
        if (file_.empty()) return;
        std::error_code ignored;
        std::filesystem::remove(file_, ignored);
        file_.clear();
    }

    std::filesystem::path file_{};
};


struct FullTensorExistingDiskCheck {
    bool exists = false;
    bool reusable = false;
    std::string reason{};
    tensor_disk::TensorTreeHeaderV1 header{};
    FullTensorBinaryTagV1 tag{};
};

inline bool full_tensor_gamma_matches(double a, double b) noexcept {
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a - b) <= 1.0e-12 * scale;
}

FullTensorExistingDiskCheck full_tensor_check_existing_disk_file(
    const std::filesystem::path& file,
    tensor_disk::LayoutCode expected_layout,
    std::uint32_t expected_ndim,
    const std::array<int, 10>& expected_spins,
    std::size_t expected_Dl,
    double expected_gamma,
    const std::array<std::size_t, 5>& expected_canonical_dims,
    std::uint64_t expected_values_count_or_zero,
    sl2cfoamxx::SimplicityConstraint expected_simplicity =
        sl2cfoamxx::SimplicityConstraint::gamma_j);

FullTensorExistingDiskCheck full_tensor_check_existing_reassociated_file(
    const std::filesystem::path& file,
    const std::array<int, 10>& spins,
    std::size_t Dl,
    double gamma,
    const std::array<std::size_t, 5>& canonical_dims,
    std::uint64_t values_count_or_zero = 0,
    sl2cfoamxx::SimplicityConstraint simplicity =
        sl2cfoamxx::SimplicityConstraint::gamma_j);

FullTensorExistingDiskCheck full_tensor_check_existing_reassociated_file(
    const std::filesystem::path& file,
    const std::array<int, 10>& spins,
    std::size_t Dl,
    double gamma,
    const std::array<std::size_t, 5>& canonical_dims,
    const FullTensorReassociatedAutoOutputPlan& expected_output_plan,
    sl2cfoamxx::SimplicityConstraint simplicity =
        sl2cfoamxx::SimplicityConstraint::gamma_j);

FullTensorExistingDiskCheck full_tensor_check_existing_canonical_file(
    const std::filesystem::path& file,
    const std::array<int, 10>& spins,
    std::size_t Dl,
    double gamma,
    const std::array<std::size_t, 5>& canonical_dims,
    std::uint64_t values_count_or_zero = 0,
    sl2cfoamxx::SimplicityConstraint simplicity =
        sl2cfoamxx::SimplicityConstraint::gamma_j);

inline bool full_tensor_auto_primary_disk_cache_is_reassociated(const FullTensorAutoOptions& options) noexcept {
    return (options.output_mode == FullTensorOutputMode::ReassociatedAuto &&
            options.save_reassociated_auto_to_disk) ||
           options.output_mode == FullTensorOutputMode::ReassociatedOnDisk;
}

inline bool full_tensor_auto_primary_disk_cache_is_canonical(const FullTensorAutoOptions& options) noexcept {
    return options.output_mode == FullTensorOutputMode::CanonicalDenseOnDisk;
}

void full_tensor_throw_if_disk_output_exceeds_guard(std::uint64_t bytes,
                                                           std::uint64_t guard,
                                                           const char* what);

bool full_tensor_should_use_disk_reassociated_buffer(
    const FullTensorExecutionPlan& plan,
    const FullTensorAutoOptions& options,
    std::uint64_t backend_peak_without_reassoc) noexcept;

void full_tensor_add_local_canonical_tile_to_global_ptr(
    const std::vector<double>& local,
    double* global,
    const std::array<std::size_t,5>& global_dims,
    std::size_t i1_begin,
    std::size_t i2_begin,
    std::size_t i3_begin,
    std::size_t i4_begin,
    std::size_t i5_begin,
    std::size_t I1t,
    std::size_t I2t,
    std::size_t I3t,
    std::size_t I4t,
    std::size_t I5t);


FullTensorReflectionCompactWeightedProgressOptions
full_tensor_auto_progress_options_for_backend(
    const FullTensorAutoOptions& options,
    const char* label,
    const char* unit_label,
    std::uint64_t update_granularity = 1);

std::uint64_t full_tensor_fully_streamed_output_peak_bytes_for_mode(
    const FullTensorExecutionPlan& plan,
    const FullTensorAutoOptions& options) noexcept;

std::uint64_t full_tensor_estimate_fully_streamed_tile_peak_bytes(
    const FullTensorExecutionPlan& plan,
    const FullTensorInputSupportPlan& tile_fullx_support,
    const FullTensorInputSupportPlan& tile_x_support,
    const FullTensorReassociatedAutoOutputPlan& tile_output_plan,
    const FullTensorAutoOptions& options) noexcept;

std::uint64_t full_tensor_estimate_fully_streamed_peak_for_panel(
    const B4Spins& spins,
    int Dl,
    const FullTensorChannels& FTC,
    const FullTensorExecutionPlan& plan,
    const FullTensorAutoOptions& options,
    std::size_t panel_size);

std::size_t full_tensor_choose_fully_streamed_boundary_panel_size(
    const B4Spins& spins,
    int Dl,
    const FullTensorChannels& FTC,
    const FullTensorExecutionPlan& plan,
    const FullTensorAutoOptions& options,
    std::uint64_t* chosen_peak_bytes = nullptr);

inline std::uint64_t full_tensor_panel_tile_count(std::size_t n, std::size_t p) noexcept {
    if (p == 0) return 0;
    return static_cast<std::uint64_t>((n + p - 1) / p);
}

FullTensorAutoResult compute_full_tensor_resident_fast(
    const B4Spins& spins,
    double gamma,
    const std::vector<double>& FactTable,
    const FullTensorChannels& FTC,
    const FullTensorInputSupportPlan& support,
    const FullTensorExecutionPlan& execution_plan,
    const FullTensorAutoOptions& options = FullTensorAutoOptions{});


FullTensorAutoResult compute_full_tensor_sj_streamed(
    const B4Spins& spins,
    std::size_t Dl,
    double gamma,
    const std::vector<double>& FactTable,
    const FullTensorChannels& FTC,
    const FullTensorInputSupportPlan& full_support,
    const FullTensorExecutionPlan& execution_plan,
    const FullTensorAutoOptions& options = FullTensorAutoOptions{});

FullTensorAutoResult compute_full_tensor_fully_streamed(
    const B4Spins& spins,
    std::size_t Dl,
    double gamma,
    const std::vector<double>& FactTable,
    const FullTensorChannels& FTC,
    const FullTensorExecutionPlan& execution_plan,
    const FullTensorAutoOptions& options = FullTensorAutoOptions{});

FullTensorAutoResult compute_full_tensor_auto(
    const std::array<int, 10>& SpinsArray,
    std::size_t Dl,
    double gamma,
    const std::vector<double>& FactTable,
    FullTensorAutoOptions options = FullTensorAutoOptions{});

FullTensorAutoResult compute_full_tensor_auto(
    const std::array<int, 10>& SpinsArray,
    std::size_t Dl,
    double gamma,
    FullTensorAutoOptions options = FullTensorAutoOptions{},
    std::size_t log_fact_max = 1000);

} // namespace b4support
