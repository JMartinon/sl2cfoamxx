// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "sl2cfoamxx/export.hpp"
#include "sl2cfoamxx/progress.hpp"
#include "sl2cfoamxx/types.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>

namespace sl2cfoamxx {

// Informational diagnostics are monotonic: each level includes the preceding
// levels. Library errors are still reported with exceptions at every level.
enum class Verbosity : std::uint8_t {
    silent = 0,
    summary = 1,
    timing = 2,
    detailed = 3
};

// Human-facing terminal presentation. UTF8 is the default and retains the
// automatic ASCII safety fallback for an explicitly non-UTF-8 locale. ASCII
// disables Unicode glyphs even when the terminal supports them.
enum class PrintingStyle : std::uint8_t {
    UTF8 = 0,
    ASCII = 1
};

// Convenience spelling for applications that prefer an option-like lowercase
// type name.
using printing_style = PrintingStyle;

// Shape of the small CPU microkernel used for the final reflection-compact
// full-tensor update. `automatic` selects the maintained production default.
enum class FullTensorCpuFinalTile : std::uint8_t {
    automatic = 0,
    tile_4x4 = 1,
    tile_4x8 = 2,
    tile_8x4 = 3
};

struct TensorCacheOptions {
    // Data root containing full_tensor/{gamma_j,gamma_j_p1}/gamma_.../
    // Dspins_.../Dl_N.sl2cxx for dense files and Dl_N_sym.sl2cxx for
    // symmetry-aware files. When empty, this uses SL2CFOAMXX_CACHE_DIR or
    // <current-directory>/data_sl2cfoamxx.
    std::filesystem::path directory{};

    // These settings apply to every subsequent compute_full_tensor() call.
    TensorStorage storage = TensorStorage::on_disk_sym;

    // Empty selects the automatic directory tree below directory. A nonempty
    // path is an exact file override and requires an on-disk storage mode.
    std::filesystem::path output_path{};

    // Reuse only files whose complete physics metadata matches. False
    // recomputes and atomically replaces the configured output.
    bool reuse_existing = true;

    // Physical scalar-payload guard for the selected representation. A mapped
    // .sl2cxx container additionally has a fixed 64-KiB metadata/alignment
    // prefix. Zero disables the guard.
    std::uint64_t maximum_payload_size_bytes =
        100ull * 1024ull * 1024ull * 1024ull;
};

// Expert tuning controls for implementation details that can materially
// change performance without changing the mathematical result. Zero selects
// the production default so applications can remain portable across hosts.
struct LowLevelOptions {
    // Host-side rho-lane width used while generating booster tensors for every
    // public computation, including GPU calls. Supported overrides are 4, 8,
    // and 16; zero selects the architecture-specific production default.
    std::uint32_t booster_rho_tile = 0;

    // Preferred materialized-row capacity for the full-tensor GPU
    // contraction. Zero selects the production default (currently 8192).
    // Device-memory admission may reduce the effective capacity.
    std::uint64_t full_tensor_gpu_row_capacity = 0;

    // Maximum number of coherent-GPU pipeline slots requested initially.
    // Zero selects the production default (currently 3). Memory admission may
    // retry with fewer slots; explicit supported values are 1, 2, and 3.
    std::uint32_t coherent_gpu_pipeline_slots = 0;

    // Per-SM ceiling used to cap the left and right producer grids in the
    // full-tensor GPU backend. Zero selects the production default (currently
    // 8); explicit values must be in [1,64]. Each launch remains limited by
    // its available producer-task count, so this does not force that many
    // waves to execute.
    std::uint32_t full_tensor_gpu_producer_waves_per_sm = 0;

    // Row-panel capacity used by the final full-tensor CPU contraction. Zero
    // selects the production default (currently 512). Supported overrides are
    // 128, 256, 512, 1024, and 2048. Its padded per-worker L/R workspace is
    // included in host-memory admission.
    std::uint32_t full_tensor_cpu_p_batch = 0;

    // CPU microkernel shape for the final reflection-compact full-tensor
    // update. General-spin rectangular output uses its separate adaptive
    // BLAS/local route and does not use this setting.
    FullTensorCpuFinalTile full_tensor_cpu_final_tile =
        FullTensorCpuFinalTile::automatic;
};

// Options specific to coherent-boundary amplitude computations.
struct CoherentAmplitudeOptions {
    // Verify sum_{b!=a} j_ab*n_ab at all five boundary tetrahedra before the
    // computation. A failed check emits a warning and computation continues;
    // it is not a substitute for the public check_closure() report when an
    // application needs to make a control-flow decision.
    bool check_closure = false;

    // Discard C1 coefficients whose magnitude is below this fraction of the
    // largest |C1|, then skip their corresponding contraction work. Zero
    // disables relative pruning. Values must be finite and lie in [0,1].
    double pruning_ratio = 1.0e-8;
};

struct LibraryConfig {
    TensorCacheOptions tensors{};

    // Shared default for compute_coherent_amplitude() and
    // compute_full_tensor(). default_config() selects GPU in a CUDA-enabled
    // build and CPU otherwise. Coherent-amplitude and full-tensor overloads
    // with an explicit Backend override this value for that call;
    // compute_amplitude() is CPU-only.
    Backend backend = Backend::CPU;

    // Display contraction progress for compute_coherent_amplitude() and
    // compute_full_tensor(). This is independent of verbosity: callers may
    // request progress-only output or combine it with diagnostic records. A
    // default interactive terminal gets a live line; redirected output and a
    // custom log_stream get newline-delimited milestone records.
    bool progress_bar = true;

    // Optional structured-event handler for custom progress displays or
    // application monitoring. Independent of progress_bar; an explicit
    // per-call callback overrides this configuration default.
    ProgressCallback progress_callback{};

    // Simplicity map shared by all three public computations. gamma_j uses
    // rho=gamma*j; gamma_j_p1 uses rho=gamma*(j+1).
    SimplicityConstraint simplicity_constraint =
        SimplicityConstraint::gamma_j;

    // Zero follows the caller's current OpenMP ceiling. A positive value
    // replaces that ceiling, including the initial OMP_NUM_THREADS setting.
    // Algorithms may use fewer workers when useful work is limited.
    std::size_t cpu_threads = 0;

    // Shared host-memory budget for CPU planners and the full-tensor GPU host
    // preflight. Zero selects their cgroup/RAM-aware automatic policy.
    std::uint64_t memory_budget_bytes = 0;

    // Silent is the normal library default. Summary emits start/completion
    // records, timing adds coarse phase timings, and detailed adds inputs plus
    // execution and memory-planner decisions.
    Verbosity verbosity = Verbosity::silent;

    // Preferred interactive diagnostics/progress style. UTF8 uses the modern
    // presentation on a compatible terminal and safely falls back to ASCII
    // for an explicit C/POSIX locale. ASCII always uses the fallback glyphs.
    PrintingStyle printing_style = PrintingStyle::UTF8;

    // Destination for informational diagnostics. nullptr selects std::clog.
    // A non-null stream is borrowed and must outlive every computation using
    // this configuration snapshot.
    std::ostream* log_stream = nullptr;

    // Advanced performance controls. Ordinary applications should leave these
    // at their automatic values and rely on the production defaults.
    LowLevelOptions low_level{};

    // Numerical policy and validation controls used only by coherent-boundary
    // amplitude calls.
    CoherentAmplitudeOptions coherent_amplitude{};
};

// The default tensor-cache root uses SL2CFOAMXX_CACHE_DIR when it is nonempty;
// otherwise it uses <current-working-directory>/data_sl2cfoamxx. An explicit
// tensors.directory takes precedence. This function only chooses paths;
// configure() (including implicit first-use configuration) creates the
// resolved directory recursively.
[[nodiscard]] SL2CFOAMXX_API LibraryConfig default_config();

// Validate paths/options, create the writable tensor-cache directory, and publish
// an immutable snapshot for subsequent computations. Active computations keep
// the snapshot with which they started.
SL2CFOAMXX_API void configure(const LibraryConfig& config);

// Return the normalized active configuration. If configure() has not been
// called, this initializes and returns default_config().
[[nodiscard]] SL2CFOAMXX_API LibraryConfig current_config();

} // namespace sl2cfoamxx
