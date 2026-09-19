<!-- SPDX-FileCopyrightText: 2026 Jeremy Martinon -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Configuration

Start with `default_config()`, change the desired fields, then call
`configure(config)`. `current_config()` returns a copy of the active settings.
A later edit to either copy takes effect only after another `configure()`.
Active computations retain their original configuration snapshot.

## Configuration defaults

The tables below give `default_config()` values. Enum names use C++ spelling;
in Python and Julia write `sf.Backend.CPU`, `sf.Verbosity.silent`, and so on.
Python uses `True`/`False`; Julia uses `true`/`false`.
Mathematica uses `DefaultConfig[]`, `CurrentConfig[]`, and `Configure[config]`,
with Association keys and string enums; its complete spelling/default table
is in the [Mathematica guide](mathematica-reference.md#configuration-defaults).

### LibraryConfig

| Field | Default | Meaning |
| --- | --- | --- |
| `backend` | `Backend::CPU` in a CPU build; `Backend::GPU` in a CUDA build | Default for coherent/full-tensor calls. Fixed amplitudes always use CPU. |
| `progress_bar` | `true` | Native contraction progress on CPU and GPU; independent of verbosity. |
| `progress_callback` | Empty | C++ default event handler. Python/Julia use per-call `progress=`; Mathematica uses `"ProgressFunction"`. |
| `simplicity_constraint` | `SimplicityConstraint::gamma_j` | `rho=gamma*j`; alternative `gamma_j_p1` uses `rho=gamma*(j+1)`. |
| `cpu_threads` | `0` (automatic) | Follow the caller's OpenMP ceiling; algorithms may use fewer workers. |
| `memory_budget_bytes` | `0` (automatic) | RAM/cgroup-aware host planning; GPU device-memory admission is separate and automatic. |
| `verbosity` | `Verbosity::silent` | Alternatives: `summary`, `timing`, `detailed`, each including the preceding levels. |
| `printing_style` | `PrintingStyle::UTF8` | Automatic ASCII fallback for a non-UTF-8 locale; `ASCII` forces it. |
| `log_stream` | `nullptr` → `std::clog` | Borrowed C++ output stream; not exposed in the language wrappers. Native output goes to stderr by default. |
| `tensors` | `TensorCacheOptions{}` | Tensor defaults below. |
| `coherent_amplitude` | `CoherentAmplitudeOptions{}` | Coherent numerical policy below. |
| `low_level` | `LowLevelOptions{}` | Expert tuning defaults below. |

Unlike `default_config()`, direct construction with C++ `LibraryConfig{}` or
Python/Julia `LibraryConfig()` starts with `backend=CPU` and an empty tensor
directory. Prefer `default_config()` to obtain the build-dependent backend
and selected cache root.

`cpu_threads` controls host work on both backends, including GPU booster
preparation. A positive value replaces OpenMP's initial `OMP_NUM_THREADS`
setting; zero follows the caller's current OpenMP ceiling, including later
`omp_set_num_threads()` changes. Algorithms may use fewer workers. The
advanced `B4_NUM_THREADS` environment variable remains an additional booster
cap. Thread counts do not allocate CPUs: request enough physical cores from
Slurm and keep independent processes' host teams disjoint.

`progress_callback` receives structured events (phase, completed/total work,
elapsed time, and ETA) for a custom display or application monitor. It is
independent of the native progress bar. A per-call callback overrides the
configured callback for that call; keep C++ handlers short and thread-safe.

| Verbosity | Added output |
| --- | --- |
| `silent` | No diagnostic log; the progress bar remains separately controlled. |
| `summary` | Operation start/end or failure, backend, and total elapsed time. |
| `timing` | Coarse phase timings and cache activity. |
| `detailed` | Execution plans, memory decisions, and detailed backend diagnostics. |

For coherent calls, detailed output includes actual booster team sizes and
per-GPU work/timing records. Linux records also include worker CPU placement.
Use silent calls with progress disabled for performance measurements; collect
these diagnostics in separate calls.

`log_stream` redirects native diagnostics and progress. Use `nullptr` for
`std::clog` (stderr), `&std::cout` for stdout, or a pointer to a C++ stream such
as `std::ofstream` for a log file or `std::ostringstream` to capture text in
memory. The stream must outlive every computation using that configuration.
For example, keep `std::ofstream log("run.log");` alive while using
`config.log_stream = &log;`. Numerical results are returned normally.

### TensorCacheOptions — `config.tensors`

| Field | Default | Meaning |
| --- | --- | --- |
| `directory` | Nonempty `SL2CFOAMXX_CACHE_DIR`, otherwise `<working-directory>/data_sl2cfoamxx` | Selected by `default_config()`; an explicit path wins. An empty path requests the same fallback at configuration time. |
| `storage` | `TensorStorage::on_disk_sym` | Memory-mapped, symmetry-aware tensor file. Alternatives: `in_memory_sym`, `in_memory_dense`, `on_disk_dense`. |
| `output_path` | Empty (`None` in Python, `nothing` in Julia) | Automatic cache tree; an exact path requires an on-disk mode. |
| `reuse_existing` | `true` | Reuse an on-disk tensor only after complete metadata validation. `false` recomputes and atomically replaces the output. |
| `maximum_payload_size_bytes` | `107374182400` (100 GiB) | Limit on physical scalar payload; `0` disables the limit. Mapped files also have a 64 KiB prefix. |

The default writes full tensors to the cache directory and reuses matching
files on later calls. Select an in-memory mode to avoid persisting results.
Configuration creates the cache directory, including for in-memory calls.
Symmetry-aware storage uses compression when the boundary permits it and
falls back to dense storage otherwise. Canonical indexed access is identical
in all modes. In Python, `tensor.numpy(copy=True)` creates a canonical dense
array; `tensor.numpy()` shares memory only for a canonical dense tensor.
`np.asarray(tensor)` chooses a dense view or compressed expansion automatically.
See [tensor representations](cpp-api.md#full-tensor-representations) and
[persistence](cpp-api.md#tensor-persistence) for layouts and cache rules.

### FullTensorPrintOptions

These per-print options are separate from library configuration.

| Field | Default | Meaning |
| --- | --- | --- |
| `max_values` | `4096` | Maximum canonical entries to print; `all_values` requests the entire tensor. |
| `precision` | `17` | Digits after the decimal point in scientific notation. |
| `include_metadata` | `true` | Include tensor metadata before values. |

Other API defaults: fixed amplitudes compute directly unless
`AmplitudeCachePolicy::prefer_full_tensor` is requested; `FullTensor::map()`
is read-only; storage conversion does not overwrite an existing destination;
and a new general `FullTensor(shape)` is zero-filled.

### CoherentAmplitudeOptions — `config.coherent_amplitude`

| Field | Default | Meaning |
| --- | --- | --- |
| `check_closure` | `false` | For angle-based calls, check all five nodes before computation; a failure warns and continues, even with silent verbosity. Skipped for supplied coefficient vectors. |
| `pruning_ratio` | `1e-8` | Skip C1 lanes below this fraction of the largest coefficient magnitude; `0` disables relative pruning. Must be finite and in `[0,1]`. |

Pruning does not change `compute_coherent_coefficients()`, prune C2–C5, or
provide a rigorous final-amplitude error bound. `check_closure()` itself
returns data without printing, with absolute and relative tolerances both
`1e-12`. It uses physical spins and tests
`residual_norm <= absolute_tolerance + relative_tolerance * scale`.

### LowLevelOptions — `config.low_level`

These settings affect performance and work partitioning. A stored value of
`0` or `automatic` requests the maintained default; it is not a literal zero
work size. Automatic choices can change in later releases.

| Field | Default | Current automatic choice / supported overrides |
| --- | --- | --- |
| `booster_rho_tile` | `0` | `16` on Apple ARM64, `4` elsewhere; overrides: `4`, `8`, `16`. Applies to host booster generation for both backends. |
| `full_tensor_gpu_row_capacity` | `0` | `8192` rows; overrides: `1024..INT_MAX`. Device-memory admission may reduce capacity. |
| `coherent_gpu_pipeline_slots` | `0` | Start with `3` slots; overrides: `1`, `2`, `3`. Memory pressure may reduce the count. |
| `full_tensor_gpu_producer_waves_per_sm` | `0` | Per-SM ceiling `8`; overrides: `1..64`. Available tasks can reduce the launched grid. |
| `full_tensor_cpu_p_batch` | `0` | `512` rows; overrides: `128`, `256`, `512`, `1024`, `2048`. Workspace counts against host admission. |
| `full_tensor_cpu_final_tile` | `FullTensorCpuFinalTile::automatic` | `tile_4x4`; alternatives: `tile_4x8`, `tile_8x4`. Applies to reflection-compact CPU output. |
