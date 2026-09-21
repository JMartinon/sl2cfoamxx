<!-- SPDX-FileCopyrightText: 2026 Jeremy Martinon -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# C++ reference and advanced use

Start with the [C++ guide](cpp-api.md) for a first calculation. This page
collects contracts needed for coherent boundaries, custom tensor processing,
persistence, and application integration. All settings and their defaults are
listed separately in [configuration](configuration.md).

- [Coherent boundaries](#coherent-boundaries): normals, coefficient bases, closure and pruning.
- [Tensor storage](#tensor-storage): raw buffers, metadata, printing and conversion.
- [Cache paths and reuse](#cache-paths-and-reuse).
- [Progress callbacks](#progress-callbacks) and [errors](#error-codes).
- [ABI and linking](#shared-library-abi-visibility).

## Coherent boundaries

`BoundaryAngles` is one `20 x 2` array. Every row is `{theta_ab, phi_ab}` in
radians, and contains one oriented normal in this canonical order:

| Rows | Normals |
| --- | --- |
| `0..3` | `n12, n13, n14, n15` |
| `4..7` | `n21, n23, n24, n25` |
| `8..11` | `n31, n32, n34, n35` |
| `12..15` | `n41, n42, n43, n45` |
| `16..19` | `n51, n52, n53, n54` |

Thus `n_ab` and `n_ba` are separate inputs: they live in different
tetrahedron frames even when the glued geometry makes their transported
vectors antiparallel. The library gathers them into the five full-tensor
recoupling bases as follows:

| Coefficient | Doubled spins and matching normals |
| --- | --- |
| `C1` | `(j12,n12), (j13,n13), (j14,n14), (j15,n15)` |
| `C2` | `(j23,n23), (j24,n24), (j25,n25), (j12,n21)` |
| `C3` | `(j34,n34), (j35,n35), (j13,n31), (j23,n32)` |
| `C4` | `(j45,n45), (j14,n41), (j24,n42), (j34,n43)` |
| `C5` | `(j15,n51), (j25,n52), (j35,n53), (j45,n54)` |

No caller-side permutation is required. For example:

```cpp
sl2cfoamxx::BoundaryAngles angles{};
angles[0] = {theta12, phi12};
angles[1] = {theta13, phi13};
// ... fill n14 through n53 in rows 2 through 18 ...
angles[19] = {theta54, phi54};

const auto amplitude = sl2cfoamxx::compute_coherent_amplitude(
    spins, Dl, gamma, angles);
```

### Boundary presets

Both reference geometries are included by `<sl2cfoamxx.hpp>` (or
`<sl2cfoamxx/sl2cfoamxx.hpp>`). They use the exact angle values in the
[coherent boundary example](../examples/coherent_boundary_example.cpp), in
radians and the canonical row order above:

```cpp
const sl2cfoamxx::DSpins spins{10,10,10,10,4,4,4,4,4,4};
auto angles = sl2cfoamxx::lorentzian_boundary_angles;
const auto amplitude = sl2cfoamxx::compute_coherent_amplitude(
    spins, 2, 1.2, angles);

auto other_angles = sl2cfoamxx::euclidean_boundary_angles;
other_angles[0][0] = 0.1; // Only this copy changes.
```

The constants are `inline constexpr BoundaryAngles`; copying them requires
no library initialization. `lorentzian_bounday_angles` and
`euclidean_bounday_angles` are aliases for the same immutable values.
The Lorentzian geometry closes for face-spin ratios `(5,5,5,5,2,2,2,2,2,2)`;
the Euclidean geometry closes for equal face spins. They are fixed boundary
data, so choosing a preset does not change the model, backend, or configuration.
Use `check_closure` with your chosen spins as usual.

### One-tetrahedron coefficients

The same one-tetrahedron convention is public. Entries of `node_spins` and
`node_angles` correspond position by position, and the returned coefficients
are ordered by ascending admissible doubled intertwiner in the
`(0,1)|(2,3)` basis. Position zero is the lower end of that admissible range,
not necessarily doubled intertwiner zero. For doubled spins `J1..J4`, the
range starts at `max(|J1-J2|, |J3-J4|)` (adjusted for parity), ends at
`min(J1+J2, J3+J4)`, and advances in steps of two.

```cpp
sl2cfoamxx::TetrahedronDSpins node_spins{2*j12, 2*j13, 2*j14, 2*j15};
sl2cfoamxx::TetrahedronAngles node_angles{{
    angles[0], angles[1], angles[2], angles[3]}};
const auto C1 = sl2cfoamxx::compute_coherent_coefficients(
    node_spins, node_angles);

const auto closure = sl2cfoamxx::check_closure(spins, angles);
if (!closure.all_closed) {
    const auto& node1 = closure.nodes[0];
    std::cerr << "node 1 residual = " << node1.residual_norm << '\n';
}
```

### Caller-supplied coefficients

Use the five-vector overload to contract any boundary coefficient vectors,
including vectors returned by `compute_coherent_coefficients()`:

```cpp
const auto amplitude = sl2cfoamxx::compute_coherent_amplitude(
    spins, Dl, gamma, C1, C2, C3, C4, C5);
const auto cpu_amplitude = sl2cfoamxx::compute_coherent_amplitude(
    spins, Dl, gamma, C1, C2, C3, C4, C5, sl2cfoamxx::Backend::CPU);
const auto tracked_amplitude = sl2cfoamxx::compute_coherent_amplitude(
    spins, Dl, gamma, C1, C2, C3, C4, C5, sl2cfoamxx::Backend::CPU,
    [](const sl2cfoamxx::ProgressEvent& event) { /* inspect progress */ });
```

Each `C` is a `CoherentCoefficients` (`std::vector<std::complex<double>>`).
The five vectors use the node bases in the table above, pairing the first two
spins with the final two in each row. Supply every admissible doubled
intertwiner in ascending order with stride two. The length must equal the
corresponding canonical full-tensor extent; sparse vectors must include
explicit zeros for omitted lanes. Each real and imaginary part must be
finite. Wrong lengths, NaNs, and infinities throw `std::invalid_argument`
before computation or progress reporting begins.

With `pruning_ratio = 0`, the result is the plain multilinear contraction

```text
sum A[i1,i2,i3,i4,i5] * C1[i1] * C2[i2] * C3[i3] * C4[i4] * C5[i5].
```

The vectors are neither normalized nor conjugated or modified. Scaling one
vector scales the amplitude by the same complex factor, and any zero vector
gives zero. CPU/GPU selection, the simplicity constraint, C1 pruning, and
progress callback replacement follow the angle overloads. No full tensor is
materialized by this call. Closure checking is skipped even if enabled in
the configuration, since arbitrary coefficients do not supply normals.
Finiteness validation does not guarantee that every floating-point intermediate
is representable; see the [numerical range note](scientific-conventions.md#supplied-boundary-coefficients)
when supplying extreme coefficient scales.

### Closure and pruning

Closure vectors and scales use physical `j=J/2`, not doubled labels. The
default criterion at each node is
`residual_norm <= 1e-12 + 1e-12 * scale`; both tolerances can be supplied
explicitly. The checker returns data and never prints.

The same check can be requested automatically for coherent-amplitude
calls with angles. A failed check emits one warning and continues; because this is an
explicit validation request, the warning is emitted even when informational
verbosity is `silent`. Use the public `check_closure()` result instead when a
failed check should change application control flow.

```cpp
auto config = sl2cfoamxx::default_config();
config.coherent_amplitude.check_closure = true;

// Relative C1/i1 pruning. Zero disables approximate relative pruning.
config.coherent_amplitude.pruning_ratio = 1.0e-8;
sl2cfoamxx::configure(config);
```

`pruning_ratio` applies to both angle and coefficient inputs, and must be
finite and in `[0,1]`. It removes C1 lanes satisfying
`|C1[i1]| < pruning_ratio * max(|C1|)` and skips the corresponding contraction
work; it does not alter the coefficient vector returned by
`compute_coherent_coefficients()`, prune C2 through C5, or provide a rigorous
bound on the final amplitude error. The default `1e-8` preserves the existing
production policy. A complete executable with Julia-validated Lorentzian and
Euclidean boundary data is provided in
[`examples/coherent_boundary_example.cpp`](../examples/coherent_boundary_example.cpp).
Copy that source into your own application directory and build it using the
[external CMake consumer](cpp-api.md#quick-start), changing the source filename in
`add_executable()` to `coherent_boundary_example.cpp`. Examples are independent
consumers; enabling the library's test build is not required.

## Tensor storage

### Logical indices and physical buffers

`Tensor<T, Rank>` remains the general rectangular, column-major storage
container. `FullTensor` composes that storage and adds the rank-five logical
view, reflection aliasing, reassociated layouts, persistence conversion, and
physics metadata. These two roles deliberately remain separate: treating a
compressed rank-one payload as an ordinary rectangular rank-five `Tensor`
would make its physical shape and strides misleading. Applications and
language bindings should expose `FullTensor` for vertex results and use
`Tensor<T, Rank>` only when they explicitly need the generic storage primitive.

`shape()`, `dims()`, `extent()`, and `strides()` describe the logical canonical
tensor. In contrast, `size()` and `stored_size()` are the number of physically
stored values. `size_bytes()`, `stored_bytes()`, `data()`, `span()`, iterators,
`operator[]`, and `at_flat()` also expose physical storage. For a compressed
tensor this payload is smaller and is not in canonical dense order; do not
combine its raw pointer with the logical strides.

Reflection-compressed storage keeps one representative of

```text
A(i1,i2,i3,i4,i5) = A(i1,i5,i4,i3,i2).
```

Those two logical coordinates therefore alias the same stored `double`.
Writing through either coordinate updates the value observed through the
other. This is intentional; select a dense mode if the two entries must become
independent.

Use `dense_copy()` when a later algorithm needs a simple canonical contiguous
buffer:

```cpp
sl2cfoamxx::FullTensor dense = tensor.dense_copy();
for (double& entry : dense) {
    // Canonical dense payload; i1 is the contiguous axis.
}
```

The returned copy has owned `in_memory_dense` storage. `logical_size()` and
`dense_size()` are the canonical shape product, while `dense_bytes()` reports
the corresponding dense allocation size.

### Scientific metadata

`tensor.metadata().scientific` is present for tensors returned by `compute_full_tensor()`,
including the legitimate all-zero input, and absent for a `FullTensor`
constructed only as a general container. Representation, shape, backing, and
path metadata are always present. New files persist that distinction; old
computed V1 cache files remain readable. A manually constructed file saved by
a pre-0.14 build has no provenance marker and is therefore necessarily decoded
like the historical zero-valued scientific tag; resave such a container with
0.14 if that distinction matters to an application.

### Printing

For a numerical comparison or a small-tensor inspection, `print()` emits
zero-based canonical entries at full diagnostic precision, regardless of the
physical storage representation:

```cpp
tensor.print(); // writes metadata and at most 4096 logical values to std::cout

sl2cfoamxx::FullTensorPrintOptions print_options;
print_options.max_values =
    sl2cfoamxx::FullTensorPrintOptions::all_values;
tensor.print(std::cout, print_options); // intentional complete dump
```

The entry labels are `A[i1,i2,i3,i4,i5]`, with `i1` varying fastest. The
default limit prevents an accidental multi-million-line dump and reports the
exact number of omitted logical values. `precision` controls the digits after
the decimal point in scientific notation, and `include_metadata=false` emits
only value records. The caller's stream formatting is restored afterward.

### Storage conversion

Existing mapped tensor files can be converted losslessly between the two
on-disk representations without recomputing the amplitude:

```cpp
const auto compact = sl2cfoamxx::FullTensor::map(
    "vertex_sym.sl2cxx", sl2cfoamxx::TensorAccess::read_only);
auto dense = compact.convert_storage(
    sl2cfoamxx::TensorStorage::on_disk_dense,
    "vertex_dense.sl2cxx");

auto compact_again = dense.convert_storage(
    sl2cfoamxx::TensorStorage::on_disk_sym,
    "vertex_sym_roundtrip.sl2cxx");
```

The source must be an `on_disk_sym` or `on_disk_dense` mapping, and the
destination must select the opposite on-disk mode. The source is never
modified. The explicit destination path must not identify the source and is
not replaced by default; pass `true` as the third argument only when replacing
an existing destination is intentional. Publication is atomic and the
returned tensor is a writable mapping of the completed file.

Dense-to-symmetry conversion first verifies every reflected pair bit for bit.
It throws instead of discarding information if the boundary metadata is not
reflection invariant or a dense tensor has been modified asymmetrically. This
also preserves signed zeros and NaN payloads exactly when conversion succeeds.

## Cache paths and reuse

### Choosing the root

Tensor-cache root selection is controlled by `tensors.directory`. A nonempty
value assigned to it wins. An empty field uses a nonempty
`SL2CFOAMXX_CACHE_DIR`; when the variable is unset or empty, it uses
`data_sl2cfoamxx` below the process's current working directory. The former
`XDG_CACHE_HOME`, `~/.cache`, and temporary-directory fallbacks are not used.

Explicit relative paths, and a relative `SL2CFOAMXX_CACHE_DIR`, are resolved
against the current working directory during `configure()`. With no environment
override, `default_config()` captures its working-directory fallback when it is
called. Changing the working directory after configuration does not relocate an
active cache. `default_config()` only chooses the path; `configure()` creates
the resolved directory recursively. Calling `current_config()` or starting the
first computation performs that configuration implicitly. Only an unwritable
path or a conflicting non-directory causes directory creation to fail.

### File names and matching

For `gamma=1.2`, `Dl=6`, and the doubled spins shown below, the two on-disk
requests use distinct automatic filenames in the same storage tree:

```text
data/
└── full_tensor/
    └── gamma_j/
        └── gamma_1.200/
            └── Dspins_150_150_150_150_60_60_60_60_60_60/
                ├── Dl_6.sl2cxx       # on_disk_dense
                └── Dl_6_sym.sl2cxx   # on_disk_sym
```

Only the file for the selected mode is created. The `_sym` suffix records a
symmetry-aware request; `storage()` and the binary layout metadata remain the
authority on the effective representation if asymmetric spins force a dense
fallback.

Selecting `SimplicityConstraint::gamma_j_p1` replaces the `gamma_j` directory
component with `gamma_j_p1`, so tensors produced by the two conventions cannot
collide in the automatic cache tree. With an exact `tensors.output_path`
override and reuse enabled, a file with a different simplicity convention,
shape, physics input, or dense/compact layout is rejected without being
overwritten. Set `tensors.reuse_existing=false` to authorize its atomic
replacement; a subsequent compatible call can then reuse it.

Files created before the simplicity option existed used `gamma_j`; their binary
tags are still compatible. Early public full-tensor caches were dense and
used the unsuffixed `Dl_N.sl2cxx` name used by `on_disk_dense`. To reuse an
older automatic cache, move its `gamma_...` subtree below
`full_tensor/gamma_j/`, or select the file explicitly with
`tensors.output_path`.

Binary tensor metadata records the simplicity constraint, exact gamma bits, spins, `Dl`, shape, and layout, so a
three-decimal path-name collision or an explicit output-path reuse cannot
silently select the wrong tensor.

The returned mapping is writable. User changes therefore persist in a reused
cache file. Set `config.tensors.reuse_existing=false` to replace an invalid or
intentionally modified file. Set `config.tensors.output_path` for an exact file
override instead of the automatic tree. Mapped outputs are computed into unique
sibling files and atomically published only after a successful flush, so an
interrupted run cannot turn a partial tensor into a valid cache hit. Hidden
`.locks` directories coordinate concurrent processes. The `.scratch`
subdirectory below the configured tensor directory is reserved by the GPU
backend and cannot be selected as an explicit output location.

The four-argument `compute_amplitude()` always computes directly on CPU.
With `AmplitudeCachePolicy::prefer_full_tensor`, the explicit `output_path`
is checked first (and exclusively when set); otherwise both automatic dense
and symmetry-aware names are considered. A cache miss, malformed file, or
physics mismatch falls back to direct calculation. In-memory tensors are not
registered globally: read them directly with `tensor.at(indices)`.

## Progress callbacks

Coherent and full-tensor computations accept a `ProgressCallback` after an
explicit backend. The native bar is controlled separately by `progress_bar`
(default `true`). For example, capture only the final reported phase count:

```cpp
std::size_t completed_units = 0;
auto tensor = sl2cfoamxx::compute_full_tensor(
    spins, Dl, gamma, sl2cfoamxx::Backend::CPU,
    [&completed_units](const sl2cfoamxx::ProgressEvent& event) {
        if (event.status == sl2cfoamxx::ProgressStatus::completed) {
            completed_units = event.completed;
        }
    });
// Read completed_units only after the synchronous call has returned.
```

The callback reports operation, phase, backend, retry attempt, completed and
total units, elapsed time, and optional ETA. Events are serialized within one
computation but may arrive on the calling thread or the library's reporting
thread. A binding should marshal them to its runtime/UI thread and keep the
handler short. Callback exceptions are ignored so presentation cannot change a
numerical result. The explicit per-call callback replaces the configured
callback for that call only, so concurrent wrapper calls can route independent
progress without reconfiguration. Passing an empty callback suppresses the
configured callback for that call. `LibraryConfig::progress_callback` remains
useful as an application-wide default, and it is independent of
`progress_bar`; either or both sinks may be enabled.

`ProgressStatus::completed` completes the reported phase, not necessarily the
entire public call; a full tensor may still need final materialization or file
publication. In the low-memory routes, `ProgressPhase::streamed_computation`
honestly identifies a mixed pass that includes streamed 6j generation and
contraction rather than claiming to measure contraction alone.

The native display uses a live line in a terminal and complete records when
redirected. Silent terminal progress disappears after success; `summary` and
higher levels retain the completed line. A cache hit may have no contraction
progress. Fixed-intertwiner calls have no progress bar. UTF-8 presentation has
an ASCII fallback; `PrintingStyle::ASCII` forces it. Diagnostics and progress
share `log_stream` (default `std::clog`); the borrowed stream must outlive all
calls using its configuration. Detailed full-tensor GPU profiling adds
synchronization. Coherent GPU diagnostics reuse existing counters and add
worker-placement records without extra CUDA synchronization. Use `timing`
for ordinary phase measurements and silent calls with progress disabled for
benchmarks.

## Error codes

`Error::code()` identifies an operational failure without parsing `what()`:

| `ErrorCode` | Value | Meaning |
| --- | --- | --- |
| `backend_unavailable` | 1 | The requested backend cannot be selected. |
| `backend_failure` | 2 | Backend execution failed. |
| `invalid_tensor_file` | 3 | A serialized tensor is invalid. |
| `cache_failure` | 4 | Required cache reuse failed. |
| `resource_limit` | 5 | A configured resource limit was exceeded. |
| `numerical_failure` | 6 | Numerical computation failed. |
| `internal_failure` | 7 | An internal invariant failed. |
| `unsupported_operation` | 8 | The requested operation is unsupported. |

These integer values are stable. Diagnostic wording is not a programmatic interface.

The standard categories retain their ordinary meanings:

- `std::invalid_argument`: invalid labels, angles, options, or enum values;
- `std::out_of_range`: an invalid tensor index or visible-device ordinal;
- `std::overflow_error` / `std::length_error`: a representation cannot encode
  the requested extent;
- `std::bad_alloc`: host allocation failure;
- `std::filesystem::filesystem_error`: open, mapping, locking, publication, or
  another filesystem failure.

This division lets a thin Python, Julia, or Wolfram Language wrapper translate
errors deterministically without maintaining a list of English substrings.
As an intentional 0.14 API refinement, operations that formerly used
`std::logic_error` for an unsupported request—notably asking a canonical-dense
`FullTensor` for a reassociated layout—now use
`ErrorCode::unsupported_operation`. Applications that caught that specific
`std::logic_error` should catch `sl2cfoamxx::Error` (or `std::exception`)
instead.

## Shared-library ABI visibility

Linux and macOS builds compile with hidden visibility by default. Public
out-of-line declarations carry `SL2CFOAMXX_API`, defined in
`<sl2cfoamxx/export.hpp>`, so only the supported C++ surface is exported from
the shared library. The macro expands to the GCC/Clang
`visibility("default")` attribute; it does not introduce another namespace,
change source spelling, or require an import/export distinction on these
platforms. Private recurrence, contraction, CUDA, diagnostics, and planning
symbols are not a linkable interface.

`FullTensor`, `Error`, and the public free functions are exported directly.
The few `sl2cfoamxx::detail` symbols visible in the dynamic table are deliberate
out-of-line storage plumbing required by arbitrary client instantiations of the
public `Tensor<T, Rank>` template; they are not user-facing entry points.
An installed-package consumer can instantiate tensor specializations that
are not instantiated by the library itself.

Public signatures intentionally use C++ standard-library types, so native
applications must use a GNU/libstdc++ toolchain that is ABI-compatible with the
one used to build the library. On macOS this means Homebrew GCC/libstdc++, not
AppleClang/libc++; the build already enforces GNU C++ for the library itself.

## Independent one-file consumer on macOS

`sl2cfoam++` is distributed as a shared library whose stable technical and
link name is `sl2cfoamxx`. Its BLAS and recurrence implementation details
remain private, so an unrelated directory containing only
`simple_example.cpp` names just `sl2cfoamxx`:

```sh
SL2CFOAMXX_ROOT=/absolute/path/to/sl2cfoamxx
SL2CFOAMXX_BUILD="${SL2CFOAMXX_ROOT}/build" # use build-gcc if that is your build directory

/opt/homebrew/bin/g++-16 \
  -std=c++20 -O2 \
  -I"${SL2CFOAMXX_ROOT}/include" \
  simple_example.cpp \
  -L"${SL2CFOAMXX_BUILD}" \
  -Wl,-rpath,"${SL2CFOAMXX_BUILD}" \
  -lsl2cfoamxx \
  -o simple_example

./simple_example
```

Replace `g++-16` with the installed Homebrew GCC version when necessary.
The `rpath` tells the executable where to find the build-tree `.dylib` at run
time; it is not another link dependency. For a relocatable application, install
the library to a prefix and use that prefix's `include` and `lib` directories:

```sh
cmake --install "${SL2CFOAMXX_BUILD}" --prefix /absolute/install/prefix
```

The install also provides a CMake package target named
`sl2cfoamxx::sl2cfoamxx` for applications that later choose to use
`find_package(sl2cfoamxx CONFIG)`.
