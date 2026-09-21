<!-- SPDX-FileCopyrightText: 2026 Jeremy Martinon -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Julia API

`SL2CfoamXX` wraps the public C++ library. Scientific calculations, CPU/GPU
dispatch, configuration, progress rendering, and tensor files use the same
implementation as the C++ and Python APIs. The package uses Julia's built-in
foreign-function interface through a private C-compatible bridge. CxxWrap,
Julia development headers, and external Julia packages are unnecessary.

## Build and run

Install Julia 1.10 or newer and the [native build dependencies](building.md).
From the repository root:

```sh
cmake -S . -B build-julia \
  -DBUILD_TESTING=OFF \
  -DSL2CFOAMXX_BUILD_JULIA=ON \
  -DJulia_EXECUTABLE=/path/to/julia
cmake --build build-julia --parallel
JULIA_LOAD_PATH="$PWD/build-julia/julia:@stdlib" \
  julia examples/julia_example.jl --contract
```

Replace compiler/interpreter paths with your installed executables. You can
omit `Julia_EXECUTABLE` if `julia` is on `PATH`. Python and Julia build options
are independent; both may be enabled in one build.

A `.jl` source file is run by `julia`; you do not manually link each script.
`using SL2CfoamXX` finds the package on `JULIA_LOAD_PATH` and loads its compiled
bridge. The bridge's runtime search path locates the C++ shared library.
`@stdlib` keeps Julia's standard libraries available. The absolute build path
also works when running a script from another directory.

For GPU, enable the [CUDA CMake options](building.md#nvidia-gpus), then run:

```sh
JULIA_LOAD_PATH="$PWD/build-julia/julia:@stdlib" \
  julia examples/julia_example.jl --backend GPU --contract
```

The single [Julia example](../examples/julia_example.jl) covers configuration,
all five computation functions, custom coefficient vectors, tensors, slicing,
mutation, persistence, closure, callbacks, and optional array contractions. Defaults are `--backend
CPU --two-j 1 --Dl 2 --threads 2`; `--contract` enables contractions. Add
`--two-j 12 --Dl 6` for a longer progress display. `--help` lists options.

## Install or use in a Julia project

Install into a prefix of your choice:

```sh
cmake --install build-julia --prefix /path/to/sl2cfoamxx-install
JULIA_LOAD_PATH="/path/to/sl2cfoamxx-install/share/julia:@stdlib" \
  julia /path/to/script.jl
```

The default Julia package parent is `<prefix>/share/julia`, controlled by
`SL2CFOAMXX_JULIA_INSTALL_DIR`. Keep the installed directory tree together
when moving it; the package and library use relative runtime lookup paths.

Alternatively, add the **built or installed** package to your own Julia project:

```julia
using Pkg
Pkg.activate("/path/to/my-project")
Pkg.develop(path="/path/to/sl2cfoamxx-install/share/julia/SL2CfoamXX")
using SL2CfoamXX
```

Then `julia --project=/path/to/my-project script.jl` needs no `JULIA_LOAD_PATH`.
This is a local package, not a registered package installed by name. When
developing the source package directly, set `SL2CFOAMXX_JULIA_LIBRARY` to the
absolute compiled `libsl2cfoamxx_julia.so` (Linux) or `.dylib` (macOS) path
before starting Julia. The normal build/install workflow needs no override.

## Computation and defaults

```julia
using SL2CfoamXX
const sf = SL2CfoamXX

config = sf.default_config()
config.backend = sf.Backend.CPU
config.cpu_threads = 2
config.progress_bar = true
config.verbosity = sf.Verbosity.summary
sf.configure(config)

spins = fill(2, 10)                 # doubled labels: physical j=1
intertwiners = zeros(Int, 5)
tensor = sf.compute_full_tensor(spins, 2, 1.2)
amplitude = sf.compute_amplitude(spins, intertwiners, 2, 1.2)
indices = sf.indices_from_intertwiners(spins, intertwiners)
@assert isapprox(tensor[indices...], amplitude; rtol=5e-10)
```

The [configuration tables](configuration.md#configuration-defaults) give every
default, including low-level options. Julia uses `Backend.CPU` style enum
names, `true`/`false`, and `nothing` for an empty optional path. Nested options
have the same names as Python: `config.tensors`, `config.low_level`, and
`config.coherent_amplitude`. `LibraryConfig()` is a direct default constructor;
prefer `default_config()` for the build-dependent backend and cache directory.
`current_config()` returns an independent snapshot. Call `configure()` to
apply changes; configuring a copy does not change calls already in progress.

| Function | Result / optional arguments |
| --- | --- |
| `compute_amplitude(spins, intertwiners, Dl, gamma; cache_policy=AmplitudeCachePolicy.disabled)` | `Float64`; fixed amplitudes run on CPU. |
| `compute_full_tensor(spins, Dl, gamma; backend=nothing, progress=nothing)` | `FullTensor`; `nothing` uses the configured backend. |
| `compute_coherent_amplitude(spins, Dl, gamma, angles; backend=nothing, progress=nothing)` | `ComplexF64`; CPU or GPU. |
| `compute_coherent_amplitude(spins, Dl, gamma, C1, C2, C3, C4, C5; backend=nothing, progress=nothing)` | `ComplexF64`; contract user-supplied coefficients on CPU or GPU. |
| `compute_coherent_coefficients(spins4, angles4)` | Independent `Vector{ComplexF64}` for one node. |
| `check_closure(spins, angles; absolute_tolerance=1e-12, relative_tolerance=1e-12)` | Five node reports and `all_closed`. |
| `indices_from_intertwiners(spins, intertwiners)` | Five **one-based** Julia indices. |

Spins/intertwiners are nonnegative doubled integer labels in the
[shared scientific order](scientific-conventions.md). Julia
labels do not change physical convention. **Array positions do:** the first
tensor entry is `tensor[1,1,1,1,1]`, and angle rows run from 1 through 20.
Angles are a `20 × 2` matrix (theta, phi in radians), or a sequence of 20 pairs;
the coefficient function takes four labels and four pairs. Noncontiguous
matrix views are accepted. Node recoupling permutations are demonstrated in
the example; using the wrong node order changes the physical contraction.

The built-in `sf.lorentzian_boundary_angles` and `sf.euclidean_boundary_angles`
presets are immutable tuples of twenty `(theta, phi)` tuples. They use radians
and the same canonical row order as ordinary angle inputs. Their values come
from the native C++ presets when the package is loaded, so they work with
precompiled and relocated installations. Pass them directly:

```julia
angles = sf.euclidean_boundary_angles
preset_spins = fill(1, 10)
@assert sf.check_closure(preset_spins, angles).all_closed
preset_amplitude = sf.compute_coherent_amplitude(preset_spins, 2, 1.2, angles)

lorentzian_spins = (10,10,10,10,4,4,4,4,4,4)
@assert sf.check_closure(lorentzian_spins, sf.lorentzian_boundary_angles).all_closed

# Create a mutable 20 × 2 copy if you want to adjust the directions.
editable_angles = [angles[row][column] for row in 1:20, column in 1:2]
```

The Euclidean preset closes for equal spins; the Lorentzian preset closes for
the doubled-spin proportions shown above. These presets do not change the
spin or angle conventions. The spellings `lorentzian_bounday_angles` and
`euclidean_bounday_angles` are aliases for the same tuples.

To supply the boundary coefficients directly, pass five vectors (or tuples)
of real or complex numbers. Each is copied to `Vector{ComplexF64}`, including
noncontiguous views. Its length must equal the corresponding tensor axis;
entries follow ascending admissible doubled intertwiners in that node's
recoupling basis. All real and imaginary components must be finite. The
contraction is `sum(T[i1,i2,i3,i4,i5] * C1[i1] * C2[i2] * C3[i3] * C4[i4] * C5[i5])`,
with no conjugation or normalization. Zero vectors are valid. The configured
pruning policy still applies; set `config.coherent_amplitude.pruning_ratio=0.0`
when comparing against an unpruned full-tensor contraction.

```julia
# For the tensor computed above, choose any coefficients of matching lengths.
C1, C2, C3, C4, C5 = [ones(ComplexF64, n) for n in size(tensor)]
C1[1] = 2im
weighted = sf.compute_coherent_amplitude(spins, 2, 1.2, C1, C2, C3, C4, C5;
                                         backend=sf.Backend.CPU)
```

You can also reuse vectors from `compute_coherent_coefficients`; the
[example](../examples/julia_example.jl) shows each node's spin and angle
permutations. Supplying vectors skips coefficient construction and the
angle-based closure check, even when `config.coherent_amplitude.check_closure`
is enabled. Use `check_closure` separately when you also have the normals.
Both amplitude forms support a positional backend and the same progress
callback options.

`version()`, `build_info()`, `backend_available(Backend.GPU)`,
`cuda_device_count()`, and `cuda_device_info(0)` inspect the loaded library.
CUDA device ordinals remain zero-based, matching CUDA. A requested unavailable
GPU raises an error; it never silently uses the CPU.

## Native Julia tensors and contractions

`FullTensor <: AbstractArray{Float64,5}`. Canonical indexing, linear indexing,
`view`, `reshape`, broadcasting, slicing, and reductions work directly, even
when the C++ payload is symmetry compressed. `size(t)` and `length(t)` count
logical entries; `t.stored_size` counts actual stored scalars.

```julia
slice = tensor[1, :, :, :, :]
scaled = 2 .* tensor
A = Array(tensor)                   # independent canonical Julia-owned copy

# Contract T[a,b,c,d,e] U[f,b,c,d,e] -> M[a,f].
using LinearAlgebra
B = Array(copy(tensor))             # another computed tensor can be used
rows = reshape(A, size(A,1), :)
other_rows = reshape(B, size(B,1), :)
M = rows * transpose(other_rows)
```

Use `Array(t)` when a third-party tensor package needs an actual dense `Array`
or when performing many dense operations. This expands compact storage only
once. Packages accepting `AbstractArray` may also use `t` directly. No einsum
dependency is required for the contractions shown in the example. These Julia
array operations run on CPU, including for tensors computed by the GPU backend.

`copy(t)`/`deepcopy(t)` creates independent native storage in the same physical
representation; `dense_copy(t)` creates an independent canonical dense native
tensor. `view(t, :, :, :, :, :)` shares native storage without copying. Views
and their reshaped/sliced descendants retain the owner through garbage
collection. `physical_view(t)` (alias `stored_view`) exposes the physical
payload as an owner-retaining vector; compact/reassociated values are **not**
necessarily in canonical tensor order. `Array(t)` always has independent
ownership, unlike Python's potentially shared `np.asarray(t)` conversion.

Writable tensors support indexed assignment and `fill!`. Read-only mappings
reject mutation through the tensor and its views before touching protected
memory. Callers must synchronize concurrent reads/writes to the same tensor.

## Persistence and metadata

```julia
sf.save_tensor(tensor, "vertex.sl2cxx")
loaded = sf.load_tensor("vertex.sl2cxx")       # owned writable copy
mapped = sf.map_tensor("vertex.sl2cxx")        # read-only by default
editable = sf.map_tensor("vertex.sl2cxx"; access=sf.TensorAccess.read_write)
editable[1,1,1,1,1] = 0.0
flush(editable)

info = sf.metadata(tensor)                    # scientific inputs and storage
file_info = sf.metadata("vertex.sl2cxx")
```

Files are compatible with C++ and Python. `FullTensor(shape)` creates an owned,
zero-filled tensor; `create_mapped(path, shape)` creates a writable mapped
tensor. Both accept `init=TensorInit.zero` and
`layout=TensorLayout.canonical_dense_column_major` defaults. Use
`convert_storage(mapped, TensorStorage.on_disk_dense, path;
overwrite_existing=false)` for a lossless file conversion. Storage conversion
retains validated physics metadata.

Properties include `shape`, `storage`, `layout`, `reassociated_layout`,
`stored_size`, `logical_size`, `stored_bytes`, `dense_bytes`, `path`,
`is_mapped`, `is_read_only`, and `is_symmetry_compressed`. `metadata(t)` also
reports spins, cutoff, gamma, and simplicity convention when available.
`format_tensor(t, FullTensorPrintOptions())` uses the bounded native printer:
defaults are 4096 values, precision 17, and included metadata. Its diagnostic
coordinate labels follow the native zero-based convention; Julia indexing
remains one-based. `tensor_tag(t)` returns a copy of the 1024-byte native tag.

## Progress, tasks, and errors

`config.progress_bar=true` enables the existing C++ progress display for both
CPU and GPU coherent/full-tensor calls. A terminal gets a live bar; redirected
stderr gets progress records. Set `verbosity=Verbosity.summary` to keep the
completed terminal line. The fixed-intertwiner function has no progress bar.
Cache hits may have no contraction events.

```julia
events = sf.ProgressEvent[]
tensor = sf.compute_full_tensor(spins, 2, 1.2;
    backend=sf.Backend.GPU, progress=event -> push!(events, event))
```

As in Python, an explicit `backend` is required when supplying `progress`.
Each event has `operation`, `phase`, `backend`, `status`, `attempt`,
`completed`, `total`, `elapsed_seconds`, `eta_available`, and `eta_seconds`.
The callback runs on the calling Julia task. The C++ callback only queues
events; it never calls Julia from an OpenMP/CUDA worker. Running updates may
be coalesced; lifecycle events remain ordered. Callbacks supplement the native
bar and do not draw a second bar automatically.

Heavy calls use Julia's thread-call pool, leaving Julia tasks responsive even
with `JULIA_NUM_THREADS=1`. Without a callback, the wrapper waits for completion
without a polling loop. Callback failures are rethrown after native completion
and resource cleanup; they do not cancel the numerical calculation. SIGINT is
deferred during native calls to protect memory until the calculation finishes.
Avoid recursively launching computations from callbacks; simultaneous native
calls are bounded by Julia's thread-call pool. Prefer parallel calls with
explicit, appropriate `cpu_threads` to prevent host oversubscription.

Invalid inputs raise `ArgumentError`; overflow and indexing failures use
`OverflowError` and `BoundsError`. Operational native failures raise
`SL2CfoamXX.Error` with a stable `code` (for example,
`ErrorCode.backend_unavailable` or `ErrorCode.invalid_tensor_file`). File-system
errors preserve native path information in `FilesystemError`. Exceptions
cannot cross the C ABI and errors belong to the individual call.

The architecture follows Julia's official
[C interface and thread-call documentation](https://docs.julialang.org/en/v1/manual/calling-c-and-fortran-code/).

## Validation

Build with `BUILD_TESTING=ON` and `SL2CFOAMXX_BUILD_JULIA=ON`, then:

```sh
ctest --test-dir build-julia -L julia --output-on-failure
```

CPU tests check preset values, aliases, closure, a preset amplitude reference,
independent coefficient references, fixed/tensor/coherent
agreement, arbitrary complex coefficient contractions, zero and basis vectors,
both simplicity conventions, unequal spins, lifted shells, array ownership,
storage, invalid inputs, callback failures, and native progress.
The GPU test runs those checks on a real device and explicitly skips when
CUDA is unavailable. It does not substitute a CPU or mock execution.
