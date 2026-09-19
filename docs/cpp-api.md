<!-- SPDX-FileCopyrightText: 2026 Jeremy Martinon -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# C++ API

Include `<sl2cfoamxx.hpp>` and link `sl2cfoamxx::sl2cfoamxx`. The library
computes fixed-intertwiner amplitudes, coherent amplitudes, and full vertex
tensors; the same numerical implementation serves all language wrappers.

[Quick start](#quick-start) · [Scientific inputs](#scientific-input-validation) ·
[Coherent boundaries](#coherent-boundaries) · [Tensors](#full-tensor-representations) ·
[Configuration](#library-configuration) · [Errors](#errors)

Detailed storage, cache, callback, and ABI contracts live in the
[C++ reference](cpp-reference.md). Runnable examples are
[basic calculations](../examples/simple_example.cpp) and
[coherent boundaries](../examples/coherent_boundary_example.cpp).

## Quick start

First [build and install the library](building.md). Save this independent
application as `main.cpp`:

```cpp
#include <sl2cfoamxx.hpp>
#include <iostream>

int main()
{
    namespace sf = sl2cfoamxx;
    auto config = sf::default_config();
    config.backend = sf::Backend::CPU;
    sf::configure(config);

    const sf::DSpins spins{2,2,2,2,2,2,2,2,2,2}; // physical j = 1
    const sf::DIntertwiners intertwiners{0,0,0,0,0};
    const auto amplitude = sf::compute_amplitude(spins, intertwiners, 0, 1.2);
    const auto tensor = sf::compute_full_tensor(spins, 0, 1.2);
    const auto indices = sf::indices_from_intertwiners(spins, intertwiners);
    std::cout << amplitude << ' ' << tensor.at(indices) << '\n';
}
```

The two values should agree to numerical precision. By default, full tensors
are stored in the configured cache directory and compatible files are reused.

Place this `CMakeLists.txt` beside `main.cpp`:

```cmake
cmake_minimum_required(VERSION 3.22)
project(my_amplitude LANGUAGES CXX)
find_package(sl2cfoamxx CONFIG REQUIRED)
add_executable(my_amplitude main.cpp)
target_link_libraries(my_amplitude PRIVATE sl2cfoamxx::sl2cfoamxx)
```

Build and run with the same GNU C++ toolchain used for the library:

```sh
cmake -S . -B build -DCMAKE_CXX_COMPILER=/path/to/g++ \
  -DCMAKE_PREFIX_PATH=/your/install/prefix
cmake --build build --parallel
./build/my_amplitude
```

Use an **installation prefix**, not the library's source build directory.
The exported target supplies the C++20 requirement; numerical dependencies
remain private. A [direct compiler command](cpp-reference.md#independent-one-file-consumer-on-macos)
is also available for a one-file macOS application. None of these examples
requires enabling the library's tests.

## Scientific input validation

Every API takes nonnegative **doubled integer labels**: `1` means physical
spin `1/2`, and `2` means spin `1`. Both odd and even labels are supported.
The array positions are:

| Type | Order |
| --- | --- |
| `DSpins` | `2 × {j12,j13,j14,j15,j23,j24,j25,j34,j35,j45}` |
| `DIntertwiners` | `2 × {i1,i2,i3,i4,i5}` |
| `BoundaryAngles` | Twenty `{theta,phi}` pairs in radians; see the [normal order](scientific-conventions.md). |

`Dl` is a nonnegative shell cutoff within the supported range. `gamma` must
be finite and nonnegative. Every amplitude boundary must admit an SU(2)
intertwiner at each node; fixed intertwiners must also have the correct range
and parity. Invalid inputs are rejected before opening a cache or allocating
a numerical workspace.

Tensor coordinates are **zero-based positions**, not doubled intertwiner
labels. Convert labels using `indices_from_intertwiners()`; for example:

```cpp
const sl2cfoamxx::DSpins spins{2,4,6,8,4,6,8,10,12,14};
const sl2cfoamxx::DIntertwiners intertwiners{2,8,6,8,10};
const auto indices = sl2cfoamxx::indices_from_intertwiners(spins, intertwiners);
// indices == {0,1,2,0,4}
```

The historical names `Spins`, `Intertwiners`, and `TetrahedronSpins` remain
aliases. New code should use `DSpins`, `DIntertwiners`, and
`TetrahedronDSpins` to make doubled units explicit.

## Choose a computation

| Call | Result | Backend |
| --- | --- | --- |
| `compute_amplitude(spins, intertwiners, Dl, gamma)` | One real fixed-intertwiner amplitude | CPU |
| `compute_coherent_amplitude(spins, Dl, gamma, angles)` | One complex coherent amplitude | CPU/GPU |
| `compute_coherent_amplitude(spins, Dl, gamma, C1, C2, C3, C4, C5)` | One complex amplitude from supplied coefficients | CPU/GPU |
| `compute_full_tensor(spins, Dl, gamma)` | A rank-five `FullTensor` | CPU/GPU |
| `compute_coherent_coefficients(spins4, angles4)` | Complex coefficients for one tetrahedron | CPU |
| `check_closure(spins, angles)` | Five node reports and `all_closed` | CPU |

Coherent and full-tensor calls use the configured backend unless you pass an
explicit final `Backend::CPU` or `Backend::GPU`. A requested GPU must be
available; there is no silent CPU fallback. For example:

```cpp
auto tensor = sl2cfoamxx::compute_full_tensor(
    spins, Dl, gamma, sl2cfoamxx::Backend::GPU);
```

A fixed-amplitude call always computes directly unless you explicitly request
`AmplitudeCachePolicy::prefer_full_tensor` as its fifth argument. That policy
first looks for a matching persisted full tensor, then falls back to direct
CPU computation on a miss or invalid file. To use an existing in-memory
tensor, read `tensor.at(indices)` directly.

## Coherent boundaries

Fill all twenty rows of `BoundaryAngles` using the
[canonical normal order](scientific-conventions.md), then call:

```cpp
const auto closure = sl2cfoamxx::check_closure(spins, angles);
const auto amplitude = sl2cfoamxx::compute_coherent_amplitude(
    spins, Dl, gamma, angles);
```

Alternatively, supply five `CoherentCoefficients` vectors
(`std::vector<std::complex<double>>`) in the canonical node bases:

```cpp
const auto amplitude = sl2cfoamxx::compute_coherent_amplitude(
    spins, Dl, gamma, C1, C2, C3, C4, C5);
```

Each vector must contain exactly one finite coefficient per admissible
intertwiner, in ascending doubled label order, matching its full-tensor axis.
The contraction multiplies the supplied coefficients without conjugating or
normalizing them; zero vectors are accepted. Geometric closure checking is
skipped for this overload because it has no normals. Both forms accept a
trailing backend and, after that, an optional per-call progress callback.
See the [coefficient input contract](cpp-reference.md#caller-supplied-coefficients)
for the bases, ordering, and contraction formula.

`check_closure()` returns data without printing and accepts nonnegative spin
assignments even when they do not admit intertwiners. Each node passes when
`residual_norm <= 1e-12 + 1e-12 * scale`; vectors and scales use physical spins.
You can supply different absolute and relative tolerances.

The [coherent example](../examples/coherent_boundary_example.cpp) contains
complete Lorentzian and Euclidean boundary data. The
[coherent reference](cpp-reference.md#coherent-boundaries) gives the five
recoupling bases, one-tetrahedron coefficient order, and closure policy.
When comparing a full coefficient contraction with a direct coherent call,
set `config.coherent_amplitude.pruning_ratio = 0`; the default relative C1
pruning is `1e-8`, an approximation without a rigorous final-error bound.

## Full-tensor representations

`tensor.at(i1,i2,i3,i4,i5)` gives checked access in canonical axis order.
`tensor(i1,i2,i3,i4,i5)` uses the same order without bounds checks.
A **canonical dense** payload is column-major: `i1` varies fastest.

| `config.tensors.storage` | Backing | Representation |
| --- | --- | --- |
| `TensorStorage::on_disk_sym` **(default)** | Writable mapped file | Symmetry-aware |
| `TensorStorage::on_disk_dense` | Writable mapped file | Canonical dense |
| `TensorStorage::in_memory_sym` | Owned memory | Symmetry-aware |
| `TensorStorage::in_memory_dense` | Owned memory | Canonical dense |

Symmetry-aware modes use reflection compression when the spins permit it and
dense storage otherwise. Logical indexed access works in every mode, but
`data()`, iteration, `size()`, and flat access expose the **physical payload**.
For compressed storage, that payload is not a canonical rank-five buffer.
Use an explicit dense copy when passing data to another numerical routine:

```cpp
auto dense = tensor.dense_copy(); // independent, owned, canonical dense
const double* values = dense.data();
const auto shape = dense.shape();
```

`logical_size()` counts canonical entries; `stored_size()` counts physical
values. `metadata()` reports representation, backing, and optional scientific
inputs. `print()` displays up to 4096 canonical entries by default.
See the [storage reference](cpp-reference.md#tensor-storage) for raw-buffer
rules, reflection aliasing, metadata, printing, and lossless conversion.

## Tensor persistence

The `.sl2cxx` format is shared by all four APIs:

```cpp
tensor.save("vertex.sl2cxx");
auto loaded = sl2cfoamxx::FullTensor::load("vertex.sl2cxx"); // owned copy
const auto mapped = sl2cfoamxx::FullTensor::map("vertex.sl2cxx"); // read-only

auto writable = sl2cfoamxx::FullTensor::map(
    "vertex.sl2cxx", sl2cfoamxx::TensorAccess::read_write);
writable.at(0, 0, 0, 0, 0) = 0.0;
writable.flush();
```

Saving, loading, and mapping preserve dense/compressed representation.
`load()` and copying a mapped tensor produce independent owned storage;
`map()` keeps the file as backing. Access a read-only mapping through a
`const` object; requesting mutable access throws even if you only intend to
read the returned reference.

Computed on-disk tensors are writable: modifying one changes its cache file.
Reuse checks all scientific and representation metadata, including exact gamma
bits. Set `reuse_existing=false` to recompute and atomically replace an output.
See [cache paths and reuse](cpp-reference.md#cache-paths-and-reuse) for naming,
exact output paths, collision handling, and concurrent writers.

## Library configuration

Start from `default_config()`, edit the fields you need, then apply them:

```cpp
auto config = sl2cfoamxx::default_config();
config.backend = sl2cfoamxx::Backend::CPU;
config.cpu_threads = 8; // default 0: use the caller's OpenMP ceiling
config.tensors.directory = "./data";
config.tensors.storage = sl2cfoamxx::TensorStorage::in_memory_dense;
config.coherent_amplitude.pruning_ratio = 0;
config.verbosity = sl2cfoamxx::Verbosity::summary;
sl2cfoamxx::configure(config);
```

Without `configure()`, the first computation uses defaults: GPU in a CUDA
build, CPU otherwise; `rho=gamma*j`; progress enabled; symmetry-aware on-disk
tensors. `current_config()` returns a copy. Active calculations retain their
configuration snapshot. See [all fields and defaults](configuration.md),
including `gamma_j_p1`, memory limits, logging, and expert controls.

`backend_available(Backend::GPU)` is a non-throwing availability probe.
`version()`, `build_info()`, `cuda_device_count()`, and `cuda_device_info(ordinal)`
report the loaded library and visible hardware. Device ordinals follow
`CUDA_VISIBLE_DEVICES`; device queries retain runtime errors as exceptions.
Availability does not guarantee that a particular calculation fits in memory.
Coherent and full-tensor GPU calls distribute contraction work among visible
devices. Host booster preparation also uses `cpu_threads`; reserve sufficient
CPU resources when running on several GPUs. See the
[NUMA guidance](building.md#numa-hosts).

## Errors

Catch errors around each computation when your application should skip a
failed calculation and continue:

```cpp
try {
    auto tensor = sl2cfoamxx::compute_full_tensor(spins, Dl, gamma);
    tensor.print();
} catch (const std::invalid_argument& error) {
    std::cerr << "Invalid input: " << error.what() << '\n';
} catch (const sl2cfoamxx::Error& error) {
    std::cerr << sl2cfoamxx::to_string(error.code())
              << ": " << error.what() << '\n';
}
```

`sl2cfoamxx::Error` derives from `std::runtime_error` and supplies a stable
`ErrorCode`. Bounds, overflow, allocation, and filesystem errors retain their
standard C++ exception types. An uncaught exception terminates the process;
see the [complete error reference](cpp-reference.md#error-codes).

For validation and deployment, see [build tests](building.md#tests) and
[regression testing](testing.md).
