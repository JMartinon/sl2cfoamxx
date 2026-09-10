<!-- SPDX-FileCopyrightText: 2026 Jeremy Martinon -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Building and installation

[Back to the README](../README.md) · [C++ API](cpp-api.md) ·
[Configuration defaults](configuration.md)

## Requirements

- Linux or macOS, CMake 3.22+, and GNU C++ with C++20 and OpenMP support.
  GCC 13 and newer are the tested toolchains; AppleClang and serial host builds
  are unsupported.
- A BLAS implementation with 32-bit-integer CBLAS: Accelerate on macOS;
  OpenBLAS, BLIS/AOCL, or MKL on Linux. Fortran-only BLAS is insufficient.
- For NVIDIA GPUs: a compatible CUDA toolkit, cuBLAS, and external cuTENSOR.
  The CUDA toolkit must support the selected GNU host compiler.
- Optional APIs: Python 3.10+ with development headers, Julia 1.10+, or a
  licensed Wolfram installation with the LibraryLink SDK. See each API guide
  for its setup. NumPy is needed for Python array operations and its tests.

C++20 is needed for facilities such as `std::span` and `std::bit_cast`; C++23
is not required. CMake 3.22 is the minimum because its BLAS finder can select
the integer interface explicitly.

## Build and install

From the repository root, choose an install prefix:

```sh
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/path/to/sl2cfoamxx-install
cmake --build build --parallel
cmake --install build
```

The default build type is **Release** for single-configuration generators.
Use `-DCMAKE_BUILD_TYPE=Debug` when needed.

On macOS, install GNU C++ with `brew install gcc`. CMake searches the active
Homebrew prefix and the usual `/opt/homebrew` and `/usr/local` installations,
including versioned `gcc@...` formulas, and selects a verified GNU compiler.
It prefers the highest installed GCC major version within the first matching
prefix. Apple's `/usr/bin/g++` is Clang and is not selected automatically.

On Linux, CMake uses its normal compiler discovery; the result must be GNU C++.
On either platform, an explicit `-DCMAKE_CXX_COMPILER=/path/to/g++`, a `CXX`
environment choice or a CMake toolchain takes precedence over automatic
selection. Project initialization hooks also keep control of compiler choice.
Existing build directories keep their cached compiler.

Tests and language wrappers are disabled by default. Use a fresh build directory
when changing compiler, BLAS provider, CUDA mode, or ABI-affecting settings.
CMake caches options in the build directory; an existing test build therefore
needs `-DBUILD_TESTING=OFF` to disable tests.

The installation exports the `sl2cfoamxx::sl2cfoamxx` CMake target. Follow the
[C++ quick start](cpp-api.md#quick-start) to compile and link your own application.
Python, Julia, and Mathematica load their compiled wrappers automatically;
individual scripts have no link step. Instructions:
[Python](python.md), [Julia](julia.md), [Mathematica](mathematica.md).

## NVIDIA GPUs

For an A100 (`sm_80`), use a separate build:

```sh
cmake -S . -B build-cuda \
  -DCMAKE_CXX_COMPILER=/path/to/g++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON \
  -DCUDA_ARCHITECTURES=80 \
  -DCUTENSOR_ROOT=/path/to/cutensor \
  -DCMAKE_INSTALL_PREFIX=/path/to/sl2cfoamxx-install
cmake --build build-cuda --parallel
cmake --install build-cuda
```

`CUTENSOR_ROOT` points to a prefix containing `include/cutensor.h` and its
library directory. For bundles with versioned directories, CMake searches
`lib/<CUDA toolkit major>` as well as `lib` and `lib64`. An explicit or cached
`CUTENSOR_LIBRARY` takes precedence; use a fresh build directory when changing
toolkits. Set the CUDA architecture list to the GPUs where the library
will run. A quoted list such as `'70;80'` builds for multiple architectures
when the toolkit and cuTENSOR support them. The selected GNU C++ compiler is
also nvcc's default host compiler.

For a V100 (`sm_70`), use a pre-CUDA-13 toolkit and cuTENSOR 2.2 or older;
CUDA 12.9 with cuTENSOR 2.2 is validated. The full-tensor backend uses FP64 WMMA
on `sm_80+` and FP64 SIMT on earlier architectures. 

## BLAS selection

Set `BLA_VENDOR` to choose the provider and `CMAKE_PREFIX_PATH` to locate a
non-system installation. The default is `Apple` on macOS and `All` elsewhere.
CMake checks both the library and a working LP64 CBLAS `dgemm` call.

**LP64 describes the BLAS integer interface, not floating-point precision.**
Matrix dimensions and leading dimensions use 32-bit integers; tensor values
remain 64-bit doubles and tensor allocation sizes use 64-bit addressing.
The current CBLAS wrapper requires LP64. An ILP64 library uses 64-bit BLAS
integers and is not supported by this build; substituting one can break the
calling convention. Supporting it would require a separate integer-interface
configuration and validation. CMake's
[`BLA_SIZEOF_INTEGER`](https://cmake.org/cmake/help/latest/module/FindBLAS.html)
is fixed to `4` accordingly.

ILP64 would extend the integer indexing range of individual BLAS calls; it
would not increase floating-point precision or inherently accelerate them.
The current contractions work in panels and guard their BLAS dimensions.
A large full tensor therefore does not by itself require ILP64 for every
panel operation. Add this interface when a measured workload reaches a
provider's LP64 limits, or an installation specifically requires ILP64.
See [Intel's interface comparison](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-0/using-the-ilp64-interface-vs-lp64-interface.html).

For AMD AOCL on Linux, use CMake 3.27 or newer:

```sh
cmake -S . -B build-aocl \
  -DCMAKE_CXX_COMPILER=/path/to/g++ \
  -DBLA_VENDOR=AOCL \
  -DCMAKE_PREFIX_PATH=/path/to/aocl
cmake --build build-aocl --parallel
```

`AOCL` selects single-threaded BLIS, suitable for the library's outer OpenMP
parallelism. `AOCL_mt` selects threaded BLIS when provider threading is
intentionally tuned. An installation with separate LP64 directories may need
`-DCMAKE_LIBRARY_PATH=/path/to/aocl/lib/LP64`. Other CMake selectors include
`OpenBLAS` and `Intel10_64lp_seq`; available names depend on the CMake version.

Keep the BLAS provider single-threaded for an initial scaling study. A threaded
AOCC-built BLIS may load LLVM `libomp` alongside GCC's `libgomp`; prefer a
single-threaded package or a matching GNU runtime. `BLIS_NUM_THREADS=1` disables
nested BLIS workers. Non-system BLAS libraries must remain visible to the
dynamic loader after installation.

## CMake options

These are defaults for a fresh build directory. Pass overrides as
`-DOPTION=value`; CMake retains them in that build directory's cache.

| Build switch | Default | Effect |
| --- | --- | --- |
| `ENABLE_CUDA` | `OFF` | Build the coherent/full-tensor GPU backends. |
| `ENABLE_NATIVE_ARCH` | `ON` | Optimize host code for the build machine's CPU. |
| `ENABLE_FAST_MATH` | `ON` | CUDA fast math only; host code keeps normal floating-point semantics. |
| `SL2CFOAMXX_BUILD_PYTHON` | `OFF` | Build the Python wrapper and fetch pinned nanobind. |
| `SL2CFOAMXX_BUILD_JULIA` | `OFF` | Build the Julia wrapper using the built-in C interface; no CxxWrap dependency. |
| `SL2CFOAMXX_BUILD_MATHEMATICA` | `OFF` | Build the LibraryLink wrapper using the installed Wolfram SDK. |
| `BUILD_TESTING` | `OFF` | Build/register public API integration tests; enabled wrappers add their integration tests. |

`ENABLE_NATIVE_ARCH=ON` adds `-march=native` to host C++ compilation. It may
improve CPU performance by using the build machine's instruction set, but the
binary may then fail on older or different CPUs. Set
`-DENABLE_NATIVE_ARCH=OFF` for portable installations. Existing build
directories retain their cached value; changing the project default does not
override it. GPU targets are selected separately by `CUDA_ARCHITECTURES`.

`ENABLE_FAST_MATH=ON` passes `--use_fast_math` to nvcc only. NVIDIA documents
relaxed single-precision division/square-root and subnormal handling, plus
fused multiply-add. The library's FP64 contractions stay double precision;
this flag does not rebuild cuBLAS/cuTENSOR or enable host C++ fast math.
The sensitive device 6j recurrence sources explicitly retain precise division
and square root, and preserve subnormal values, even when this option is ON.
Its performance and rounding effects depend on the kernels and inputs. Use
OFF to disable the general fast-math flag (FMA remains enabled), and compare
representative results before changing it for numerical work. See
[NVIDIA's option reference](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math).

| Build setting | Default | Meaning |
| --- | --- | --- |
| `CMAKE_BUILD_TYPE` | `Release` | Default for single-configuration generators when unset. |
| `BLA_VENDOR` | `Apple` on macOS; `All` elsewhere | BLAS selector; a supplied value or nonempty `BLA_VENDOR` environment variable takes precedence. macOS requires `Apple`. |
| `CUDA_ARCHITECTURES` | `80` | CUDA target architectures; use a quoted list such as `'70;80'` for multiple targets. |
| `CUTENSOR_ROOT` | Empty | External cuTENSOR prefix; discovery also considers the same environment variable and standard search paths. |
| `SL2CFOAMXX_NANOBIND_SOURCE_DIR` | Empty | Python builds fetch nanobind 2.13.0 into `third-party/nanobind`; set a matching source tree for offline builds. |
| `SL2CFOAMXX_PYTHON_INSTALL_DIR` | `<libdir>/pythonX.Y/site-packages` | Python installation path relative to the install prefix; `X.Y` is the selected interpreter version. |
| `SL2CFOAMXX_JULIA_INSTALL_DIR` | `<datadir>/julia` (normally `share/julia`) | Julia package parent relative to the install prefix. |
| `SL2CFOAMXX_MATHEMATICA_INSTALL_DIR` | `<datadir>/wolfram/Applications` | Mathematica package parent relative to the install prefix. |
| `Wolfram_ROOT` | Empty, then auto-discovered | Wolfram installation directory or macOS app bundle containing the LibraryLink SDK. |
| `SL2CFOAMXX_WOLFRAM_SYSTEM_ID` | Auto-detected on supported macOS/Linux targets | Target kernel's `$SystemID`, used for the native package resource directory. |

Standard CMake selectors are also available: `CMAKE_CXX_COMPILER` must resolve
to GNU C++ (Homebrew discovery supplies the macOS default); `Python_EXECUTABLE`,
`Julia_EXECUTABLE`, and `Wolfram_EXECUTABLE` are auto-detected unless supplied;
`CMAKE_CUDA_COMPILER` locates `nvcc`; and `CMAKE_CUDA_HOST_COMPILER` defaults to
the selected C++ compiler. `SL2CFOAMXX_CBLAS_INCLUDE_DIR`,
`CUTENSOR_INCLUDE_DIR`, `CUTENSOR_LIBRARY`, and `WolframLibrary_INCLUDE_DIR` are discovered dependency paths
that can be supplied explicitly. Installation uses the standard
`CMAKE_INSTALL_PREFIX` and `CMAKE_INSTALL_LIBDIR` settings.

## NUMA hosts

The library preserves the affinity and memory policy chosen by Slurm,
`numactl`, or the application. It does not pin threads itself. Large owned
buffer initialization, copies, storage conversion, and CPU tensor output use
static contiguous OpenMP ownership outside existing parallel regions. On
Linux's usual first-touch policy, this avoids placing a whole shared payload
on the master thread's node.

`cpu_threads=0` follows the caller's OpenMP ceiling, including subsequent
`omp_set_num_threads()` changes. A positive `cpu_threads` replaces the initial
`OMP_NUM_THREADS` setting, including for GPU booster preparation. It does not
reserve cores or change affinity. Automatic memory planning honors cgroup and
Slurm limits and the calling context's `Mems_allowed_list`. A restricted mask
caps capacity to permitted nodes and uses their available/free memory for
headroom. The master's CPU affinity alone does not describe the whole worker
team and is not used to infer memory placement.

These choices do not guarantee scaling across sockets. Start with a process
bound to one NUMA node, physical cores, `OMP_PLACES=cores`,
`OMP_PROC_BIND=close`, local memory, and single-threaded BLAS. Compare with
all-socket execution using explicit CPU and memory policies. Native multi-GPU
workers select devices safely but are not automatically pinned to nearby CPU
NUMA nodes.

## Tests

Tests are **OFF by default**. Opt in to the public API integration suite:

```sh
cmake -S . -B build-tests \
  -DCMAKE_CXX_COMPILER=/path/to/g++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON
cmake --build build-tests --parallel
ctest --test-dir build-tests --output-on-failure
```

The suite checks complete public workflows: numerical agreement, coherent
contractions, tensor storage and persistence, input errors, and progress.
Enable a language wrapper to include its integration suite; Python tests need
NumPy and Mathematica tests need a licensed kernel. In a CUDA build, the C++
workflow requires a working compatible GPU and fails if it is unavailable.
The optional language suites explicitly skip their GPU checks when no compatible
device is available.
Use `ctest -L cpu`, `ctest -L cuda`, or a language label (`python`, `julia`,
`mathematica`) to select a relevant group. There is no separate Wigner-oracle
build dependency.
