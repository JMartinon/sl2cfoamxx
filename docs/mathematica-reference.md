<!-- SPDX-FileCopyrightText: 2026 Jeremy Martinon -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Mathematica reference

The [Mathematica guide](mathematica.md) covers installation, first calculations,
and ordinary tensor algebra. This reference lists complete configuration
keys, tensor operations, and integration contracts.

[Configuration](#configuration-defaults) · [Tensor functions](#tensor-functions) ·
[Build settings](#build-and-loading-settings) · [Callbacks](#callback-contract) ·
[Errors](#error-details) · [Validation](#validation)

## Configuration defaults

`DefaultConfig[]` selects the build-dependent defaults. `CurrentConfig[]`
returns an independent snapshot. Edit either Association, then call
`Configure[config]` to apply it. Unknown or missing fields are rejected;
update the complete configuration instead of passing a partial Association.
An active calculation retains its configuration snapshot.

For nested settings, update the nested Association explicitly:

```wl
config = DefaultConfig[];
config["Tensors"] = Join[config["Tensors"], <|
    "Directory" -> "/absolute/path/to/my-cache",
    "Storage" -> "in_memory_dense",
    "ReuseExisting" -> False
|>];
config["CoherentAmplitude"] = Join[config["CoherentAmplitude"],
    <|"PruningRatio" -> 0.|>];
Configure[config];
```

| Top-level field | Default |
| --- | --- |
| `"Backend"` | `"GPU"` in a CUDA build; `"CPU"` otherwise. |
| `"ProgressBar"` | `True`. |
| `"SimplicityConstraint"` | `"gamma_j"`, meaning `rho = gamma j`; alternative `"gamma_j_p1"` means `rho = gamma (j+1)`. |
| `"CPUThreads"` | `0`: automatic. |
| `"MemoryBudgetBytes"` | `0`: automatic host-memory planning. GPU memory admission is separate. |
| `"Verbosity"` | `"silent"`; alternatives `"summary"`, `"timing"`, `"detailed"`. |
| `"PrintingStyle"` | `"UTF8"`, with locale-dependent ASCII fallback; `"ASCII"` forces ASCII. |

| `config["Tensors"]` field | Default |
| --- | --- |
| `"Directory"` | Nonempty `SL2CFOAMXX_CACHE_DIR`, otherwise `data_sl2cfoamxx` under the working directory. `Automatic` requests the same fallback. |
| `"Storage"` | `"on_disk_sym"`. Alternatives: `"in_memory_sym"`, `"in_memory_dense"`, `"on_disk_dense"`. |
| `"OutputPath"` | `None`: automatic cache tree. An explicit path requires an on-disk storage mode. |
| `"ReuseExisting"` | `True`: reuse only a tensor whose physics and representation metadata match. |
| `"MaximumPayloadSizeBytes"` | `107374182400` (100 GiB); `0` disables the limit on the physical payload. |

`Configure` creates the cache directory, including for in-memory calls.
Symmetry-aware storage compresses compatible boundaries and uses dense
storage when symmetry does not apply. Each mapped file also has a 64 KiB
metadata/alignment prefix.

| `config["CoherentAmplitude"]` field | Default |
| --- | --- |
| `"CheckClosure"` | `False`. Enabling it warns about failed closure and continues computation. Use `CheckClosure` to make your own decision. |
| `"PruningRatio"` | `1.*^-8`: relative C1 coefficient pruning. `0.` disables this pruning. |

Pruning does not alter the vectors returned by `ComputeCoherentCoefficients`
and is not a rigorous final-amplitude error bound. Disable it when comparing
a full coefficient contraction with a direct coherent calculation.

| `config["LowLevel"]` field | Default and allowed overrides |
| --- | --- |
| `"BoosterRhoTile"` | `0` (automatic); `4`, `8`, `16`. |
| `"FullTensorGPURowCapacity"` | `0` (automatic, currently `8192`); `1024..2^31-1`. Memory admission may reduce it. |
| `"CoherentGPUPipelineSlots"` | `0` (automatic, starts with `3`); `1`, `2`, `3`. |
| `"FullTensorGPUProducerWavesPerSM"` | `0` (automatic, currently `8`); `1..64`. |
| `"FullTensorCPUPBatch"` | `0` (automatic, currently `512`); `128`, `256`, `512`, `1024`, `2048`. |
| `"FullTensorCPUFinalTile"` | `"automatic"`; `"tile_4x4"`, `"tile_4x8"`, `"tile_8x4"`. |

Leave these expert controls at their defaults unless you are measuring a
specific workload. Native configuration validation applies to every override.

## Tensor functions

| Function | Behavior / defaults |
| --- | --- |
| `CreateTensor[shape]` | Owned rank-five container, zero-filled. `"Initialization" -> "zero"`; alternative `"uninitialized"`. `"Layout" -> "canonical_dense_column_major"`. |
| `CreateMappedTensor[path, shape]` | Writable file mapping with the same initialization/layout options. |
| `TensorFromArray[array]` | Independent native dense copy of a rectangular rank-five finite-real array. |
| `CloneTensor[t]` | Independent writable native copy retaining physical representation. |
| `DenseCopyTensor[t]` | Independent writable canonical dense native copy. |
| `LoadTensor[path]` | Read a file into owned, writable native memory. |
| `MapTensor[path]` | Map a file with `"Access" -> "read_only"`; alternative `"read_write"`. |
| `SaveTensor[t, path]`, `FlushTensor[t]` | Save the current representation or flush mapped writes. |
| `ConvertTensorStorage[t, mode, path]` | Lossless conversion of an existing mapped tensor into `"on_disk_dense"` or `"on_disk_sym"`; `"OverwriteExisting" -> False`. |
| `TensorMetadata[t]`, `TensorMetadata[path]` | Representation/backing and optional scientific inputs. |
| `TensorShape[t]`, `Dimensions[t]` | Five canonical logical dimensions. |
| `TensorValue[t, indices]`, `SetTensorValue[t, indices, value]` | One-based canonical access; mutation returns the original object. |
| `TensorTag[t]` | Independent `ByteArray` containing the 1024-byte native metadata tag. |
| `FormatTensor[t]` | Native diagnostic text; `"MaxValues" -> 4096`, `"Precision" -> 17`, `"IncludeMetadata" -> True`. Use `All` for unlimited values. Diagnostic coordinate labels are zero-based. |

Computed tensors record spins, cutoff, gamma, and simplicity convention in
`TensorMetadata[t]["Scientific"]`. A general container created from dimensions
or a Wolfram array has `"Scientific" -> None`. Empty optional paths and absent
reassociated layouts are also represented by `None`.

Disk storage conversion preserves the source. The destination is explicit
and is not overwritten by default. Dense-to-symmetry conversion requires
exact reflected-pair equality so it cannot silently discard information.
See the [C++ cache reference](cpp-reference.md#cache-paths-and-reuse) for
representation and cache matching rules.

`FullTensorObject` assignment shares the same native object. `CloneTensor`
and `DenseCopyTensor` are independent copies; `Normal` and `TensorArray`
produce independent Wolfram-owned arrays. Native storage is released when
no managed-expression references remain. Notebook output history can retain
references even after `Clear[tensor]`; see Wolfram's
[managed-expression lifetime model](https://reference.wolfram.com/language/ref/CreateManagedLibraryExpression.html).

## Build and loading settings

| Setting | Default / purpose |
| --- | --- |
| `SL2CFOAMXX_BUILD_MATHEMATICA` | `OFF`; enable the compiled bridge and package. |
| `Wolfram_ROOT` | Optional installed Wolfram prefix or macOS application bundle. |
| `WolframLibrary_INCLUDE_DIR` | Discovered SDK directory; can be supplied explicitly. |
| `Wolfram_EXECUTABLE` | Discovered `WolframKernel`/`MathKernel`; used for runtime tests. Building requires headers but does not start the kernel. |
| `SL2CFOAMXX_WOLFRAM_SYSTEM_ID` | Auto-detected for macOS ARM64, macOS x86-64, and Linux x86-64; must match the runtime's `$SystemID`. |
| `SL2CFOAMXX_MATHEMATICA_INSTALL_DIR` | `share/wolfram/Applications`, relative to the installation prefix. |

CMake uses the installed `WolframLibrary.h` SDK and does not download Wolfram
headers, a runtime, or a license. Building needs headers; execution needs a
compatible licensed kernel. Use the SDK belonging to that kernel.
The wrapper requires `WithCleanup` (introduced in Wolfram Language 12.2;
see [Wolfram's feature list](https://reference.wolfram.com/language/guide/SummaryOfNewFeaturesIn122)).
Validation covers Wolfram 15.0.1 on Linux and macOS ARM64, not every version
since 12.2. The bridge also builds against the macOS 14.3 SDK; that runtime
has not been validated.

Load the built or installed package: source files alone lack the compiled
`LibraryResources` library. Keep the package and native library directories
together when relocating an installation. Start a fresh kernel after
rebuilding; `Needs` does not reload an already loaded package. On macOS,
native libraries remain resident until the kernel exits so C++ thread cleanup
can finish safely. `LibraryUnload` releases managed tensors and waits for
native jobs, but cannot replace the resident binary.

| Environment variable | Purpose |
| --- | --- |
| `SL2CFOAMXX_MATHEMATICA_LIBRARY` | Optional absolute bridge path for a custom layout; set before starting the kernel. |
| `SL2CFOAMXX_MATHEMATICA_PACKAGE_DIR` | Full `SL2CfoamXX` package directory used by supplied examples/tests when it differs from the standard build location. |

## Callback contract

The callback receives an Association with `"Operation"`, `"Phase"`,
`"Backend"`, `"Status"`, `"Attempt"`, `"Completed"`, `"Total"`,
`"ElapsedSeconds"`, `"ETASeconds"`, and `"ETAAvailable"`. It runs in the
calling Wolfram kernel; native numerical workers only queue events. Running
updates may be coalesced, and lifecycle events keep their order. A callback
alone does not enable a notebook progress indicator: set `"ProgressBar"`
for that display.

Calls using a callback or notebook monitor use a native job and poll its
event queue. Ordinary calls use a synchronous LibraryLink call without a
polling loop. A callback `Failure`, `Throw`, or `Abort` triggers cleanup that
waits for native computation to finish before propagating the condition.
This is resource cleanup, not cancellation of the numerical calculation.
Keep callbacks short; recursively launching further calculations can add
substantial work and competing native threads.

`"ProgressBar"` is enabled by default and independent of verbosity. The
notebook wrapper uses `Monitor`; a command-line kernel uses native stderr
progress. Preparatory booster work and final publication can occur outside
the reported phase; a cache hit may have no events. Fixed-intertwiner calls
have no progress bar.

## Error details

Failures distinguish invalid arguments, bounds, overflow, allocation, file
system errors, and native operational failures. Operational failures retain
the stable C++ `"Code"`, such as `"backend_unavailable"`,
`"invalid_tensor_file"`, or `"unsupported_operation"`, together with
`"NativeCode"` and available system/path information. A rare allocation or
marshalling failure in the typed packed-array export is reported as
`"NativeArrayError"`. User callback `Failure` values are preserved.

## Validation

Enable `BUILD_TESTING=ON` and, if discovery fails, set
`-DWolfram_EXECUTABLE=/path/to/WolframKernel`. Build, then run:

```sh
ctest --test-dir build-mathematica -L mathematica --output-on-failure
```

The workflow covers independent coefficient references, fixed/coherent/tensor
agreement, both simplicity conventions, asymmetric inputs, dense and compact
storage, mappings, conversions, array ownership, errors, and callback cleanup.
A separate lifecycle entry exercises unload/reload and normal kernel exit.
An unavailable GPU is explicitly skipped; a license or activation failure
is a failure, not a successful validation.

Wolfram 15.0.1 kernel workflows pass on macOS ARM64 (CPU) and Linux (CPU and
V100 GPU), including relocated installations, Unicode metadata, and large
examples. The self-contained example notebook is parsed by the kernel, and
its initialization cells and complete calculation are exercised on the selected
backend. Visual `Monitor` rendering has not been inspected in an active front end. These results do not establish
compatibility with other Wolfram releases or GPU architectures.
