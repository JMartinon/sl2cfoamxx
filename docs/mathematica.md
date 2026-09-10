<!-- SPDX-FileCopyrightText: 2026 Jeremy Martinon -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Mathematica API

`SL2CfoamXX` brings the C++ library into Mathematica through LibraryLink.
Build the bridge once, load the package, and call it from any notebook or
Wolfram Language script. You do not compile or link individual notebooks.

[Build and load](#build-and-load) · [First calculation](#first-calculation) ·
[Coherent amplitudes](#coherent-amplitudes) · [Tensor algebra](#tensor-algebra) ·
[Files](#save-and-load-tensors) · [Configuration](#configuration) ·
[Complete example](#run-the-complete-example)

The [Mathematica reference](mathematica-reference.md) contains all option
defaults, tensor functions, and advanced integration details.

## Build and load

You need the [native build requirements](building.md), an installed Wolfram
LibraryLink SDK, and a compatible licensed kernel. Wolfram 15.0.1 is validated
on Linux and macOS ARM64; [version and SDK details](mathematica-reference.md#build-and-loading-settings)
are listed in the reference.

From the repository root:

```sh
cmake -S . -B build-mathematica \
  -DCMAKE_CXX_COMPILER=/path/to/g++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DSL2CFOAMXX_BUILD_MATHEMATICA=ON \
  -DWolfram_ROOT=/path/to/Wolfram
cmake --build build-mathematica --parallel
```

`Wolfram_ROOT` can be a macOS application bundle. CMake searches common
installation paths; if discovery fails, set `WolframLibrary_INCLUDE_DIR` to
the directory containing `WolframLibrary.h`. To enable the GPU backend, also
follow the [NVIDIA build instructions](building.md#nvidia-gpus).

In Mathematica, add the **parent of the built package** to `$Path`:

```wl
AppendTo[$Path, "/absolute/path/to/repo/build-mathematica/mathematica"];
Needs["SL2CfoamXX`"];
BuildInfo[]
```

Alternatively, load the exact entry point with
`Get["/absolute/path/to/repo/build-mathematica/mathematica/SL2CfoamXX/Kernel/init.m"]`.
Use the built package, which contains the native bridge. Start a fresh kernel
whenever you rebuild the native library.

For an installed package, run
`cmake --install build-mathematica --prefix /your/prefix`, append
`/your/prefix/share/wolfram/Applications` to `$Path`, then use the same
`Needs` call. Keep the installed directory structure together when relocating it.

## First calculation

```wl
config = DefaultConfig[];
config["Backend"] = "CPU";
config["CPUThreads"] = 2;
Configure[config];

spins = ConstantArray[2, 10];                 (* physical j = 1 *)
intertwiners = ConstantArray[0, 5];
fixed = ComputeAmplitude[spins, intertwiners, 0, 1.2];
tensor = ComputeFullTensor[spins, 0, 1.2];
indices = IndicesFromIntertwiners[spins, intertwiners];
{fixed, TensorValue[tensor, indices], Dimensions[tensor]}
```

The first two values should agree to numerical precision. Full tensors use
symmetry-aware on-disk storage by default, and subsequent calls reuse
compatible files. To obtain an ordinary Mathematica array:

```wl
array = Normal[tensor];                     (* independent packed real array *)
array[[1, 1, 1, 1, 1]]
array[[1]]                                 (* ordinary slicing *)
```

Inputs follow the shared [scientific conventions](scientific-conventions.md):

- Spins and intertwiners are exact, nonnegative **doubled integers**: `1`
  means physical spin `1/2`. Machine-real labels such as `2.` are rejected.
- The ten spin positions are `{j12,j13,j14,j15,j23,j24,j25,j34,j35,j45}`.
  `Dl` is an exact nonnegative cutoff. `gamma` is finite and nonnegative;
  angles are finite reals in radians. Exact real input such as `6/5` is
  converted to machine precision.
- Tensor coordinates use **one-based positions** in `{i1,i2,i3,i4,i5}` order.
  Use `IndicesFromIntertwiners` to convert doubled labels to positions.
  CUDA device ordinals remain zero-based.

Invalid input and operational errors return `Failure[...]`. Check
`FailureQ[result]` before using a result in another calculation.

| Function | Result and options |
| --- | --- |
| `ComputeAmplitude[spins, intertwiners, Dl, gamma]` | Machine real, CPU only. `"CachePolicy" -> "disabled"`; use `"prefer_full_tensor"` for a best-effort persisted lookup. |
| `ComputeCoherentAmplitude[spins, Dl, gamma, angles]` | Machine complex. `"Backend" -> Automatic`, `"ProgressFunction" -> None`. |
| `ComputeFullTensor[spins, Dl, gamma]` | Managed tensor. Same backend/progress options; `"ReturnType" -> "Tensor"` or `"Array"` for a copied array. |
| `ComputeCoherentCoefficients[spins4, angles4]` | Packed complex coefficient vector for one tetrahedron. |
| `CheckClosure[spins, angles]` | Association with `"AllClosed"` and five `"Nodes"`; absolute and relative tolerances default to `10^-12`. |

`"Backend" -> Automatic` uses the configured backend. An explicit
`"Backend" -> "GPU"` requires an available CUDA backend and compatible
device; it returns a failure otherwise. Query `BackendAvailableQ["GPU"]`,
`CUDADeviceCount[]`, or `CUDADeviceInfo[0]` to inspect availability.
`BuildInfo[]` and `LibraryVersion[]` describe the loaded library.

## Coherent amplitudes

Supply twenty `{theta,phi}` pairs in the
[canonical oriented-normal order](scientific-conventions.md). Normals at
different nodes live in different local frames; the
[complete example](../examples/mathematica_example.nb) provides a closed boundary
and the coefficient permutations for all five recoupling bases.

```wl
closure = CheckClosure[spins, angles];
closed = closure["AllClosed"];
amplitude = ComputeCoherentAmplitude[spins, Dl, gamma, angles];
```

For a direct comparison against a complete tensor contraction, set
`config["CoherentAmplitude"]["PruningRatio"]` to zero using the nested-update
pattern below. The default `1.*^-8` prunes small C1 contributions and is not
a rigorous bound on the final amplitude error.

## Tensor algebra

A managed tensor retains native C++ storage. Use `Normal[tensor]` once to get
an independent, canonical packed array for Mathematica arithmetic:

```wl
array = Normal[tensor];
scaled = 2. array;
total = Total[Flatten[array]];
slice = array[[1, All, All, All, All]];
other = TensorFromArray[scaled];             (* independent native dense copy *)
```

The managed object is not a List. Use `TensorValue` for individual native
entries, and `Normal` before slicing or tensor algebra. Changing an exported
array does not change the native tensor. In contrast, `alias = tensor` shares
the same native object; use `CloneTensor` or `DenseCopyTensor` for independence.

Given five coefficient vectors in canonical axis order, contract them with:

```wl
coherentFromArray = Fold[Dot, array, Reverse[coefficients]];
```

`Dot` consumes the last axis of its left operand, hence the reversed vectors;
it does not conjugate them automatically. See [Wolfram's Dot reference](https://reference.wolfram.com/language/ref/Dot.html).
Use the recoupling permutations from the complete example to build the vectors.

To contract two tensors over their last four axes, reshape them into matrices:

```wl
a = Normal[tensor];
b = Normal[other];
rowsA = ArrayReshape[a, {First[Dimensions[a]], Times @@ Rest[Dimensions[a]]}];
rowsB = ArrayReshape[b, {First[Dimensions[b]], Times @@ Rest[Dimensions[b]]}];
matrix = rowsA . Transpose[rowsB];
```

Entry `{p,q}` is the sum of `a[[p,u,v,w,x]] b[[q,u,v,w,x]]` over
`u,v,w,x`; the four contracted dimensions must match. `TensorContract`, `TensorProduct`, and
`Transpose` support general contractions, but a full outer product can be much
larger than the result. Mathematica array operations run on CPU, even when
the native tensor was computed on GPU.

`StoredTensorValues[tensor]` instead copies the physical native payload.
It may be compressed or reassociated and must not be interpreted as
`Flatten[Normal[tensor]]`. Native dense storage has `i1` fastest;
`Normal` preserves logical axis order while arranging data for Wolfram arrays.
`TensorMetadata[tensor]["Strides"]` describes native logical column-major
strides, not the exported array's storage.

## Save and load tensors

```wl
SaveTensor[tensor, "vertex.sl2cxx"];
loaded = LoadTensor["vertex.sl2cxx"];                    (* owned, writable *)
mapped = MapTensor["vertex.sl2cxx"];                     (* read-only *)
writable = MapTensor["vertex.sl2cxx", "Access" -> "read_write"];
SetTensorValue[writable, {1, 1, 1, 1, 1}, 0.];
FlushTensor[writable];
TensorMetadata["vertex.sl2cxx"]
```

The file format is shared with C++, Python, and Julia. Loading creates owned
storage; mapping retains the file as backing. A `Normal` copy is writable even
when its source mapping is read-only. The [tensor reference](mathematica-reference.md#tensor-functions)
lists construction, metadata, inspection, and lossless storage conversion.

## Configuration

`DefaultConfig[]` and `CurrentConfig[]` return independent Associations.
Edit a complete configuration and apply it with `Configure[config]`; unknown
or missing fields are rejected. Active calculations keep their original
snapshot. For nested fields, replace the nested Association explicitly:

```wl
config = CurrentConfig[];
config["Tensors"] = Join[config["Tensors"], <|
    "Directory" -> "/absolute/path/to/my-cache",
    "Storage" -> "in_memory_dense",
    "ReuseExisting" -> False
|>];
config["CoherentAmplitude"] = Join[config["CoherentAmplitude"],
    <|"PruningRatio" -> 0.|>];
Configure[config];
```

Defaults include `"ProgressBar" -> True`, `"Verbosity" -> "silent"`,
`"Storage" -> "on_disk_sym"`, and `"SimplicityConstraint" -> "gamma_j"`.
See the [complete field/default tables](mathematica-reference.md#configuration-defaults).

Progress works for coherent and full-tensor calls on both backends: a notebook
uses `Monitor`, and a command-line kernel writes native progress to stderr.
For custom handling, use `"ProgressFunction" -> (handler[#] &)` with an
explicit backend. The callback receives a progress Association in the calling
kernel; see the [callback contract](mathematica-reference.md#callback-contract).

Errors are ordinary Wolfram `Failure` objects:

```wl
result = LoadTensor["missing.sl2cxx"];
If[FailureQ[result], Print[result["Message"]], Dimensions[result]]
```

Operational failures include a stable `"Code"` such as
`"backend_unavailable"`. See [error details](mathematica-reference.md#error-details)
for the full error and callback-cleanup behavior.

## Run the complete example

The [self-contained notebook](../examples/mathematica_example.nb) contains
all geometry, configuration, calculations, numerical checks, persistence,
progress handling, and tensor contractions. Open it in Mathematica and
evaluate its input cells in order with **Shift+Enter**. No companion script
is required.

After evaluating the package, boundary, and function-definition cells, run:

```wl
result = RunSL2CfoamXXExample["CPU"];
If[!FailureQ[result], result["Array"]]
```

The notebook locates `build-mathematica` relative to its own directory.
For another location, set `SL2CFOAMXX_MATHEMATICA_PACKAGE_DIR` to the full
built/installed `SL2CfoamXX` package directory before starting the kernel,
or add the package's parent to `$Path`. Its optional GPU cell checks backend
availability before running. A larger calculation is
`RunSL2CfoamXXExample["GPU", 12, 6]`, for physical spin `j = 6` and cutoff 6.

The workflow uses a fresh temporary cache directory and removes its files
on completion or failure. The returned tensor and array remain in memory.
It explicitly selects `gamma_j_p1` and zero coherent pruning for its numerical
comparisons. Configure a persistent cache directory again before continuing
with your own calculations.

See [validation instructions and limits](mathematica-reference.md#validation)
for kernel tests and notebook front-end status.
