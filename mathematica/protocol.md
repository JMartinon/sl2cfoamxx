<!-- SPDX-FileCopyrightText: 2026 Jeremy Martinon -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Private Mathematica LibraryLink protocol, version 1

`sfx_dispatch[op, ints, reals, text1, text2]` loads with types
`{Integer,{Integer,1,"Constant"},{Real,1,"Constant"},"UTF8String","UTF8String"}`
and returns `"UTF8String"`. All calls return JSON:
`{"ok":true,"value":VALUE}` or
`{"ok":false,"error":{"kind":"InvalidArgument|Bounds|Overflow|Allocation|Filesystem|Library|Runtime","code":INTEGER,"message":STRING,"system_code":INTEGER,"path1":STRING,"path2":STRING}}`.
Stable library error codes match C++; other codes are zero. JSON floats use
17 significant digits and always contain a decimal point or exponent, so
RawJSON preserves machine-real scalar types, including `0.0` and `-0.0`.
The Wolfram package decodes responses through an explicitly UTF-8-encoded
`ByteArray` and `ImportByteArray[..., "RawJSON"]`; direct `ImportString`
rejected valid Unicode path characters in the tested Wolfram 15 kernel.
Complex values are `[real,imag]`. All enum values are
the native integers, booleans JSON booleans, absent optional values `null`.
Unused vectors/strings must be empty. Every operation checks exact lengths.
All wire indices, including physical indices, are **one-based**; CUDA device
ordinals remain zero-based. `backend=-1` preserves the omitted native backend;
CPU=0/GPU=1 are explicit. Jobs require an explicit backend.

Manager names: `SL2CfoamXXTensor`, `SL2CfoamXXJob`. Create managed expressions
before passing destination IDs. A destination must be a live, empty managed
tensor; it is filled exactly once. Each lookup pins the native shared owner.
Mode-0 allocation failure is caught; subsequent dispatch reports failure.

On macOS the bridge retains one process-lifetime loader reference to itself
and its dependencies. Wolfram otherwise unloads their GCC TLS destructor
code before the kernel thread exits. LibraryLink uninitialization still
unregisters managers, joins workers, and frees tensor owners. Reloading a
different native binary requires a fresh kernel.

| Op | Operation | ints | reals | text1 / text2 | value |
|---|---|---|---|---|---|
| 0 | protocol version | [] | [] | | 1 |
| 1 | default config | [] | [] | | config record |
| 2 | current config | [] | [] | | config record |
| 3 | configure | config integers (17) | [pruning_ratio] | directory / output_path | null |
| 4 | build info | [] | [] | | native snake_case fields |
| 5 | backend available | [backend] | [] | | boolean |
| 6 | CUDA count | [] | [] | | integer |
| 7 | CUDA device info | [ordinal] | [] | | native snake_case fields |
| 8 | boundary-angle preset | [preset] | [] | | 20 lists of [theta,phi]; 0=Lorentzian, 1=Euclidean |
| 10 | fixed amplitude | [Dl,cache_policy,spins10,intertwiners5] | [gamma] | | real |
| 11 | coherent amplitude | [Dl,backend,spins10] | [gamma,angles40] | | complex pair |
| 12 | full tensor | [destination,Dl,backend,spins10] | [gamma] | | tensor metadata |
| 13 | coefficients | [spins4] | [angles8] | | list of complex pairs |
| 14 | closure | [spins10] | [atol,rtol,angles40] | | {all_closed,nodes} |
| 15 | intertwiner indices | [spins10,intertwiners5] | [] | | indices5 |
| 16 | coherent amplitude from coefficients | [Dl,backend,spins10,lengths5] | [gamma,coefficient pairs] | | complex pair |
| 20 | create tensor | [destination,init,layout,shape5] | [] | | metadata |
| 21 | create mapped | [destination,init,layout,shape5] | [] | path / | metadata |
| 22 | load | [destination] | [] | path / | metadata |
| 23 | map | [destination,access] | [] | path / | metadata |
| 24 | metadata | [id] | [] | | metadata |
| 25 | file metadata | [] | [] | path / | metadata |
| 26 | canonical get | [id,indices5] | [] | | real |
| 27 | canonical set | [id,indices5] | [value] | | null |
| 28 | physical get | [id,index] | [] | | real |
| 29 | physical set | [id,index] | [value] | | null |
| 30 | fill | [id] | [value] | | null |
| 31 | clone | [destination,source] | [] | | metadata |
| 32 | dense copy | [destination,source] | [] | | metadata |
| 33 | save | [id] | [] | path / | null |
| 34 | flush | [id] | [] | | null |
| 35 | convert storage | [destination,source,storage,overwrite] | [] | path / | metadata |
| 36 | format | [id,max_values,precision,include_metadata] | [] | | string; max_values=-1 means all |
| 37 | tag | [id] | [] | | list of 1024 byte integers |
| 38 | release tensor | [id] | [] | | null; retained stale handles fail safely |
| 40 | tensor from canonical values | [destination,shape5] | flattened row-major values | | metadata |
| 50 | start full job | [job,destination,Dl,backend,spins10] | [gamma] | | null |
| 51 | start coherent job | [job,Dl,backend,spins10] | [gamma,angles40] | | null |
| 52 | poll job | [job] | [] | | {done:boolean,events:[event records]} |
| 53 | finish job | [job] | [] | | coherent pair or full metadata; joins, transfers result once |
| 54 | abandon/release job | [job] | [] | | null; disables queue, joins, discards unclaimed result/error |
| 55 | start coherent coefficient job | [job,Dl,backend,spins10,lengths5] | [gamma,coefficient pairs] | | null |

Coefficient pairs are flattened real,imag values in C1, C2, C3, C4, C5 order.
The five lengths count complex entries, and the real array has exactly
`1 + 2 Sum[lengths5]` elements. The bridge checks these counts before copying;
the native API checks the finite entries and canonical intertwiner dimensions.
Jobs own all five copied vectors. Contraction does not conjugate or normalize
the entries, uses configured C1 pruning, and skips angle-based closure checks.

Preset angles come from the public C++ `lorentzian_boundary_angles` and
`euclidean_boundary_angles` constants. The package exposes their copied values
as `$LorentzianBoundaryAngles` and `$EuclideanBoundaryAngles` packed real arrays.

Angles are interleaved theta,phi in the public native node order. Configuration
integer order is: backend, progress_bar, simplicity_constraint, verbosity,
printing_style, tensor_storage, reuse_existing, check_closure,
full_tensor_cpu_final_tile, booster_rho_tile, coherent_gpu_pipeline_slots,
full_tensor_gpu_producer_waves_per_sm, full_tensor_cpu_p_batch, cpu_threads,
memory_budget_bytes, maximum_payload_size_bytes, full_tensor_gpu_row_capacity.
Config records are `{ints:[...],reals:[pruning_ratio],directory:string,output_path:string}`.

Tensor metadata fields: `shape`, `strides`, `logical_values`, `stored_values`,
`stored_bytes`, `dense_bytes`, `storage`, `layout`, `reassociated_layout`,
`mapped`, `writable`, `symmetry_compressed`, `dense`, `path`, `scientific`.
Scientific is null or `{spins,Dl,gamma,simplicity_constraint}`. Strides describe
native logical column-major indexing, not the copied Wolfram array.
Closure node fields: `vector`, `residual_norm`, `scale`, `relative_residual`,
`closed`. Progress fields: `operation`, `phase`, `backend`, `status`, `attempt`,
`completed`, `total`, `elapsed_seconds`, `eta_seconds`, `eta_available`.

`sfx_tensor_array[id,physical]` loads with `{Integer,"Boolean"}` and result
`{Real,_}`. False returns independent Wolfram-owned canonical rank-five packed
array; True returns independent physical rank-one packed array. Validate the
handle via metadata first. This function alone reports LibraryFunctionError
for allocation/marshalling failure; it has no last-error global/TLS state.
All ordinary operational errors are available through dispatch.

No Wolfram API is used by native numerical workers. Job inputs and outputs
are native-owned. Poll coalesces running updates while retaining lifecycle
events. Finish/abandon/destruction never depend on further polling. Managed
destruction and uninitialization join every worker before unloading code.
On Abort or user callback failure, the WL wrapper must protect cleanup with
AbortProtect, call 54, and then propagate the original condition. CPU/GPU
computation completes before cleanup returns; no cancellation is claimed.
Normal operations create no thread or polling loop.
