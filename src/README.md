# Private source layout

The installed API lives in `../include/sl2cfoamxx`. Everything below `src/`
is private implementation detail and may change without an ABI commitment.

- `api/` — public entry points, validation, configuration, tensor containers,
  cache lookup, and CPU/GPU dispatch.
- `amplitude/` — fixed-intertwiner and coherent-amplitude support planning,
  packed panels, slab selection, and contractions.
- `full_tensor/` — full-output support, packed data, reflection orbits, CPU
  contraction, streaming, and execution planning.
- `boosters/` — booster-channel construction, quadrature, kappa evaluation,
  and B4 tensor filling.
- `wigner/` — seed-free Wigner 3j/6j recurrences and 6j tensor generation.
- `core/` — platform, storage, and informational-output utilities shared by
  the production implementations.
- `gpu/` — CUDA-only coherent and full-tensor production backends, including
  the cuTENSOR API adapter and architecture-selected SIMT/WMMA producers.

The intended dependency direction is:

```text
api -> amplitude/full_tensor -> boosters/wigner -> core
```

GPU bridges may consume the packed feature-layer types, but CPU numerical code
must not depend on CUDA. The optional tests exercise complete workflows
through the public API; they do not compile private numerical kernels or
require a separate Wigner-symbol library.

Host/device POD representations shared by CPU and CUDA code belong in `core/`;
backend namespaces may provide aliases but must not duplicate their definitions.
Likewise, process-global vendor-library coordination belongs in a common GPU
adapter so independently implemented backends cannot acquire separate locks for
the same external resource.

Private `.hpp` files declare shared types and functions; matching `.cpp` files
implement substantial non-template operations. The amplitude and full-tensor
domains now have twenty implementation units covering support construction,
packing, contraction, slab selection, execution planning, coherent coefficients,
and progress reporting. `core/HostMemoryBudget.cpp` also keeps platform,
cgroup, Slurm, and NUMA probing out of its small interface header. CMake compiles
these sources into the same shared library with hidden private symbols and the
existing host numerical flags.

Keep template definitions visible where callers instantiate them. Booster
harvesting specializes by axis, rank, and tile size; recurrence reconstruction
accepts callback types; tensor and buffer containers specialize by element type,
rank, or alignment. Explicit instantiation is an alternative when all supported
specializations can be listed. Small indexing, support-intersection, and scalar
loop helpers remain inline where they are used within numerical kernels.

The remaining booster and Wigner headers still mix templates and small helpers
with some non-template implementations. Their size is not a requirement of C++:
additional filling entry points, channel builders, and recurrence setup can be
separated incrementally, with numerical and performance validation. Header
placement alone does not make code faster or preserve numerical ordering.
CUDA kernels live in `.cu` files and retain their source-specific numerical flags.
