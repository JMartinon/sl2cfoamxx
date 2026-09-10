<!-- SPDX-FileCopyrightText: 2026 Jeremy Martinon -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Scientific conventions

All four APIs use the same scientific conventions:

- Spins and intertwiners are **doubled integers**: `1` means physical `j=1/2`,
  and `2` means physical `j=1`. Spins are ordered
  `(j12,j13,j14,j15,j23,j24,j25,j34,j35,j45)`; intertwiners follow
  `(i1,i2,i3,i4,i5)`. Internal virtual intertwiners are denoted
  `(k1,k2,k3,k4,k5)`, with `k1=i1` after gauge fixing; the remaining
  virtual labels are internal summation indices.
- The full tensor is ordered as **`A(i1,i2,i3,i4,i5)` in column-major order**:
  `i1` varies fastest in a canonical dense payload. This is the existing public
  order; internal contractions use matrices grouped by `i1` to feed BLAS.
  Symmetry-compressed storage packs triangular panels while preserving logical
  indexing; its physical buffer is not an ordinary dense array. Expand it when
  a canonical dense array is needed.
- `Dl` is the nonnegative shell cutoff. The Barbero-Immirzi parameter `gamma`
  must be finite and `≥0`; negative values are rejected before computation.
- Boundary angles are 20 `(theta, phi)` pairs in radians. Their order is
  listed below; normals `n_ab` and `n_ba` belong to different local frames.
- Tensor indices are positions in admissible intertwiner ranges: zero-based
  in C++/Python, **one-based in Julia/Mathematica**. Use `indices_from_intertwiners()`
  (`IndicesFromIntertwiners` in Mathematica) to
  convert doubled labels into indices for the calling language.

| C++/Python angle rows | Julia/Mathematica angle rows | Oriented normals |
| --- | --- | --- |
| `0..3` | `1..4` | `n12, n13, n14, n15` |
| `4..7` | `5..8` | `n21, n23, n24, n25` |
| `8..11` | `9..12` | `n31, n32, n34, n35` |
| `12..15` | `13..16` | `n41, n42, n43, n45` |
| `16..19` | `17..20` | `n51, n52, n53, n54` |

The public API checks nonnegative labels, parity, admissibility, supported
cutoffs, and finite angles. Invalid scientific inputs raise
`std::invalid_argument` in C++, `ValueError` in Python, and `ArgumentError` in Julia.
Mathematica returns a structured `Failure` that can be checked with `FailureQ`.
See the
[C++ scientific guide](cpp-api.md#scientific-input-validation) for
recoupling bases, closure checks, and error handling.

| Function | Result | Backend |
| --- | --- | --- |
| `compute_amplitude(spins, intertwiners, Dl, gamma)` | One real fixed-intertwiner amplitude | CPU |
| `compute_coherent_amplitude(spins, Dl, gamma, angles)` | One complex coherent amplitude | CPU/GPU |
| `compute_full_tensor(spins, Dl, gamma)` | Rank-five `FullTensor` | CPU/GPU |
| `compute_coherent_coefficients(spins, angles)` | Coefficients for one tetrahedron | CPU |
| `check_closure(spins, angles)` | Five closure reports | CPU |

Coherent and full-tensor calls accept an explicit backend override. There is
no silent GPU-to-CPU fallback. `backend_available()`, `build_info()`, and
`cuda_device_info()` expose the loaded library and available hardware.
Mathematica uses capitalized names such as `ComputeFullTensor`,
`BackendAvailableQ`, `BuildInfo`, and `CUDADeviceInfo`.
