<!-- SPDX-FileCopyrightText: 2026 Jeremy Martinon -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Inside sl2cfoam++

sl2cfoam++ evaluates Lorentzian EPRL vertex amplitudes by turning the booster decomposition into a tensor network. Repeated special-function evaluations become reusable arrays; angular-momentum admissibility determines which parts of those arrays need to exist; explicit contraction orders control the computational work and memory footprint.

The common mathematical object is the vertex tensor

$$
A_{i_1i_2i_3i_4i_5}(j_{ab};\gamma,\Delta\ell).
$$

Its five axes label boundary intertwiners. The library evaluates a selected component, the complete tensor, or its contraction with coherent boundary coefficients. These pages follow the mathematics through the numerical representations and contraction algorithms used for each result.

## The numerical organization

The mathematical input consists of ten boundary spins, the Barbero–Immirzi parameter, a shell cutoff, and either boundary intertwiners or coherent-state normals. Three numerical tasks then build the amplitude:

1. **[Wigner symbols and supports](wigner-symbols.md).** Triangle inequalities and parity identify admissible channels. Normalized recurrences evaluate the required 6j sequences, which are packed into supported blocks.
2. **[Booster tensors](boosters.md).** Minimal reduced-coefficient seeds and Lorentz-group recurrences generate the spectral integrand. Fixed quadrature fills four booster tensors on the compatible supports.
3. **[Contraction](contractions.md).** Slicing and blocking the network produce a fixed component, the complete tensor, or a coherent amplitude. Each output has its own contraction organization.

![Dependency diagram: boundary data determine admissibility supports; Wigner recurrences and booster recurrences with quadrature supply supported tensors to sliced contractions, yielding a fixed component, full tensor, or coherent amplitude.](assets/numerical-organization.svg){ #numerical-organization-figure .overview-figure }

This diagram shows dependencies rather than a mandatory order of execution. The supports can be constructed before the numerical 6j values. Memory-limited CPU execution and GPU execution can produce 6j operands in slabs during contraction instead of retaining every symbol tensor at once. Booster construction remains a host computation in the current implementation.

## Contents

<div class="grid cards" markdown>

- **[Theory and conventions](theory.md)**

    The vertex amplitude, recoupling channels, cutoff, and tensor weights.

- **[Wigner symbols and supports](wigner-symbols.md)**

    Stable 6j and 3j sequences, normalization, admissibility, and GPU generation.

- **[Boosters and quadrature](boosters.md)**

    Minimal seeds, Lorentz-group recurrences, and the finite-interval integral for $B_4$.

- **[Coherent states](coherent-states.md)**

    Coherent boundary coefficients and their early contraction with booster tensors.

- **[Contraction algorithms](contractions.md)**

    The CPU and GPU contraction trees for fixed, full-tensor, and coherent results.

- **[Execution and memory](execution.md)**

    Supports, symmetry, storage, streaming, and parallel execution.

- **[References](references.md)**

    The papers and mathematical methods behind the implementation.

</div>

## Internal architecture

One numerical implementation serves the C++, Python, Julia, and Mathematica interfaces. The public layer validates scientific inputs and selects the computation. Private amplitude and full-tensor layers construct supports and schedule contraction work; the booster and Wigner layers evaluate special functions; shared core utilities provide storage, memory information, and BLAS coordination.

CUDA implementations consume the supported numerical representations through dedicated bridges. The logical amplitude and its conventions are shared across execution backends, even when the order of contraction and placement of intermediate arrays differ.
