<!-- SPDX-FileCopyrightText: 2026 Jeremy Martinon -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# References

## Foundational EPRL vertex

Jonathan Engle, Etera Livine, Roberto Pereira, and Carlo Rovelli, **LQG vertex with finite Immirzi parameter**, *Nuclear Physics B* **799**, 136–149 (2008). [arXiv:0711.0146](https://arxiv.org/abs/0711.0146), [journal DOI](https://doi.org/10.1016/j.nuclphysb.2008.02.018).

The foundational construction of the EPRL vertex at finite Barbero–Immirzi parameter, including the Lorentzian theory and its SU(2) boundary Hilbert space.

## Simplicity constraints and Y-map

Sergei Alexandrov, **The new vertices and canonical quantization**, *Physical Review D* **82**, 024024 (2010). [arXiv:1004.2260](https://arxiv.org/abs/1004.2260), [journal DOI](https://doi.org/10.1103/PhysRevD.82.024024).

Section 2.1.1, Eq. (10), gives the shifted Lorentzian restriction corresponding to $(\rho,k)=(\gamma(j+1),j)$ in this documentation's notation. It follows from requiring the projected connection to have the Ashtekar–Barbero parameter $\gamma$ while selecting the lowest SU(2) subspace.

You Ding and Carlo Rovelli, **The volume operator in covariant quantum gravity**, *Classical and Quantum Gravity* **27**, 165003 (2010). [arXiv:0911.0543](https://arxiv.org/abs/0911.0543), [journal DOI](https://doi.org/10.1088/0264-9381/27/16/165003).

The derivation of the boundary space through weakly imposed constraints and its relation to the canonical volume operator. This paper treats the Euclidean theory; it supplies context for the constraint prescription rather than a separate derivation of the Lorentzian shifted labels.

Francesco Gozzini, **A high-performance code for EPRL spin foam amplitudes**, *Classical and Quantum Gravity* **38**, 225010 (2021). [arXiv:2107.13952](https://arxiv.org/abs/2107.13952), [journal DOI](https://doi.org/10.1088/1361-6382/ac2b0b).

The `sl2cfoam-next` reference. Section 2 uses the shifted Y-map and cites Alexandrov and Ding–Rovelli when discussing the linear simplicity constraints. The [theory chapter](theory.md#simplicity-constraints-and-the-two-y-maps) states both prescriptions supported by this library and identifies its default.

## Booster decomposition

Simone Speziale, **Boosting Wigner's nj-symbols**, *Journal of Mathematical Physics* **58**, 032501 (2017). [arXiv:1609.01632](https://arxiv.org/abs/1609.01632), [journal DOI](https://doi.org/10.1063/1.4977752).

The primary reference for the booster decomposition and the spectral representation used by the booster code. Equation numbers here refer to the [arXiv PDF](https://arxiv.org/pdf/1609.01632); the experimental HTML conversion renumbers some equations.

Pietro Donà and Pietropaolo Frisoni, **How-To compute EPRL spin foam amplitudes**, *Universe* **8**, 208 (2022). [arXiv:2202.04360](https://arxiv.org/abs/2202.04360), [journal DOI](https://doi.org/10.3390/universe8040208).

A graphical and mathematical account of EPRL amplitudes. Section 6 develops the single-vertex decomposition into an SU(2) 15j symbol and four boosters, providing the context for the introductory vertex figure.

## Lorentz-group recurrence

R. L. Anderson, R. Rączka, M. A. Rashid, and P. Winternitz, **Recursion and Symmetry Relations for the Clebsch–Gordan Coefficients of the Homogeneous Lorentz Group**, *Journal of Mathematical Physics* **11**, 1059–1068 (1970). [Publisher's article page](https://pubs.aip.org/aip/jmp/article-abstract/11/3/1059/224199/Recursion-and-Symmetry-Relations-for-the-Clebsch?redirectedFrom=fulltext), [DOI:10.1063/1.1665197](https://doi.org/10.1063/1.1665197).

The source of the reduced-coefficient recurrence relations adapted by the code. The implementation's scaled variables, seed normalization, and execution order are described in the [booster chapter](boosters.md).

## Recurrence for Wigner symbols

James H. Luscombe and Marshall Luban, **Simplified recursive algorithm for Wigner 3j and 6j symbols**, *Physical Review E* **57**, 7274 (1998). [DOI:10.1103/PhysRevE.57.7274](https://doi.org/10.1103/PhysRevE.57.7274).

The numerical recurrence-method reference. The [Wigner-symbol chapter](wigner-symbols.md) develops the complete-sequence recurrences, normalization, endpoint phases, and triangle supports used throughout the vertex calculation.

Klaus Schulten and Roy G. Gordon, **Exact recursive evaluation of 3j- and 6j-coefficients for quantum-mechanical coupling of angular momenta**, *Journal of Mathematical Physics* **16**, 1961–1970 (1975). [Authors' publication page](https://www.ks.uiuc.edu/Publications/Papers/paper.cgi?tbcode=SCHU75), [journal DOI](https://doi.org/10.1063/1.522426).

The stable sequence-recurrence construction underlying the Schulten–Gordon / Luscombe–Luban approach used by the Wigner kernels.

## Coherent intertwiners

Etera R. Livine and Simone Speziale, **A new spinfoam vertex for quantum gravity**, *Physical Review D* **76**, 084028 (2007). [arXiv:0705.0674](https://arxiv.org/abs/0705.0674), [journal DOI](https://doi.org/10.1103/PhysRevD.76.084028).

The foundational definition of coherent intertwiners by SU(2) group averaging, their geometric interpretation, and their dependence on closure. Equation (14) gives the invariant projection; Eq. (15) makes explicit that its norm is generally not one. The [coherent-state chapter](coherent-states.md) describes how the library computes this projection in an orthonormal intertwiner basis.

## Coherent amplitudes

Pietro Donà, Marco Fanizza, Giorgio Sarno, and Simone Speziale, **Numerical study of the Lorentzian Engle–Pereira–Rovelli–Livine spin foam amplitude**, *Physical Review D* **100**, 106003 (2019). [arXiv:1903.12624](https://arxiv.org/abs/1903.12624), [journal DOI](https://doi.org/10.1103/PhysRevD.100.106003).

The geometric and numerical context for coherent boundary amplitudes. The [coherent-state chapter](coherent-states.md) specifies the coefficient algorithm and conventions of this implementation.

## Working notes

Jeremy Martinon, **On the 4-valent booster function in LQG**, working note dated August 4, 2026. Section 3.2.1, Eq. (48), gives the minimal seed used as the starting point of booster evaluation. The supplied `B4_review.pdf` informed this documentation; no public publication identifier was supplied.

The [contraction chapter](contractions.md) describes the contraction design from the current source, including the differences between the CPU and GPU treatments of C1 and supported full-tensor storage.

The supplied working-note PDFs are reference material and are not redistributed as part of this site. Software citation metadata is maintained in the repository's `CITATION.cff`.
