<!-- SPDX-FileCopyrightText: 2026 Jeremy Martinon -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Mathematica wrapper

The user documentation lives with the other API guides:

- [Mathematica guide](../docs/mathematica.md): build, load, calculate, and work with tensors.
- [Mathematica reference](../docs/mathematica-reference.md): defaults and advanced API contracts.
- [Self-contained example notebook](../examples/mathematica_example.nb).

This directory contains the LibraryLink bridge and Wolfram package sources.

Coherent amplitudes accept either twenty angle pairs or five caller-supplied
coefficient vectors:

```wl
ComputeCoherentAmplitude[spins, Dl, gamma, angles]
ComputeCoherentAmplitude[spins, Dl, gamma, C1, C2, C3, C4, C5,
    "Backend" -> "CPU"]
```

Each vector contains finite real or complex numbers in its canonical tensor
axis order. The contraction does not conjugate or normalize them. Both forms
support the same backend and progress options; see the guide for vector
lengths, basis conventions, and pruning.
