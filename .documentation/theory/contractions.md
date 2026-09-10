<!-- SPDX-FileCopyrightText: 2026 Jeremy Martinon -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Contraction algorithms

The vertex is a tensor network of five weighted SU(2) 6j tensors and four
booster tensors. Its contraction order depends on the output: a fixed
intertwiner component, a rank-five tensor, or a coherent scalar. The library
uses explicit contraction trees, with local choices based on admissible support
sizes, rather than running a general tensor-network path search.

The five 6j factors share the index $x$. Slicing over $x$ removes this
five-way connection from each working network and bounds the size of its
intermediates. The lifted indices $\ell_{24}$, $\ell_{25}$, and $\ell_{35}$
each connect four tensors; the other internal indices connect three.
The trees resolve these connections through reusable pair products, matrix
multiplication, and kernels that intersect exact supports. A three-way
connection does not require a single library
operation with three operands: the production GPU trees combine fused
operands with binary cuTENSOR contractions.

## Network and notation

All symbols below denote physical spins. The implementation represents spins
and intertwiners by doubled integers; array coordinates are positions in
admissible channel lists. The lifted labels are

$$
\ell_{ab}=j_{ab},j_{ab}+1,\ldots,j_{ab}+\Delta\ell,
\qquad ab\in\{23,24,25,34,35,45\}.
$$

Write $S_A,\ldots,S_E$ for the weighted tensors called `sjA`, …, `sjE` in
the source. Their bare 6j factors are:

| Tensor | Bare 6j factor |
| --- | --- |
| $S_A(x,i_1,k_5,\ell_{25})$ | $\left\{\begin{matrix}i_1&\ell_{25}&x\\k_5&j_{14}&j_{15}\end{matrix}\right\}$ |
| $S_B(x,k_5,k_4,\ell_{35},\ell_{45})$ | $\left\{\begin{matrix}k_5&j_{14}&x\\k_4&\ell_{35}&\ell_{45}\end{matrix}\right\}$ |
| $S_C(x,k_3,k_4,\ell_{24},\ell_{34},\ell_{35})$ | $\left\{\begin{matrix}k_4&\ell_{35}&x\\k_3&\ell_{24}&\ell_{34}\end{matrix}\right\}$ |
| $S_D(x,k_3,k_2,\ell_{23},\ell_{24})$ | $\left\{\begin{matrix}k_3&\ell_{24}&x\\k_2&j_{13}&\ell_{23}\end{matrix}\right\}$ |
| $S_E(x,i_1,k_2,\ell_{25})$ | $\left\{\begin{matrix}i_1&\ell_{25}&x\\k_2&j_{13}&j_{12}\end{matrix}\right\}$ |

The stored values also absorb dimension factors and real recoupling phases.
In particular, $S_A$ contains $2x+1$, which must not be inserted a second time
in the contraction. Phase splitting in the generators preserves a real
representation for admissible integer and half-integer labels. Booster
production includes the permutation phases associated with its four node
layouts.

With $B_n$ denoting the booster attached to tetrahedron $n$, the complete
index structure is

$$
\begin{aligned}
A_{i_1i_2i_3i_4i_5}(\Delta\ell)
={}&\sum_{\ell_{23},\ell_{24},\ell_{25},\ell_{34},\ell_{35},\ell_{45}}
\sum_{k_2,k_3,k_4,k_5,x}
S_A(x,i_1,k_5,\ell_{25})\\
&\quad\times S_B(x,k_5,k_4,\ell_{35},\ell_{45})
S_C(x,k_3,k_4,\ell_{24},\ell_{34},\ell_{35})\\
&\quad\times S_D(x,k_3,k_2,\ell_{23},\ell_{24})
S_E(x,i_1,k_2,\ell_{25})\\
&\quad\times B_2(i_2,k_2,\ell_{23},\ell_{24},\ell_{25})
B_3(i_3,k_3,\ell_{34},\ell_{35},\ell_{23})\\
&\quad\times B_4(i_4,k_4,\ell_{45},\ell_{24},\ell_{34})
B_5(i_5,k_5,\ell_{25},\ell_{35},\ell_{45}).
\end{aligned}
$$

Every sum is restricted by its admissibility supports. Tetrahedron 1 is gauge
fixed, so there is no $B_1$.

## Fixed component on the CPU

`compute_amplitude` fixes all five external intertwiners. For one $x$ slice,
the CPU first forms two wing intermediates:

$$
\begin{aligned}
T_1(k_4,\ell_{35},\ell_{45},\ell_{25})
&=\sum_{k_5}S_B(x,k_5,k_4,\ell_{35},\ell_{45})
S_A(x,i_1,k_5,\ell_{25})
B_5(i_5,k_5,\ell_{25},\ell_{35},\ell_{45}),\\
T_3(k_3,\ell_{23},\ell_{24},\ell_{25})
&=\sum_{k_2}S_D(x,k_3,k_2,\ell_{23},\ell_{24})
S_E(x,i_1,k_2,\ell_{25})
B_2(i_2,k_2,\ell_{23},\ell_{24},\ell_{25}).
\end{aligned}
$$

The pair products $S_AB_5$ and $S_EB_2$ are reused across the remaining
internal intertwiner. This avoids repeating their multiplication for every
term in a three-factor dot product.

For each local block $(\ell_{24},\ell_{34},\ell_{35})$, the next stages are

$$
\begin{aligned}
T_2(k_4,\ell_{25})
&=\sum_{\ell_{45}}
B_4(i_4,k_4,\ell_{45},\ell_{24},\ell_{34})
T_1(k_4,\ell_{35},\ell_{45},\ell_{25}),\\
T_4(k_3,\ell_{25})
&=\sum_{\ell_{23}}
B_3(i_3,k_3,\ell_{34},\ell_{35},\ell_{23})
T_3(k_3,\ell_{23},\ell_{24},\ell_{25}).
\end{aligned}
$$

The block contributes

$$
\sum_{\ell_{25},k_3,k_4}
T_4(k_3,\ell_{25})
S_C(x,k_3,k_4,\ell_{24},\ell_{34},\ell_{35})
T_2(k_4,\ell_{25}).
$$

Taking $S_C$ to have rows $k_3$ and columns $k_4$, the general-spin
implementation chooses between forming $S_CT_2$ and $S_C^TT_4$ from the
support dimensions. Reflection-compatible inputs can use canonical local-block
orbits. Summing the block contributions and the $x$ slices gives the scalar.

Only the active wing buffers and local panels are needed at once. Input 6j
values can still be resident across many slices: slicing a contraction is
distinct from regenerating its inputs. The fixed-component production route
is CPU-only.

## Full tensor on the CPU

`compute_full_tensor` retains all external intertwiner axes. For a fixed
$i_1$, group the output labels into $m=(i_2,i_3)$ and $n=(i_4,i_5)$.
The wings retain $i_2$ and $i_5$, and the ensuing panels are
$T_4(k_3,\ell_{25},m)$ and $T_2(k_4,\ell_{25},n)$.

The CPU reassociates the final update as

$$
\begin{aligned}
D_4(k_4,\ell_{25},m)
&=\sum_{k_3}S_C(k_3,k_4)T_4(k_3,\ell_{25},m),\\
\Delta A_{mn}
&=\sum_{\ell_{25},k_4}
D_4(k_4,\ell_{25},m)T_2(k_4,\ell_{25},n).
\end{aligned}
$$

Rows from several local contributions are accumulated in bounded panels
before the final matrix update. This increases reuse of output tiles and
limits temporary memory. Work over $i_1$ and $x$ is scheduled using estimates
of actual support work, rather than only counting channel combinations.
General rectangular output uses vendor BLAS when sufficiently large and
custom local kernels for smaller operations. Reflection-compressed output
has a separate triangular update kernel.

When the boundary admits reflection, canonical local-block orbits supply both
a representative and its reflected contribution. These contributions are
folded into a triangular output panel. General boundary data retain the
rectangular panel. These internal representations all describe the same
rank-five object; the output container carries the mapping to its logical
intertwiner axes. [Execution and storage](execution.md) explains that mapping.

## Coherent amplitude on the CPU

For coefficients in the vertex's recoupling bases, the coherent amplitude is
the bilinear contraction

$$
A_{\mathrm{coh}}=
\sum_{i_1,\ldots,i_5}
C_1(i_1)C_2(i_2)C_3(i_3)C_4(i_4)C_5(i_5)
A_{i_1i_2i_3i_4i_5}.
$$

The contraction introduces no complex conjugation of the coefficients.
Their construction and basis conventions are described in
[Coherent boundary states](coherent-states.md).

`compute_coherent_amplitude` projects four booster tensors during their
production:

$$
\widetilde B_n(k_n,\ldots)
=\sum_{i_n}C_n(i_n)B_n(i_n,k_n,\ldots),
\qquad n=2,3,4,5.
$$

The producer consumes each real booster recurrence slab while it is available
and writes the projected complex values. Consequently, the contraction never
needs to assemble the full unprojected booster tensors with four large
external axes.

The CPU follows the wing and local-panel tree with complex $\widetilde B_n$.
It processes batches of active $i_1$ values, sharing supports and real 6j data
between them. Each $i_1$ still has its own $S_A$, $S_E$, and complex wings.
The remaining $C_1$ weighting is accumulated within the local-block work.
The amount of sharing can shrink under memory pressure, down to a streamed
calculation for one $i_1$ at a time. Relative pruning of small $C_1$ values is
an additional approximation, discussed in [Execution and storage](execution.md#numerical-approximations).

## Coherent amplitude on the GPU

The GPU eliminates $i_1$ before its principal contraction pipeline. To match
the device notation, set

$$
(a,k,i,j,v)=(i_1,k_2,k_3,k_4,k_5),\qquad
(u,q,p,r,s,t)=(\ell_{23},\ell_{24},\ell_{25},\ell_{34},\ell_{35},\ell_{45}).
$$

For each $x$ it forms

$$
K(v,k,p)=\sum_a C_1(a)S_A(x,a,v,p)S_E(x,a,k,p).
$$

The real and imaginary parts of $C_1$ are applied separately, using two real
strided-batched matrix multiplications before combining their outputs into
complex $K$. For a slab $q=\ell_{24}$, a fused operand is

$$
H(k,u,i,p)=S_D(x,i,k,u,q)\widetilde B_2(k,u,q,p).
$$

The middle and tail are

$$
\begin{aligned}
X_2(v,p,u,i)&=\sum_k K(v,k,p)H(k,u,i,p),\\
X_3(v,p,i,r,s)&=\sum_u X_2(v,p,u,i)\widetilde B_3(i,r,s,u),\\
X_4(j,r,s,v,p)&=\sum_i S_C(x,i,j,q,r,s)X_3(v,p,i,r,s),\\
W(x,q)&=\sum_{j,r,s,v,p,t}
X_4(j,r,s,v,p)\widetilde B_5(v,p,s,t)
S_B(x,v,j,s,t)\widetilde B_4(j,t,q,r).
\end{aligned}
$$

The result is $\sum_{x,q}W(x,q)$. Device recurrences generate the real 6j
data. Complex binary cuTENSOR operations form $X_2$ and $X_3$. A real FP64
strided-batched DGEMM forms $X_4$: because $S_C$ is real, real and imaginary
components of $X_3$ can be exposed as an expanded real matrix axis. A custom
kernel intersects supports for the remaining product and reduction.

Separate streams overlap central-symbol recurrence, the cuTENSOR/DGEMM middle,
and the tail. Events prevent a pipeline slot from being reused before its
consumer finishes. Multiple GPUs dynamically claim $x$ slices; their scalar
results are reduced on the host in fixed $x$ order.

## Full tensor on the GPU

The full-tensor GPU route preserves the external intertwiners. Recurrence
kernels write products $W_1=S_BB_5$ and $W_3=S_DB_2$ directly into the operand
layouts consumed by cuTENSOR. Binary contractions with $S_A$ and $S_E$ form
the two wings.

For each local block, denote the ensuing panels by $U=T_4$ and $V=T_2$.
With the convention $S_C[k_3,k_4]$, the schedule chooses either

$$
L=S_C^TU,\quad R=V,
\qquad\text{or}\qquad
L=U,\quad R=S_CV.
$$

Both yield the same update $L^TR$. Their internal row counts differ, so
the support dimensions determine the less expensive association. Custom
producers form bounded, materialized left and right row slabs. The final
update is a real FP64 `cublasDgemmStridedBatched` operation.

Reflection-compatible input uses one descriptor per canonical local-block
orbit and immediately folds each dense update into compact output. General
input uses all supported logical blocks and rectangular output through the
same GPU route. Reflection is an optional optimization, not a restriction on
the boundary spins accepted by the GPU algorithm.

The panel producers use FP64 warp matrix operations on `sm_80` and newer
architectures, and ordinary FP64 SIMT dot products on older supported
architectures such as Volta. Both consume the same prepared operands and
produce the same slab layout. Multiple GPUs receive disjoint contiguous
$i_1$ ranges, so their output pieces can be assembled without a tensor-wide
sum across devices.

## Implementation map

The numerical trees are implemented in
[fixed CPU contractions](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/amplitude/AmplitudePackedContractionGeneral.cpp),
[coherent CPU contractions](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/amplitude/AmplitudePackedCoherentI1Batch.cpp),
[full CPU contractions](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/full_tensor/CpuContraction.cpp),
[coherent GPU contractions](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/gpu/CoherentGpuBackend.cu),
and [full GPU contractions](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/gpu/FullTensorSemiFusedGpuBackend.cu).
The [weighted 6j generator](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/amplitude/AmplitudePackedSJTensors_HalfSjC.cpp)
contains the phase and dimension assignments used by the scalar network.
