<!-- SPDX-FileCopyrightText: 2026 Jeremy Martinon -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Coherent boundary states

The coherent amplitude contracts the rank-five vertex tensor with five
complex vectors, one for each boundary tetrahedron. Each vector expresses
a Livine–Speziale coherent intertwiner in the SU(2) basis used by the vertex.
These states were introduced by Livine and Speziale in
[*A new spinfoam vertex for quantum gravity*](https://arxiv.org/abs/0705.0674)
by averaging a tensor product of SU(2) coherent states over the group.
The average projects onto the invariant subspace; its norm generally
differs from one.
The library computes these vectors from spinors and Wigner 3j recurrences,
then absorbs four of them into the boosters before the large contraction.
This removes four boundary-intertwiner axes from the tensor network.

For the geometric setting and numerical examples, see Donà, Fanizza, Sarno,
and Speziale, [*Numerical study of the Lorentzian EPRL spin foam
amplitude*](https://arxiv.org/abs/1903.12624). The definitions below specify
the phase, pairing, and normalization conventions of this implementation.

## Normals, spinors, and recoupling bases

An oriented face normal is represented by

$$
\vec{n}(\theta,\phi)
=(\sin\theta\cos\phi,\sin\theta\sin\phi,\cos\theta),
\qquad
z(\theta,\phi)=
\begin{pmatrix}
\cos(\theta/2)\\ e^{i\phi}\sin(\theta/2)
\end{pmatrix}.
$$

The normals $\vec{n}_{ab}$ and $\vec{n}_{ba}$ belong to the local frames
of tetrahedra $a$ and $b$. They are separate data. The spinor section above
fixes a phase as well as a direction: replacing a spinor by
$e^{i\alpha}z$ multiplies its spin-$j$ coherent state by $e^{2ij\alpha}$.
The coefficient construction inserts no antipodal map or complex conjugation
of the supplied spinors.

The public APIs provide the fixed `lorentzian_boundary_angles` and
`euclidean_boundary_angles` reference data in this convention. The Lorentzian
preset closes for face-spin ratios `(5,5,5,5,2,2,2,2,2,2)`, while the Euclidean
preset closes for equal face spins. Both contain twenty ordered pairs in
radians; selecting a preset supplies boundary normals without changing the
Lorentzian EPRL model evaluated by the library.

For each ordered quadruple, the intertwiner basis couples the first two
legs and the last two legs. The five local orders are:

| Tetrahedron | Ordered spin–normal pairs | Coupling channel |
| --- | --- | --- |
| $1$ | $(j_{12},\vec{n}_{12}),\ (j_{13},\vec{n}_{13}),\ (j_{14},\vec{n}_{14}),\ (j_{15},\vec{n}_{15})$ | $(j_{12},j_{13})\mid(j_{14},j_{15})$ |
| $2$ | $(j_{23},\vec{n}_{23}),\ (j_{24},\vec{n}_{24}),\ (j_{25},\vec{n}_{25}),\ (j_{12},\vec{n}_{21})$ | $(j_{23},j_{24})\mid(j_{25},j_{12})$ |
| $3$ | $(j_{34},\vec{n}_{34}),\ (j_{35},\vec{n}_{35}),\ (j_{13},\vec{n}_{31}),\ (j_{23},\vec{n}_{32})$ | $(j_{34},j_{35})\mid(j_{13},j_{23})$ |
| $4$ | $(j_{45},\vec{n}_{45}),\ (j_{14},\vec{n}_{41}),\ (j_{24},\vec{n}_{42}),\ (j_{34},\vec{n}_{43})$ | $(j_{45},j_{14})\mid(j_{24},j_{34})$ |
| $5$ | $(j_{15},\vec{n}_{51}),\ (j_{25},\vec{n}_{52}),\ (j_{35},\vec{n}_{53}),\ (j_{45},\vec{n}_{54})$ | $(j_{15},j_{25})\mid(j_{35},j_{45})$ |

These are also the bases of the corresponding full-tensor axes and booster
boundary channels. The shared spin–normal mapping constructs each vector
directly in its target basis. A different pairing generally requires a Racah
transformation mixing intertwiner components; the same numerical label $i$
does not identify the same state across different pairings.

## The coefficient being evaluated

Within one tetrahedron, write the ordered physical spins as
$(j_1,j_2,j_3,j_4)$, and let $J_a=2j_a$. The admissible doubled intertwiner
labels $I=2i$ are

$$
\begin{aligned}
I_{\min}&=\max(|J_1-J_2|,|J_3-J_4|),\\
I_{\max}&=\min(J_1+J_2,J_3+J_4),\\
I&=I_{\min},I_{\min}+2,\ldots,I_{\max},
\end{aligned}
$$

provided the pair sums $J_1+J_2$ and $J_3+J_4$ have the same parity and the
interval is nonempty. The coefficient vector contains one entry per
admissible channel, ordered by increasing $i$.

The magnetic components of a spin-$j$ coherent state are

$$
\psi_j(m;z)=
\sqrt{\binom{2j}{j+m}}\,z_0^{j+m}z_1^{j-m},
\qquad -j\le m\le j.
$$

The north spinor $(1,0)$ therefore has only $m=j$. The code stores this vector
with integer index $k=j+m$, running from $0$ to $J$.

The real, orthonormal four-valent intertwiner basis is

$$
\begin{aligned}
\iota^{(i)}_{m_1m_2m_3m_4}
={}&\sqrt{2i+1}\sum_{M=-i}^{i}(-1)^{i+M}\\
&\times
\begin{pmatrix}j_1&j_2&i\\m_1&m_2&-M\end{pmatrix}
\begin{pmatrix}j_3&j_4&i\\m_3&m_4&M\end{pmatrix}.
\end{aligned}
$$

The coefficient of the coherent intertwiner in this basis is

$$
C_i=\sum_{m_1,m_2,m_3,m_4}
\iota^{(i)}_{m_1m_2m_3m_4}
\prod_{a=1}^{4}\psi_{j_a}(m_a;z_a).
$$

There is no conjugation of the coherent magnetic components in this
expression. The factor $\sqrt{2i+1}$ normalizes the intertwiner basis and is
applied once, after accumulating the magnetic sums.

## Eliminate one magnetic sum by an SU(2) rotation

For the normalized first spinor $z_1=(a,b)^T$, define

$$
g=\begin{pmatrix}\overline a&\overline b\\-b&a\end{pmatrix},
\qquad gz_1=\begin{pmatrix}1\\0\end{pmatrix}.
$$

The same matrix rotates all four spinors. SU(2) invariance of the
intertwiner preserves $C_i$, while the first leg now contributes only
$m_1=j_1$. The implementation sets the rotated pivot exactly to $(1,0)$
to remove residual roundoff in its vanishing component. This local SU(2)
choice is separate from the Lorentzian gauge fixing that removes the
booster at tetrahedron 1.

The two 3j selection rules then fix $m_2=M-j_1$ and $m_4=-M-m_3$.
Writing the remaining magnetic vectors for the rotated spinors gives

$$
\begin{aligned}
C_i={}&\sqrt{2i+1}\sum_M(-1)^{i+M}
\begin{pmatrix}j_1&j_2&i\\j_1&M-j_1&-M\end{pmatrix}
\psi_{j_2}(M-j_1)\\
&\times\sum_{m_3}
\begin{pmatrix}j_3&j_4&i\\m_3&-M-m_3&M\end{pmatrix}
\psi_{j_3}(m_3)\psi_{j_4}(-M-m_3).
\end{aligned}
$$

This is the expression evaluated numerically. All sums follow their exact
magnetic and triangle supports.

## Generate magnetic vectors from their peak

Directly evaluating every binomial coefficient and complex power would
repeat work and risk underflow at the endpoints. The implementation chooses
a seed near the peak of the binomial distribution,

$$
k_0=\operatorname{round}(J|z_0|^2),
$$

and evaluates its magnitude using logarithms and log-factorials:

$$
\begin{aligned}
\log|\psi_{k_0}|={}&\tfrac12
\bigl[\log\Gamma(J+1)-\log\Gamma(k_0+1)
-\log\Gamma(J-k_0+1)\bigr]\\
&+k_0\log|z_0|+(J-k_0)\log|z_1|,\\
\arg\psi_{k_0}={}&k_0\arg z_0+(J-k_0)\arg z_1.
\end{aligned}
$$

Two recurrences generate the rest of the vector:

$$
\begin{aligned}
\psi_{k+1}&=\psi_k
\sqrt{\frac{J-k}{k+1}}\frac{z_0}{z_1},\\
\psi_{k-1}&=\psi_k
\sqrt{\frac{k}{J-k+1}}\frac{z_1}{z_0}.
\end{aligned}
$$

The two pole cases have explicit one-component expressions, including their
spinor phases. The internal default treats a spinor component of magnitude
at most $10^{-14}$ as zero. Spinors are normalized before the rotation and
magnetic-vector generation; for vectors constructed from angles this mainly
protects against roundoff. Each complete magnetic vector has unit squared
norm up to floating-point error.

## Build the real 3j workspace

The 3j data depend on the four spins, independently of the normals. A
workspace groups them into blocks labeled by $\mu=2M$. Each block stores:

- the supported intertwiner rows and magnetic index pairs;
- the signed first factor $(-1)^{i+M}
  \left(\begin{smallmatrix}j_1&j_2&i\\j_1&M-j_1&-M\end{smallmatrix}\right)$;
- a real matrix of the second 3j factor, with rows indexed by $i$ and columns
  by the admissible $m_3$ values.

The [Wigner implementation](wigner-symbols.md) generates whole rows using
the stable two-sided Schulten–Gordon / Luscombe–Luban construction. It
propagates minimal-solution ratios inward from both ends, bridges the
oscillatory region with the three-term recurrence, fixes the magnitude by
orthogonality, and fixes the
sign analytically at the upper endpoint. Cyclic column permutations place
the requested symbols in the recurrence's preferred order without an
additional phase.

For a row varying $m_2$ at fixed $j_1,j_2,j_3,m_1$, the normalization is

$$
(2j_1+1)\sum_{m_2}
\begin{pmatrix}j_1&j_2&j_3\\m_1&m_2&-m_1-m_2\end{pmatrix}^{\!2}=1,
$$

and the upper-endpoint phase is $(-1)^{j_2-j_3-m_1}$. For a row varying
$j_1$ at fixed $j_2,j_3,m_2,m_3$, the normalization uses weights $2j_1+1$
and the upper-endpoint phase is $(-1)^{j_2-j_3+m_2+m_3}$.

To evaluate a block, the code forms the complex products
$\psi_{j_3}(m_3)\psi_{j_4}(-M-m_3)$, multiplies them by the stored real
matrix, then weights the result by the signed first factor and
$\psi_{j_2}(M-j_1)$. Accumulation over blocks completes every $C_i$.
Workspace construction can distribute independent blocks across CPU threads;
the production coefficient evaluator accumulates them serially. An internal
compensated-summation option covers both the matrix row products and the
final block sum; ordinary production calls leave it disabled.

## Projection normalization and closure

Normalizing the spinors does not normalize their projection onto the
invariant subspace. The library retains the coefficients as computed, so

$$
\mathcal N=\sum_i|C_i|^2
$$

is generally different from one. A convention with normalized coherent
intertwiners would replace each vector by $C_i/\sqrt{\mathcal N}$ when
$\mathcal N>0$. The corresponding vertex would be divided by
$\prod_{a=1}^{5}\sqrt{\mathcal N_a}$. The implementation applies no such
final rescaling.

The closure vector at tetrahedron $a$ is

$$
\mathbf G_a=\sum_{b\ne a}j_{ab}\vec{n}_{ab}.
$$

It provides a geometric diagnostic of the supplied data. The coefficient
construction remains defined for non-closing data and does not project
normals onto a closed configuration. The optional closure check computes
these vectors without changing the angles or the coefficients; its default
is disabled. Closure alone does not establish the geometry of a Lorentzian
4-simplex.

## Fuse four coefficient vectors into the boosters

In the conventions above, the coherent amplitude is the bilinear contraction

$$
A_{\mathrm{coh}}=
\sum_{i_1,\ldots,i_5} A_{i_1i_2i_3i_4i_5}
\prod_{a=1}^{5}C_{a,i_a}.
$$

No coefficient is conjugated during booster filling or the final
contraction. For tetrahedra 2 through 5, distribute the sums over $i_a$
into the booster factors:

$$
\widetilde B_a(k_a,\boldsymbol\ell_a)
=\sum_{i_a}C_{a,i_a}B_a(i_a,k_a,\boldsymbol\ell_a),
\qquad a=2,3,4,5.
$$

The real rank-five booster data are generated in bounded slabs. Each slab
is immediately contracted over its boundary-intertwiner axis, producing
complex rank-four data. The slab is then reused. Thus the coherent path
does not first materialize four complete real boosters or the full vertex
tensor. Its outputs retain the exact supports derived from the 6j symbols.

When exact booster permutation symmetries apply, one real slab can feed
several tetrahedra. The producer gathers the union of required intertwiner
labels, aligns each target's own coefficient vector with that union, and
applies the appropriate permutation phases. Distinct coherent vectors are
still contracted separately. Exactly zero coefficients can be omitted
without introducing an approximation.

## The remaining gauge-fixed coefficient

Tetrahedron 1 has no booster. Its coefficient $C_1$ must instead meet the two
endpoint 6j factors containing $i_1$, denoted $sjA$ and $sjE$ in the
[contraction network](contractions.md).

The CPU route keeps bounded groups of $i_1$ channels, shares the work common
to those channels, and weights their final contributions by $C_{1,i_1}$.
The GPU route eliminates $i_1$ earlier. For each fixed $x$, it forms

$$
K_x(k_5,k_2,\ell_{25})=
\sum_{i_1} C_{1,i_1}
sjA_{x,i_1,k_5,\ell_{25}}
sjE_{x,i_1,k_2,\ell_{25}}.
$$

Since both 6j factors are real, two real strided-batched matrix products
compute the real and imaginary parts of $K_x$. The remaining GPU network
then uses $K_x$ and the four complex fused boosters. Its middle stages use
cuTENSOR, followed by a real matrix product acting on the complex components
and a final kernel restricted to the exact supports. The detailed ordering
and intermediate indices are described in [Contractions](contractions.md).

Both backends therefore remove four external intertwiner axes before the
main contraction, while accommodating the additional complex arithmetic.
Complex values already enter through the fused boosters; the treatment of
$C_1$ determines where its remaining intertwiner sum is eliminated.

## Optional truncation of the C1 support

The coherent execution plan can omit channels satisfying

$$
|C_{1,i}|<\varepsilon\max_j|C_{1,j}|.
$$

The shared CPU/GPU setting is `coherent_amplitude.pruning_ratio`, with
default $\varepsilon=10^{-8}$; zero disables this relative threshold.
Exactly zero coefficients are still skipped. The retained vector is not
renormalized. This setting affects contraction work associated with $C_1$;
it does not truncate $C_2$ through $C_5$ or change coefficient generation.

This is an additional approximation beyond the shell cutoff and quadrature.
The omitted coefficient norm alone is not a relative error bound on the
vertex: the associated tensor components and cancellations also matter.
The exact contraction identity above uses all nonzero coefficients.
