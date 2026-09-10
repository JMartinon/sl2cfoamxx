<!-- SPDX-FileCopyrightText: 2026 Jeremy Martinon -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Booster evaluation

The four-valent booster carries the Lorentzian part of the vertex amplitude.
The implementation evaluates it through reduced SL(2,C) coupling coefficients
and a spectral integral, following Eq. (52) of
[Speziale's *Boosting Wigner's nj-symbols*](https://arxiv.org/pdf/1609.01632).
The reduced coefficient is denoted $\chi$ in Speziale's Eq. (44), and plays
the role of Anderson's $X$ after accounting for their different conventions.
The internal implementation uses the corresponding $\kappa$ coefficient and
a rescaled quantity called `Z`.

For each booster, the four legs are paired as $(1,2)$ and $(3,4)$. A virtual
SL(2,C) representation $(\rho,k)$ couples each pair. The spectral formulation
combines coefficients evaluated at the boundary spins with coefficients at
the extended spins, sums over admissible $k$, and integrates over
$\rho\in\mathbb R$. Phase and dimension factors connect these coefficients
to the SU(2) intertwiner normalization used in the [vertex](theory.md).

Direct evaluation of the closed form repeatedly forms large, cancelling
complex terms. Instead, the library computes a minimal coefficient and
propagates it through the Lorentz-group recurrence relations of
[Anderson, Rączka, Rashid, and Winternitz, *Recursion and Symmetry Relations
for the Clebsch–Gordan Coefficients of the Homogeneous Lorentz
Group*](https://pubs.aip.org/aip/jmp/article-abstract/11/3/1059/224199/Recursion-and-Symmetry-Relations-for-the-Clebsch?redirectedFrom=fulltext).
Their Eqs. (19) give three ways to advance a coupled spin, Eqs. (25) cover
the exceptional diagonals, and Eqs. (29)–(32) connect minimal boundary
coefficients. The following formulas describe their adaptation as used by
the implementation, rather than identifying the paper's $X$ directly with
the stored numerical values.

## From the reduced coefficient to the stored quantity

At fixed representations $(\rho_1,k_1)$, $(\rho_2,k_2)$, and $(\rho,k)$,
write the varying SU(2) spins as $(J_1,J_2,J)$. On this page **all spins in
equations are ordinary integer or half-integer spins**; the code represents
them by doubled integers. In a booster, the external labels are
$k_1=j_1$, $k_2=j_2$, while the varying spins reach the virtual labels
$J_1=\ell_1$, $J_2=\ell_2$.

The SU(2) Clebsch–Gordan factor removes all magnetic dependence:

$$
\langle \rho_1,k_1;J_1m_1,\rho_2,k_2;J_2m_2
\mid\rho,k;Jm\rangle
=C^{Jm}_{J_1m_1,J_2m_2}\,\chi(J_1,J_2,J).
$$

For fixed representation labels, Speziale's Eq. (44) has the form

$$
\begin{aligned}
\chi(J_1,J_2,J)
={}&\mathcal N(\rho_1,k_1;\rho_2,k_2;\rho,k)\\
&\times(-1)^{(k_1+k_2+k+J_1+J_2+J)/2}
\sqrt{d_{J_1}d_{J_2}d_J}\,\kappa(J_1,J_2,J).
\end{aligned}
$$

Here $d_s=2s+1$. Writing $K=k_1+k_2+k$ and
$P=\rho_1+\rho_2+\rho$, the representation-dependent factor is

$$
\mathcal N=\frac{x(\rho_1,k_1;\rho_2,k_2;\rho,k)}{4\sqrt{2\pi}}
\Gamma\!\left(\frac{1-iP+K}{2}\right)
\Gamma\!\left(\frac{1-iP-K}{2}\right),
$$

where $x$ is Speziale's representation-dependent phase. Thus
$\mathcal N$ is constant as the three SU(2) spins vary. Anderson's canonical basis
has a different phase convention. Its continuous parameter also obeys
$\rho_{\mathrm A}=2\rho$: the radicand in that paper is
$J^2+\rho_{\mathrm A}^2/4$. These changes must be made before transferring a
recurrence to $\kappa$.

The algebraic rule for such a change is simple. If a recurrence reads
$X(u)=\sum_v a_{uv}X(v)$ and $X(u)=g(u)\kappa(u)$, the transformed coefficient
is $a_{uv}g(v)/g(u)$. Spin-dependent phases and dimensions therefore change
the recurrence coefficients; a normalization constant at fixed
representations cancels.

The implementation additionally distributes the spectral denominator among
the coefficients. Define

$$
\mathcal D_{12}(\rho,k)
=\cosh\!\bigl[\pi(\rho_1+\rho_2+\rho)\bigr]
+\cos\!\bigl[\pi(k_1+k_2+k)\bigr],
\qquad
\widetilde\kappa=\frac{\kappa}{\sqrt{\mathcal D_{12}}}.
$$

The actual table is

$$
Z(J_1,J_2,J)
=(-1)^{k_1+k_2+J_1+J_2}
\sqrt{d_{J_1}d_{J_2}d_J}\,
\widetilde\kappa(J_1,J_2,J).
$$

Thus `z_to_k_factor` removes the signed dimension factor and returns
$\widetilde\kappa$, not an unscaled $\kappa$. There are two coefficients for
each pair in the booster integrand, one at boundary spins and one at virtual
spins. Their two factors $\mathcal D_{12}^{-1/2}$ supply that pair's complete
spectral denominator. The second pair supplies $\mathcal D_{34}^{-1}$ in the
same way.

## Minimal coefficients and logarithmic scaling

The minimal $\kappa$ coefficient consists of a complex Gamma prefactor and a
finite magnetic sum. In schematic form,

$$
\kappa_{\min}=P(\rho_1,j_1;\rho_2,j_2;\rho,k,J)
\sum_{n=n_{\min}}^{n_{\max}} w_n F_n,
$$

where

$$
w_n=\begin{pmatrix}j_1&j_2&J\\n&k-n&-k\end{pmatrix},
$$

$P$ collects the factors independent of $n$, and $F_n$ contains factorial
and Gamma ratios. This is the minimal configuration of Eq. (48) in the
working derivation listed in the [references](references.md). The admissible magnetic
range is

$$
n_{\min}=\max(-j_1,-j_2+k),\qquad
n_{\max}=\min(j_1,j_2+k).
$$

The seed construction is implemented by `kappaMinVec` in
`src/boosters/B4_function.hpp`. Its magnetic row comes from the normalized
[Luscombe–Luban 3j recurrence](wigner-symbols.md), implemented in
`src/wigner/ThreeJRecurrence.hpp`. This SU(2) recurrence prepares one factor
inside the seed; it is distinct from the Anderson SL(2,C) recurrence that
propagates the completed seed across $(J_1,J_2,J)$. The magnetic rows depend
only on discrete labels, so the booster builder reuses them across the
quadrature nodes of the current computation.

The remaining magnetic terms advance by the ratio

$$
\frac{F_{n+1}}{F_n}=
\sqrt{\frac{(j_1-n)(j_2+k-n)}
{(j_1+n+1)(j_2-k+n+1)}}
\frac{1+j_1+j_2-k+2n+i(\rho_1-\rho_2+\rho)}
{1-j_1-j_2-k+2n+i(\rho_1-\rho_2-\rho)}.
$$

This is the ratio for $F_n$; the 3j weights $w_n$ multiply its successive
values. Advancing by a ratio avoids a new set of Gamma evaluations for every
magnetic label.

The seed evaluator separates magnitude from phase throughout:

1. Triangle and magnetic-range checks eliminate forbidden seeds.
2. Log-factorials and complex log-Gamma values form the prefactor and the
   first magnetic term.
3. The first term is scaled to unit magnitude. During the sum, the current
   term and partial sum are rescaled together when their largest magnitude
   leaves the interval $[10^{-100},10^{100}]$; the removed scale is accumulated
   logarithmically.
4. The final magnitude and phase are reconstructed after summation.

The log-Gamma implementation in `src/wigner/ln_gamma.hpp` uses a nine-term
Lanczos approximation, reflection, and explicit phase handling. The seed
absorbs the supplied spectral normalization through
$\exp(-\texttt{logDen}/2)$, where
$\texttt{logDen}=\log\mathcal D_{12}$. With
$h=\pi|\rho_1+\rho_2+\rho|/2$, the denominator is evaluated through
$\mathcal D_{12}=2\cosh^2h$ for even $k_1+k_2+k$ and
$\mathcal D_{12}=2\sinh^2h$ for odd $k_1+k_2+k$. Logarithmic evaluation
avoids forming an overflowing hyperbolic cosine.

All discrete labels remain **doubled integers** through the seed interface,
3j lookup, and admissibility checks. Physical half-integer values are formed
only inside floating-point expressions. In particular, a phase with doubled
exponent $Dq$ is evaluated as

$$
(-1)^{Dq/2}=i^{Dq}\in\{1,i,-1,-i\},
$$

so odd doubled labels retain their complex phase exactly.

## The seeded domain and its boundaries

At fixed representation labels, the physical domain is

$$
J_1=k_1,k_1+1,\ldots,\ell_{1,\max},\qquad
J_2=k_2,k_2+1,\ldots,\ell_{2,\max},
$$

$$
\max(k,|J_1-J_2|)\le J\le\min(J_{\max},J_1+J_2),
\qquad J_1+J_2+J\in\mathbb Z.
$$

For the EPRL booster, only $0\le k\le k_1+k_2$ is needed: the same $k$
occurs in a coefficient at the minimal external spins $(j_1,j_2)$, and
cannot exceed their coupled boundary intertwiner. This restriction matters.
The Anderson paper treats more general representation triples, including
$k>k_1+k_2$; the production booster does not need to seed that additional
case.

The initial cell is

$$
(J_1,J_2,J)=(k_1,k_2,J_0),\qquad
J_0=\max(k,|k_1-k_2|),
$$

provided it lies within the requested domain and has the admissible parity.
The code distinguishes three cases:

| Representation labels | Initial coupled spin | Additional minimal boundary |
| --- | --- | --- |
| $\lvert k_1-k_2\rvert\le k\le k_1+k_2$ | $J_0=k$ | The initial cell suffices. |
| $k_1>k_2+k$ | $J_0=k_1-k_2$ | Increase $J_2$ while decreasing $J$, with $J_1=k_1$. |
| $k_2>k_1+k$ | $J_0=k_2-k_1$ | Increase $J_1$ while decreasing $J$, with $J_2=k_2$. |

In the second case, the minimal boundary is the segment

$$
Z(k_1,k_2+n,k_1-k_2-n),
\qquad n=0,\ldots,k_1-k_2-k.
$$

All its cells have the same $J_1+J_2+J=2k_1$, so a recurrence that only
raises this sum cannot generate the segment from its endpoint. A dedicated
one-step relation does so. Define the real generator factors

$$
Q_a(s)=\sqrt{(s^2-k_a^2)(s^2+\rho_a^2)},\qquad
Q(s)=\sqrt{(s^2-k^2)(s^2+\rho^2)},
$$

$$
r_a(s)=\frac{2Q_a(s)}{s},\qquad
r(s)=\frac{2Q(s)}{s}.
$$

At their lower representation bound the corresponding $Q$ and $r$ vanish;
$r(0)$ is set to zero. For a boundary cell $(k_1,v,w)$ with
$v+w=k_1$, the implemented update is

$$
Z(k_1,v+1,w-1)
=\frac{r(w)}{r_2(v+1)}
\sqrt{\frac{(v+\tfrac32)w}{(w-\tfrac12)(v+1)}}\,
Z(k_1,v,w).
$$

This is the boundary propagation associated with Anderson's Eqs. (29)–(30),
after the normalization changes above. The code uses the forward form
when $k\le k_2$ and an algebraically equivalent inverse form when $k_2<k$.
Exchanging the two external representation labels gives the third case in
the table. Cells outside the planned maxima are skipped.

## The three general recurrence branches

For an interior cell, the code uses one of three homogeneous relations,
each with at most four predecessors. They correspond to Anderson's
Eqs. (19b), (19c), and (19a), respectively. The equations below are for the
**stored $Z$**, with the code's $\rho$ convention and all spin-dependent
rescalings already included.

### Raising the first spin

Write the target as $(p+1,v,w)$. The first branch is

$$
\begin{aligned}
L_1 Z(p+1,v,w)={}&i\alpha_1 Z(p,v,w)\\
&+\beta_1 Q(w)Z(p,v,w-1)\\
&+\eta_1 Q_1(p)Z(p-1,v,w)\\
&+\delta_1 Q_2(v)Z(p,v-1,w),
\end{aligned}
$$

where

$$
L_1=Q_1(p+1)
\sqrt{\frac{
(w+p+v+2)(w+p+v+1)(-w+p+v+1)(w+p-v+1)
}{(2p+1)(2p+3)}},
$$

$$
\begin{aligned}
\alpha_1={}&\sqrt{(w-p+v)(w+p+v+1)}\\
&\times\left[-\frac{(p+1)k\rho}{w}
+\frac{(w-v)k_1\rho_1}{p}
+\frac{(p+1)k_2\rho_2}{v}\right],
\end{aligned}
$$

$$
\beta_1=\frac{p+1}{w}
\sqrt{(w+p-v)(-w+p+v+1)},
$$

$$
\eta_1=\frac{p+1}{p}
\sqrt{\frac{
(-w+p+v)(w+p-v)(w-p+v)(w-p+v+1)
}{(2p+1)(2p-1)}},
$$

$$
\delta_1=-\frac{p+1}{v}
\sqrt{\frac{2v+1}{2v-1}}
\sqrt{(w+p-v+1)(-w+p+v)}.
$$

The first predecessor has a purely imaginary coefficient; the other three
coefficients are real. This is why a real-valued booster can still require
complex recurrence tables. The signs are part of the $Z$ convention.
Dropping the factor $i$, or transplanting the paper's coefficients without
its phase conversion, would define a different recurrence.

### Raising the second spin

The second branch has the predecessor pattern

$$
\begin{aligned}
Z(u,q+1,w)\ \leftarrow\ \{
&Z(u,q,w),\ Z(u,q,w-1),\\
&Z(u-1,q,w),\ Z(u,q-1,w)\}.
\end{aligned}
$$

Its coefficients follow from the first branch by exchanging the labels of
external representations 1 and 2 and the corresponding SU(2) axes, and
**reversing the sign of the imaginary coefficient**. More explicitly,
apply the first-branch formulas with
$(p,v)=(q,u)$ and
$(k_1,\rho_1;k_2,\rho_2)\mapsto(k_2,\rho_2;k_1,\rho_1)$.
Keep $L_1,\beta_1,\eta_1,\delta_1$ so obtained, but replace
$i\alpha_1$ with $-i\alpha_1$. The second-step predecessor is now
$Z(u,q-1,w)$, while the cross-axis predecessor is $Z(u-1,q,w)$.
This specifies the entire second branch without assuming a symmetry of the
input spins or spectral labels.

### Raising the coupled spin

For a target $(u,v,t+1)$, the third branch is

$$
\begin{aligned}
L_3 Z(u,v,t+1)={}&i\alpha_3 Z(u,v,t)\\
&+\beta_3 Q(t)Z(u,v,t-1)\\
&+\eta_3 Q_1(u)Z(u-1,v,t)\\
&+\delta_3 Q_2(v)Z(u,v-1,t),
\end{aligned}
$$

with

$$
L_3=\frac{Q(t+1)}{2t+1}
\sqrt{(t+u+v+2)(t+u+v+1)(t+u-v+1)(t-u+v+1)},
$$

$$
\begin{aligned}
\alpha_3={}&\sqrt{(t+u+v+1)(-t+u+v)}\\
&\times\left[\frac{k\rho(u-v)}{t}
+(t+1)\left(\frac{k_2\rho_2}{v}-\frac{k_1\rho_1}{u}\right)\right],
\end{aligned}
$$

$$
\beta_3=\frac{t+1}{t(2t+1)}
\sqrt{(t+u-v)(t-u+v)(-t+u+v)(1-t+u+v)},
$$

$$
\eta_3=\frac{t+1}{u}\sqrt{\frac{2u+1}{2u-1}}
\sqrt{(t+u-v)(1+t-u+v)},
$$

$$
\delta_3=\frac{t+1}{v}\sqrt{\frac{2v+1}{2v-1}}
\sqrt{(1+t+u-v)(t-u+v)}.
$$

Only $Q(t+1)$, $Q(t)$, and the term proportional to $\rho$ vary over the
quadrature nodes. The remaining factors depend on discrete geometry and
the fixed external representations.

### Choosing a valid branch

The compiler visits $J_1$, then $J_2$, then $J$ in increasing order, with
unit steps in ordinary-spin notation. It prefers the first branch when
$J_1>k_1$ and its left-hand geometric radicand is positive. Otherwise it
tries the second branch with $J_2>k_2$, and finally the third with $J>k$.
Each selected relation reads cells already available in that ordering.
Initial, minimal-boundary, and exceptional-diagonal cells are marked as
protected and are not overwritten by this general traversal.

The displayed coefficients apply when their associated predecessor is
physical. They must not be evaluated literally at every formal boundary.
For example, when $p-1<k_1$, the term involving $Z(p-1,v,w)$ is absent;
one must not first evaluate a singular $\eta_1$ and then multiply it by
zero. Likewise, factors such as $k_1\rho_1/p$ are assigned their boundary
value zero when $p=k_1=0$. The implementation handles these cases with
branch and range guards, and keeps zero-filled lower halos for invalid
predecessor addresses.

## Exceptional diagonals

The three general branches all fail when one SU(2) spin is zero and the
other two coincide: the triangle is admissible, but lowering any single
axis destroys it. Anderson's Eqs. (25a)–(25c) instead advance the two equal
spins together. The executor prepares these diagonals before replaying the
general recurrence.

For $k=0$, let $z_s=Z(s,s,0)$, starting at
$s_0=\max(k_1,k_2)$. The implemented three-term diagonal relation is

$$
\mathcal L_s z_{s+1}=\mathcal H_s z_s-\mathcal P_s z_{s-1},
$$

$$
\mathcal L_s=-\frac{2(s+1)}{\sqrt{(2s+1)(2s+3)}}
r_1(s+1)r_2(s+1),
$$

$$
\mathcal P_s=-\frac{2s}{\sqrt{(2s-1)(2s+1)}}r_1(s)r_2(s),
$$

$$
\begin{aligned}
\mathcal H_s={}&8s^2+8s+4-4k_1^2-4k_2^2\\
&+4\rho_1^2+4\rho_2^2-4\rho^2
-\frac{8k_1\rho_1k_2\rho_2}{s(s+1)}.
\end{aligned}
$$

At the first cell $s=s_0$, the predecessor is absent and
$\mathcal P_s$ is set to zero before evaluation. At $s=0$, the final
fraction in $\mathcal H_s$ is also set to zero. In particular, a first
half-integer cell $s=1/2$ must not trigger the apparent singular denominator
in $\mathcal P_s$.

The other two dedicated relations fill $Z(0,s,s)$ when $k_1=0$ and
$Z(s,0,s)$ when $k_2=0$, using the corresponding forms of Anderson's
Eqs. (25b) and (25c). Their first predecessors are omitted in the same
way. The boundary and diagonal evaluators also guard divisions by small
coefficients. These guards are numerical safeguards, not estimates of the
error in the resulting recurrence table.

## Compiling and replaying the recurrence

The recurrence separates naturally into a discrete traversal and arithmetic
at the quadrature nodes. `PureGeomCacheZ` records triangle-dependent square
roots and branch choices. `KappaTapeZ::compile` then includes the fixed
external $(\rho_1,k_1)$ and $(\rho_2,k_2)$ factors and emits the ordered
operations needed for one virtual $k$. Consecutive operations of the same
branch are grouped into chunks, while their dependency order is preserved.

After division by $L_1$ or its second-branch counterpart, an update has the
form

$$
Z_{\rm out}
=i(a\rho+b)Z_1+cQ(J)Z_2+dZ_3+eZ_4.
$$

For the third branch, the stored constants instead give

$$
Z_{\rm out}
=\frac{i(a\rho+b)Z_1+cQ(J-1)Z_2+dZ_3+eZ_4}{Q(J)},
$$

where $J$ is the target coupled spin. The tape stores $a,b,c,d,e$ and
addresses; it does not store a separately expanded instruction stream for
each quadrature node. Per-node lookup arrays supply $Q(J)$, $1/Q(J)$,
and $Q(J-1)/Q(J)$ over the required spin range.

`KappaFastZ_ExecuteTapeTile` evaluates the minimal seed at each node,
propagates the minimal boundaries and special diagonals, then replays this
tape for a tile of 4, 8, or 16 spectral nodes. Each cell stores contiguous
real lanes followed by contiguous imaginary lanes. For example,
$i\alpha Z_1$ contributes $-\alpha\operatorname{Im}Z_1$ to the real part
and $\alpha\operatorname{Re}Z_1$ to the imaginary part. This simple real
arithmetic can be vectorized across nodes without repeating geometry
choices inside the lane loop. The paired tables for legs $(1,2)$ and
$(3,4)$ subsequently provide the booster integrand.

This strategy avoids reevaluating a cancelling closed-form sum at every
virtual spin. It does not establish unconditional numerical stability for
all spins, spectral parameters, or cutoffs. Seed cancellation, recurrence
roundoff, underflow, and the quadrature approximation remain distinct
limitations; the recurrence does not carry a certified error bound.

## Mapping the spectral integral

The integration variable here is the spectral parameter $\rho$, rather than
the rapidity in the original group-integral definition. The implementation
uses

$$
\rho(\xi)=\rho_c+W\frac{\xi}{\sqrt{1-\xi^2}},
\qquad -1<\xi<1,
\qquad
\frac{d\rho}{d\xi}=\frac{W}{(1-\xi^2)^{3/2}}.
$$

For the four physical boundary spins of this booster,

$$
J_{\mathrm{tot}}=j_1+j_2+j_3+j_4,\qquad
W=\sqrt{(J_{\mathrm{tot}}+4)\max(1,\gamma)}.
$$

Channels are grouped by their maximum virtual representation label
$k_{\max}$. For boundary and extended intertwiners $(a,b)$ in ordinary-spin
units, this maximum is $\min(a,b)$. The integration center is shared by the
group:

$$
\rho_c=
\begin{cases}
\gamma k_{\max}, & \rho_j=\gamma j,\\
\gamma(k_{\max}+1), & \rho_j=\gamma(j+1).
\end{cases}
$$

The first simplicity prescription is the default. The same chosen
prescription determines the four external $\rho_j$ values. The integration
center depends on the group's $k_{\max}$, while the current $k$ still enters
the spectral measure.

The implementation sums nonnegative virtual labels with the admissible
parity. With $w_q$ denoting the transformed-interval quadrature weights, each
node contributes the combined weight

$$
\omega_q=w_q\frac{W}{(1-\xi_q^2)^{3/2}}
\left[\rho(\xi_q)^2+k^2\right]s_k,
\qquad
s_k=
\begin{cases}
\tfrac12,& k=0,\\
1,& k>0.
\end{cases}
$$

The remaining booster phases and dimension factors are applied when the
recurrence tables are combined.

## A composite 128-node rule

`src/boosters/B4Quadrature.hpp` stores the positive half of a 64-point
Gauss–Legendre rule. Reflection reconstructs the full rule, which is then
mapped independently onto $[-1,0]$ and $[0,1]$. Thus the production quadrature
has **64 nodes on each half interval, 128 nodes in total**:

$$
\int_{-1}^{1}f(\xi)\,d\xi
\approx\frac12\sum_{q=1}^{64}v_q
\left[
f\!\left(\frac{u_q-1}{2}\right)
+f\!\left(\frac{u_q+1}{2}\right)
\right],
$$

where $(u_q,v_q)$ is the 64-point rule on $[-1,1]$. The nodes avoid the
endpoints of the spectral map. This is a fixed composite rule, with no
adaptive refinement or estimated quadrature error. The shell cutoff,
quadrature approximation, and floating-point evaluation are separate sources
of numerical error; the computation supplies no certified combined error
bound.

## Producing only the required booster entries

The preceding 6j support analysis specifies which shell combinations and
intertwiner channels can contribute to the vertex. Booster construction
carries those supports forward. It additionally applies local triangle,
parity, and exact exchange-symmetry checks when constructing its output tasks.

The harvesting stage reads the required cells from both recurrence tables,
combines boundary and extended coefficients, restores phase and dimension
factors, and accumulates their quadrature contributions into the output
tensor. Intermediate recurrence cells may still be needed to reach those
outputs, even when the cells themselves are never harvested.

For example, harvesting $Z(J_1,J_2,J)$ through the first general branch can
require $Z(J_1-1,J_2,J)$ and $Z(J_1-2,J_2,J)$ even if neither corresponds to
a requested booster output. Each predecessor may in turn require earlier
cells. A reduced recurrence domain must therefore contain the complete
dependency chain of every harvested entry. Pruning the booster task list
does not justify setting arbitrary internal $Z$ cells to zero.

The current tape compiler traverses the admissible cells within the planned
spin maxima; it does not construct a minimal dependency graph separately
for each harvested output. The useful reductions occur at complementary
levels: supports eliminate unused output channels and tasks, physical
minima and maxima bound recurrence storage, and triangle conditions avoid
nonphysical recurrence updates.

For coherent amplitudes, the producer also sums the boundary-intertwiner
index against its complex coherent coefficients during harvesting. It
therefore emits the smaller complex booster tensors used by the
[coherent contraction](coherent-states.md). Fixed-intertwiner and full-tensor
calculations retain their respective real booster outputs.

The production booster builders execute on the host using OpenMP and
vectorization. Their tensors feed the CPU or GPU
[contraction stage](contractions.md); GPU execution still includes this host
preparation phase. The division of work and memory lifetimes are described
in [execution](execution.md).
