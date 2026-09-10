<!-- SPDX-FileCopyrightText: 2026 Jeremy Martinon -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Wigner-symbol generation

The SU(2) part of the calculation repeatedly needs families of closely related
Wigner symbols. The five tensors in the vertex network contain 6j symbols;
the minimal booster coefficients and the coherent intertwiners contain 3j
symbols. The library generates a complete admissible sequence at a time using
normalized recurrences. It does not evaluate a factorial sum separately for
every entry or require an external Wigner-symbol table.

The numerical construction follows the nonlinear-ratio method of
[Luscombe and Luban, *Simplified recursive algorithm for Wigner 3j and 6j
symbols*](https://doi.org/10.1103/PhysRevE.57.7274). Ratios describe the small
tails near the endpoints; a three-term recurrence advances through the
interior. Orthogonality determines the magnitude and an analytic endpoint
phase determines the sign. The formulas below describe the implementation,
using physical angular momenta. Stored labels are doubled integers, so a unit
step of a spin or magnetic quantum number advances its stored label by two.

## A complete 6j sequence

After a symmetry permutation of its arguments, write the sequence as

$$
h(j)=\left\{\begin{matrix}
j&j_2&j_3\\j_4&j_5&j_6
\end{matrix}\right\}.
$$

The four triangles are $(j,j_2,j_3)$, $(j,j_5,j_6)$,
$(j_4,j_2,j_6)$, and $(j_4,j_5,j_3)$. The latter two must be
admissible before a sequence is generated. The first two determine

$$
j_{\min}=\max(|j_2-j_3|,|j_5-j_6|),\qquad
j_{\max}=\min(j_2+j_3,j_5+j_6).
$$

The sums $j_2+j_3$ and $j_5+j_6$ must have the same integer or
half-integer parity. A nonempty sequence then contains
$j_{\min},j_{\min}+1,\ldots,j_{\max}$. These bounds describe the
entire sequence needed for orthogonality normalization, even when the tensor
ultimately stores only a smaller intersection with its support.

Set $t(j)=j(j+1)$ and $t_a=j_a(j_a+1)$. For $j_{\min}>0$,
the recurrence is

$$
X(j)h(j+1)+Y(j)h(j)+Z(j)h(j-1)=0,
$$

with

$$
\begin{aligned}
G_{a,b}(j)&=\sqrt{
\bigl[j^2-(a-b)^2\bigr]
\bigl[(a+b+1)^2-j^2\bigr]},\\
E(j)&=G_{j_2,j_3}(j)G_{j_5,j_6}(j),\\
X(j)&=jE(j+1),\qquad Z(j)=(j+1)E(j),\\
Y(j)&=(2j+1)\Bigl[
t(j)\bigl(t_2+t_3-2t_4+t_5+t_6-t(j)\bigr)
+(t_5-t_6)(t_2-t_3)\Bigr].
\end{aligned}
$$

If $j_{\min}=0$, then $j_2=j_3$ and $j_5=j_6$.
The generic equation has a removable degeneracy at zero. The code uses the
regularized coefficients

$$
\begin{aligned}
Q(j)&=j\sqrt{
\bigl[(j_2+j_3+1)^2-j^2\bigr]
\bigl[(j_5+j_6+1)^2-j^2\bigr]},\\
X(j)&=Q(j+1),\qquad Z(j)=Q(j),\\
Y(j)&=(2j+1)\bigl[-t(j)+2(t_2-t_4+t_5)\bigr].
\end{aligned}
$$

For positive $j$ these follow by dividing the generic equation by
$j(j+1)$; their finite continuation supplies the boundary equation at zero.
No separately evaluated 6j seed is required.

## Stable reconstruction from the two endpoints

Index an admissible sequence by $r=0,\ldots,N-1$, with the common form

$$
X_r f_{r+1}+Y_r f_r+Z_r f_{r-1}=0.
$$

A direct recurrence from one endpoint can amplify the wrong solution in a
small, nonoscillatory tail. Instead, the implementation first computes
ratios in the direction in which their magnitudes remain bounded. Starting
from the lower boundary,

$$
s_r=\frac{f_r}{f_{r+1}},\qquad
s_0=-\frac{X_0}{Y_0},\qquad
s_r=-\frac{X_r}{Y_r+Z_r s_{r-1}}.
$$

Starting from the upper boundary,

$$
u_r=\frac{f_r}{f_{r-1}},\qquad
u_{N-1}=-\frac{Z_{N-1}}{Y_{N-1}},\qquad
u_r=-\frac{Z_r}{Y_r+X_r u_{r+1}}.
$$

Each tail accepts only finite ratios of magnitude at most one.
A small denominator or a ratio outside that range ends the
ratio sweep; it does not truncate the Wigner sequence.

The reconstruction anchors one value at unity after the lower ratio tail.
Multiplying the stored lower ratios reconstructs that tail. The ordinary
three-term recurrence then advances through the interior, and the stored
upper ratios extend the value arriving at the upper tail. This connects the
two endpoint solutions without separately normalizing two amplitude
sequences or dividing by a potentially tiny matching value. The current
implementation has no intermediate rescaling pass; it applies the final
orthogonality scale once. A missing ratio tail simply leaves more of the
sequence to the ordinary recurrence. A one-element sequence is determined
directly by normalization and phase.

For the 6j convention above, normalization and the upper-end sign are

$$
(2j_4+1)\sum_{j=j_{\min}}^{j_{\max}}(2j+1)h(j)^2=1,
\qquad
\operatorname{sgn}h(j_{\max})=(-1)^{j_2+j_3+j_5+j_6}.
$$

The integer exponent is evaluated from doubled labels. Very small endpoint
values can underflow to signed zero; preserving that sign prevents an
otherwise correct normalized sequence from acquiring an overall sign error.
Fused multiply-add operations also reduce rounding in the recurrence and
normalization sums. These measures improve numerical stability, but do not
turn floating-point Wigner symbols into exact arithmetic.

## 3j sequences for magnetic sums

The same reconstruction principle generates

$$
f(m)=\begin{pmatrix}
j_1&j_2&j_3\\m_1&m&-m_1-m
\end{pmatrix}
$$

for fixed $j_1,j_2,j_3,m_1$. After triangle and magnetic-parity checks,

$$
m_{\min}=\max(-j_2,-j_3-m_1),\qquad
m_{\max}=\min(j_2,j_3-m_1).
$$

Define

$$
\begin{aligned}
U(m)&=\sqrt{(j_2+1-m)(j_2+m)
(j_3-m_1+1-m)(j_3+m_1+m)},\\
V(m)&=t_2+t_3-t_1-2m(m+m_1).
\end{aligned}
$$

Then

$$
U(m+1)f(m+1)+V(m)f(m)+U(m)f(m-1)=0.
$$

The normalized row obeys

$$
(2j_1+1)\sum_{m=m_{\min}}^{m_{\max}}f(m)^2=1,
\qquad
\operatorname{sgn}f(m_{\max})=(-1)^{j_2-j_3-m_1}.
$$

For the [minimal booster coefficient](boosters.md#minimal-coefficients-and-logarithmic-scaling),
the required row is

$$
w_m=\begin{pmatrix}
j_1&j_2&J\\m&k-m&-k
\end{pmatrix},\qquad
\max(-j_1,k-j_2)\le m\le\min(j_1,k+j_2).
$$

A cyclic column permutation brings this into the form above and introduces
no 3j phase. The label $k$ here belongs to the virtual SL(2,C)
representation inside the booster, and $J$ is its coupled SU(2) spin.
The row depends only on discrete labels, so it can be reused at all
quadrature nodes for that seed geometry. The seed's magnetic sum multiplies
these real 3j weights by the complex Gamma-ratio terms.

Thus “seed-free” describes the Wigner recurrence: it requires no externally
computed Wigner value. The Lorentzian $\kappa$ recurrence still needs its
minimal coefficient, which uses the generated 3j row.

## 3j sequences across coupling channels

When the magnetic numbers are fixed and many coupled spins are needed, the
library instead generates

$$
g(j)=\begin{pmatrix}
j&j_2&j_3\\-m_2-m_3&m_2&m_3
\end{pmatrix},
$$

with

$$
j_{\min}=\max(|j_2-j_3|,|m_2+m_3|),\qquad
j_{\max}=j_2+j_3.
$$

Writing $M=m_2+m_3$, the recurrence coefficients are

$$
\begin{aligned}
A(j)&=\sqrt{
\bigl[j^2-(j_2-j_3)^2\bigr]
\bigl[(j_2+j_3+1)^2-j^2\bigr]
\bigl[j^2-M^2\bigr]},\\
X(j)&=jA(j+1),\qquad Z(j)=(j+1)A(j),\\
Y(j)&=(2j+1)\bigl[M(t_2-t_3)-(m_2-m_3)t(j)\bigr].
\end{aligned}
$$

They enter $X(j)g(j+1)+Y(j)g(j)+Z(j)g(j-1)=0$.
At $j_{\min}=0$, the first forward ratio is obtained from its finite
limit,

$$
\frac{g(1)}{g(0)}=\frac{m_2}{\sqrt{j_2(j_2+1)}}.
$$

This case has $j_2=j_3$ and $m_2=-m_3$; the ratio is needed only
when the row has more than one entry. In particular, $m_2=0$ gives an
exactly vanishing next entry. Orthogonality and phase now read

$$
\sum_{j=j_{\min}}^{j_{\max}}(2j+1)g(j)^2=1,
\qquad
\operatorname{sgn}g(j_{\max})=(-1)^{j_2-j_3+m_2+m_3}.
$$

Both 3j variants are used in [coherent boundary states](coherent-states.md).
After rotating the first spinor to the north pole, the first pair's fixed
magnetic data permit one spin sequence across the admissible intertwiner
channels. The second pair supplies a magnetic sequence for each retained
channel. Reusing these rows organizes the coefficient calculation around
small matrices rather than repeated scalar symbol evaluations.

## From sequences to supported tensors

The recurrence variable is chosen to fit the surrounding tensor loops; it
need not be the network's sliced index $x$. Symmetries of the 6j symbol
allow an internal intertwiner to occupy the first slot of $h(j)$.
For example, packed wing generation can sweep $k_5$ or $k_2$ while
holding $x$ and the relevant virtual spins $\ell_{ab}$ fixed.

The CPU reuses coefficient structure as well as output workspaces. The two
pair factors forming $E(j)$ depend on fewer labels than the complete 6j
symbol, so reusable pair-factor sequences are retained in a thread-local
cache. Each factor carries a small correction to its leading floating-point
value. Prepared recurrences also separate the diagonal into a part
independent of $j_4$ and a part proportional to $j_4(j_4+1)$.
Changing that fixed label then reuses the off-diagonal coefficients. These
are coefficient caches, rather than tables of complete Wigner symbols.

Each generated sequence is normalized on its full admissible range. The
filler then selects the intersection with the requested channel support,
applies the required dimension and recoupling-phase factors, and writes the
result into a locally dense tensor block. The resulting $S_A,\ldots,S_E$
are weighted tensors, not bare 6j symbols. Their supports also restrict the
booster outputs and the later contraction, as described in
[Theory and conventions](theory.md#supports-and-exact-structural-pruning).

## Generating the 6j tensors on the GPU

The 6j recurrences can run directly on the GPU. The coherent and full-tensor
GPU routes use this to minimize host-to-device transfers: the host supplies
the boundary labels, channel axes and other input data, while the large 6j
arrays are generated where they will be consumed. The GPU uses the same
admissible sequences, endpoint ratios, zero-endpoint branch, orthogonality
normalization and phase convention as the CPU.

Within a device block, compatible recurrences share prepared coefficients
and keep their working sequences in shared memory. The coefficient
calculation factors differences of squares to retain exact triangle-boundary
zeros and uses compensated square roots. Normalization and tensor weights
are combined when values are stored in the required device layout.

Generation follows the contraction's slices and tiles. Boundary wing
tensors are produced for the active $x$ slice, and the central $S_C$
tensor is produced in bounded virtual-spin slabs. In the full-tensor route,
some generators multiply newly computed 6j values by their booster values
before writing a contraction operand. This also avoids materializing and
reading a separate intermediate array. Device generation therefore reduces
both transfer volume and, where fusion applies, device-memory traffic.

The 3j rows used to construct booster seeds and coherent coefficients are
currently generated on the CPU. Booster data and coherent input vectors
still have to reach the GPU; device-side 6j generation does not eliminate
every host-to-device transfer. The subsequent use of these inputs is
described in [Contraction algorithms](contractions.md).
