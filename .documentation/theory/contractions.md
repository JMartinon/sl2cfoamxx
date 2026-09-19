<!-- SPDX-FileCopyrightText: 2026 Jeremy Martinon -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Contraction algorithms

The vertex is a [tensor network](theory.md#vertex-network-figure) of five
weighted SU(2) 6j tensors and four booster tensors. Its contraction order depends on the output: a fixed
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

The following table identifies where the arithmetic runs. A matrix product in
an equation does not necessarily correspond to a BLAS call: small supported
rows often benefit from a custom loop, whereas retained boundary axes make
larger dense products worthwhile.

| Output and backend | Wing and middle operations | Final accumulation |
| --- | --- | --- |
| Fixed component, CPU | Reused pair products; supported AXPY and dot kernels | Real scalar reduction |
| Coherent scalar, CPU | Complex row kernels with separate real and imaginary arrays | Bilinear complex dots and $C_1$ weighting |
| Full tensor, CPU | Supported triple dots and rank-one updates; custom loops or DGEMM for the central factor | Rectangular DGEMM or tiled custom kernels; custom triangular kernel with reflection |
| Coherent scalar, GPU | Two real DGEMMs for $K$; binary complex cuTENSOR for $X_2,X_3$; one real DGEMM for $X_4$ | Fused support-aware warp kernel and scalar reductions |
| Full tensor, GPU | Recurrence-fused operands; binary real cuTENSOR wings; FP64 WMMA or SIMT panel producers | Strided-batched DGEMM and optional compact reflection fold |

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

## From supported tensors to matrix kernels

Supports describe exact admissibility, including parity. A kernel intersects
the relevant channel ranges before entering its arithmetic loop. Within a
surviving block, the code uses ordinary contiguous arrays; it does not attach
a sparse index to every value. Such a block can still contain zeros from
other constraints or from the numerical value of a symbol. Some GPU operands
are deliberately padded to make their inner loops regular. Support pruning
and dense matrix arithmetic therefore coexist.

The main elementary operations are a scaled vector update, usually called
AXPY, and a dot product:

$$
y_h\leftarrow y_h+\alpha z_h,
\qquad d=\sum_h y_hz_h.
$$

A CPU *microkernel* here is a short loop that applies these operations, or
updates a small output tile, while reusing values in registers. Most CPU
microkernels use restricted pointers and OpenMP SIMD directives. They are
portable C++ loops: the compiler chooses the emitted vector instructions.
OpenMP SIMD within a loop is distinct from OpenMP threads processing different
blocks.

For larger matrices, DGEMM denotes the double-precision real BLAS operation

$$
G\leftarrow\alpha\,\operatorname{op}(L)\operatorname{op}(R)+\beta G.
$$

A transpose flag changes how an operand is read; it need not transpose the
stored array. A leading dimension is the physical pitch between columns in
a column-major view, or rows in a row-major view. It can exceed the number of
rows or columns currently used. Strided batching repeats the same matrix
operation at fixed offsets in the operands. These are the standard
[DGEMM](https://www.netlib.org/lapack/explore-html/dd/d09/group__gemm_ga1e899f8453bcbfde78e91a86a2dab984.html)
and [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html) conventions.

The CPU calls go through the private `CpuBlas.hpp` wrapper: Accelerate on
macOS, and a compatible CBLAS provider such as OpenBLAS, BLIS, or MKL on other
platforms. BLAS dimensions use the LP64 interface; the values are FP64.
cuBLAS handles the GPU matrix products. cuTENSOR handles contractions with
more general index and stride patterns; its descriptors identify contracted
indices and indices retained in both operands. The application selects the
contraction graph and layouts, while the vendor library selects its internal
kernel. See the [cuTENSOR user guide](https://docs.nvidia.com/cuda/cutensor/latest/user_guide.html)
for that descriptor model.

## Fixed component on the CPU

### Wing products and reuse

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

At fixed $(\ell_{35},\ell_{45})$, form the scratch panel

$$
P_5(k_5,\ell_{25})=
S_A(x,i_1,k_5,\ell_{25})B_5(i_5,k_5,\ell_{25},\ell_{35},\ell_{45}).
$$

Its $\ell_{25}$ axis is contiguous. The panel is initially zeroed and filled
on the intersection of the $S_A$ and $B_5$ supports. For each $k_4$, a row of
$T_1$ is then a sum of contiguous $P_5$ rows, scaled by the corresponding
$S_B$ value. The implementation performs these sums with supported AXPY
loops. Computing $P_5$ once reuses the multiplication $S_AB_5$ across all
$k_4$ values.

The other wing similarly reuses $P_2=S_EB_2$ across $k_3$. OpenMP can
distribute the independent $(\ell_{35},\ell_{45})$ and
$(\ell_{23},\ell_{24})$ blocks with dynamic scheduling. Each worker owns its
pair-product scratch; it does not allocate a new temporary for every scalar
symbol.

### Local contraction around the central 6j block

Let $\boldsymbol q=(\ell_{24},\ell_{34},\ell_{35})$ denote a local block,
and let $C_{\boldsymbol q}$ be the corresponding $S_C$ matrix with rows
$k_3$ and columns $k_4$. The next stages are

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

Again, rows run along $\ell_{25}$. Each row builder intersects booster and
wing supports and accumulates only the allowed intervals. It constructs
panels for the current $\boldsymbol q$, not global tensors containing every
$\boldsymbol q$ simultaneously. The block contributes

$$
z_{\boldsymbol q}=
\sum_{\ell_{25},k_3,k_4}
T_4(k_3,\ell_{25})C_{\boldsymbol q}(k_3,k_4)T_2(k_4,\ell_{25}).
$$

Write $r_3,r_4$ for the local $k_3,k_4$ support lengths and $N_{25}$ for
the $\ell_{25}$ extent. Two associations compute this scalar:

| Association | Projected panel | Final operation |
| --- | --- | --- |
| Project onto $k_3$ | $Y=C_{\boldsymbol q}T_2$, shape $r_3\times N_{25}$ | Sum supported row dots between $T_4$ and $Y$ |
| Project onto $k_4$ | $Y=C_{\boldsymbol q}^{T}T_4$, shape $r_4\times N_{25}$ | Sum supported row dots between $Y$ and $T_2$ |

The general-spin path projects onto the smaller support: the first
association when $r_3\leq r_4$, the second otherwise. This reduces the
projected scratch to $\min(r_3,r_4)N_{25}$ values. The opposite panel can be
built a row at a time.

The reflection-specialized path also has a tiny-block alternative. It stores
rows on the smaller side and directly evaluates

$$
z_{\boldsymbol q}=\sum_{a,b}
C_{\boldsymbol q}(a,b)\,\langle T_4(a,:),T_2(b,:)\rangle.
$$

This avoids constructing a projected panel when there are very few row pairs.
The current selector uses this pairwise kernel if $r_3r_4\leq16$, or if
$\min(r_3,r_4)\leq2$, $r_3r_4\leq64$, and $N_{25}\leq8$ together.
Otherwise it uses the projection choice above. These cutoffs describe the
implementation, not a mathematical condition on the amplitude.

### Short-row microkernels and reflection

The adaptive real AXPY and dot helpers explicitly expand active intervals of
length one through four. Longer intervals use a contiguous `omp simd` loop;
the dot helper uses a SIMD reduction. The short cases avoid setting up a
general loop for just a few values. All of these are custom kernels: the
fixed-component contraction does not call DGEMM or GEMV.

If the complete boundary and the fixed external intertwiners validate the
reflection described in [Execution and storage](execution.md), the code
visits local-block orbits under
$\ell_{24}\leftrightarrow\ell_{35}$ and $k_3\leftrightarrow k_4$.
A self-reflected block has weight one; a pair of distinct blocks contributes
twice its representative. A support-based cost estimate can choose the
cheaper orientation of that representative. It accounts for row construction,
zeroing, projection work, and the lengths of the final intersected rows.
It does not scan every numerical $C$ entry to make the decision.

Workers reuse their local scratch and reduce block contributions to the
$x$-slice scalar. Input 6j tensors may be resident or produced in slabs;
slicing the contraction does not itself require regenerating resident
inputs. The public fixed-component production contraction is CPU-only.

## Full tensor on the CPU

### Retaining the boundary axes

`compute_full_tensor` preserves all five intertwiner axes. For a fixed
$i_1$, group the other axes into $m=(i_2,i_3)$ and $n=(i_4,i_5)$. If $I_a$
is the number of values on axis $i_a$, the matrix dimensions are
$M=I_2I_3$ and $N=I_4I_5$. Array positions, rather than spin values, flatten
as $m=\widehat i_2+I_2\widehat i_3$ and
$n=\widehat i_4+I_4\widehat i_5$.

The wing equations are the same as above, but $T_1$ now retains $i_5$ and
$T_3$ retains $i_2$. Their seed kernel directly computes a three-factor dot
product. For stored support intervals $[b_h,e_h)$ of the three operands,
it sets

$$
h_{\min}=\max(b_1,b_2,b_3),\qquad h_{\max}=\min(e_1,e_2,e_3),
\qquad d=\sum_{k=h_{\min}}^{h_{\max}-1}a_kb_kc_k.
$$

An empty intersection returns zero. The reduction uses restricted pointers
and `omp simd reduction`; unlike the fixed-component wing, this kernel does
not materialize a reusable pair product. The full-tensor layout and retained
boundary dimensions lead to a different data-access strategy.

Fix $\boldsymbol q$ and abbreviate $p=\ell_{25}$. Define

$$
\begin{aligned}
U_p(k_3,m)&=T_4(k_3,p;i_2,i_3)
=\sum_{\ell_{23}}T_3(k_3,\ell_{23},\ell_{24},p;i_2)
B_3(i_3,k_3,\ell_{34},\ell_{35},\ell_{23}),\\
V_p(k_4,n)&=T_2(k_4,p;i_4,i_5)
=\sum_{\ell_{45}}B_4(i_4,k_4,\ell_{45},\ell_{24},\ell_{34})
T_1(k_4,\ell_{35},\ell_{45},p;i_5).
\end{aligned}
$$

These are built by custom rank-one updates. The $B_4$ panel over
$(\ell_{45},i_4)$ is packed once for a supported $k_4$ and reused over all
$p$: each $T_1$ scalar updates contiguous $i_4$ values. On the other side,
the $B_3$ panel over $(\ell_{23},i_3)$ is reused over $p$; each $B_3$ scalar
updates a contiguous $i_2$ row from $T_3$. Both inner updates use SIMD loops.
The destination panels are zeroed before the sums, and exact zero scaling
values can skip an update.

### Multiplying by the central matrix

The active CPU tree always uses the association

$$
D_p=C_{\boldsymbol q}^{T}U_p,
\qquad
\Delta A=\sum_pD_p^{T}V_p.
$$

Here $C$ has shape $r_3\times r_4$, $U_p$ has shape $r_3\times M$, and
$D_p$ has shape $r_4\times M$. For reflection pairs, the CPU can choose the
member with the narrower logical $k_4$ support. For general spins it does
not switch the middle multiplication to $CV_p$.

The memory views make the first multiplication inexpensive to dispatch:

| Buffer | Storage used by this stage |
| --- | --- |
| $C$ | Column-major $r_3\times r_4$; the same bytes are row-major $C^T$ |
| $U_p$ | Row-major $r_3\times M$, boundary index $m$ contiguous |
| $D_p$ | Row-major $r_4\times M$, boundary index $m$ contiguous |
| $V_p$ | Row-major $r_4\times N$, boundary index $n$ contiguous |

For an ordinary stored $C$ block, the middle BLAS call therefore requires
no packing or physical transpose. It is row-major DGEMM with both operands
marked `NoTrans`, dimensions $(r_4,M,r_3)$, and leading dimensions
$(r_3,M,M)$. The scalars are $\alpha=1$, $\beta=0$, so BLAS overwrites $D_p$
without a preceding destination-zeroing pass. A reflected view of a
half-stored $S_C$ block may first need a logical transpose buffer.

BLAS is selected when the **aggregate dense work over all $p$ slices**

$$
2N_{25}r_3r_4M\geq131072
$$

and the dimensions fit the CBLAS interface. The selected branch then issues
one ordinary DGEMM per $p$; this is not a CPU strided-batched call. Below the
threshold, custom loops clear $D$, skip exact zero $C$ coefficients, and
apply contiguous SIMD updates over $m$.

### Stacking rows for the final update

One local block supplies $N_{25}r_4$ reduction rows, indexed by
$(p,\text{local }k_4)$. The code transposes and packs $D$ and $V$ into
column-major panels $L[P_{\mathrm{cap}},M]$ and
$R[P_{\mathrm{cap}},N]$, with the reduction row contiguous. Several local
blocks can occupy successive row segments. A flush then computes

$$
A\leftarrow A+L^TR,
$$

using only the $P_{\mathrm{used}}$ populated rows. Stacking lengthens the dot
products and reuses each output tile across many local contributions.

The public default batch target is 512 rows. A batch flushes before the next
block would overflow it and at task completion. An individual block larger
than the target grows the buffer to fit: 512 is not a truncation or a hard
limit on a block. The expert setting also permits 128, 256, 1024, and 2048;
zero selects the public default rather than performing cache autodetection.

For general rectangular output, the final dispatch is:

| Condition | Final kernel |
| --- | --- |
| $\min(M,N)\geq8$ and $2P_{\mathrm{used}}MN\geq32768$, with dimensions and pitches representable by CBLAS | Column-major DGEMM, `Trans`/`NoTrans`, $\alpha=1$, $\beta=1$ |
| Smaller work, a skinny boundary matrix, or dimensions outside the BLAS integer range | Custom $4\times4$ output-tile kernel |

The DGEMM dimensions are $(M,N,P_{\mathrm{used}})$; its operand leading
dimensions are both $P_{\mathrm{cap}}$, and its output leading dimension
is $M$. It accumulates directly into the existing output panel.

In the $4\times4$ kernel, each reduction step loads four values from $L$
and four from $R$, then updates sixteen independent accumulators:

$$
g_{ab}\leftarrow g_{ab}+L_{ha}R_{hb},\qquad 0\leq a,b<4.
$$

The $h$ loop uses an OpenMP SIMD reduction over those accumulators. Boundary
columns are padded with zeros to multiples of four, so the arithmetic loop
has no partial-tile tests. Only the final stores check the actual output
bounds. This amortizes loads across several products while avoiding a BLAS
call for very small matrices.

### Reflection and the triangular microkernel

After validating the full reflection symmetry and channel-value maps, the
right boundary columns can be mapped from $(i_4,i_5)$ to the left-axis order
$(i_5,i_4)$. The resulting matrix is symmetric in the two boundary pairs.
It is stored as one upper triangle of $M(M+1)/2$ doubles for each $i_1$.
Equal axis lengths alone do not establish this symmetry.

For a distinct local-block pair, only one representative is contracted. Its
row panels supply both orientations $(L,R)$ and $(R,L)$, giving $Z+Z^T$ in
the mapped coordinates. A self-reflected block supplies the same two
orientations with a factor $1/2$, giving $(Z+Z^T)/2$ and avoiding double
counting. The recoupling phases are already in the input tensors; the fold
does not introduce another phase.

The compact final update uses a custom triangular kernel. It is not a
BLAS symmetric rank-$k$ update: $L$ and $R$ are distinct operands. The
default tile is $4\times4$, with optional $4\times8$ and $8\times4$ shapes.
These hold sixteen or thirty-two accumulators and reuse each loaded value
across a tile. Boundary columns are padded to a multiple of eight; only
stores apply the upper-triangle condition and true boundary limits. No
full square output is required for this CPU update.

### Scheduling and working memory

At fixed $(i_1,x)$, the two wings are cached across the local blocks that
reuse them. Each worker also retains its $U,V,D$ scratch, small packed
booster panels, and final $L/R$ batch. This separates wing reuse from the
short lifetime of the central local panels.

OpenMP tasks own an $i_1$ value and an interval of $x$ slices. Support-based
work estimates allow heavy $i_1$ panels to be split over $x$ when whole
panels would leave workers idle or badly imbalanced. High-cost tasks are
queued first and consumed with dynamic scheduling. The implementation does
not require a separate parallel region for every small output tile.

When tasks for one $i_1$ run on several workers, output panels are allocated
lazily for the worker/$i_1$ pairs actually used, then reduced after the
contraction. A bounded-memory fallback keeps one output panel per worker
and merges under a per-$i_1$ lock. Resident and streamed input plans use the
same reassociated tree. The memory planner and the mapping back to the
logical rank-five tensor are covered in [Execution and storage](execution.md).

## Coherent amplitude on the CPU

### Early booster projection

For coefficients in the vertex's recoupling bases, the coherent amplitude is

$$
A_{\mathrm{coh}}=
\sum_{i_1,\ldots,i_5}
C_1(i_1)C_2(i_2)C_3(i_3)C_4(i_4)C_5(i_5)
A_{i_1i_2i_3i_4i_5}.
$$

The amplitude is multilinear in the five coefficient vectors; it introduces
no conjugation of their entries. Their construction is described in
[Coherent boundary states](coherent-states.md).

The public API accepts these five vectors directly through the
`compute_coherent_amplitude(spins, Dl, gamma, C1, C2, C3, C4, C5)` overload,
or constructs them from the boundary angles. Supplied vectors follow the
complete ascending admissible intertwiner ranges and are used without
normalization. Both forms share the contraction below and the configured C1
pruning policy. Geometric closure checking requires the angle-based form.

`compute_coherent_amplitude` contracts four coefficient vectors while
producing the boosters:

$$
\widetilde B_n(k_n,\ldots)
=\sum_{i_n}C_n(i_n)B_n(i_n,k_n,\ldots),\qquad n=2,3,4,5.
$$

The producer consumes a real booster recurrence slab while it is available
and writes the projected complex values. The contraction therefore loses
four external axes before forming its wings. It need not first build the
full unprojected booster tensors and then read them again to apply the
coefficients.

For each retained $i_1$, the CPU uses the same $T_1,T_3,T_2,T_4$ equations
as the fixed component, replacing $B_n$ by $\widetilde B_n$. The wings and
local panels become complex, while the five 6j tensors remain real.

### Complex row microkernels

The CPU complex buffers use separate contiguous real and imaginary arrays.
A real 6j coefficient $c$ scales both arrays independently. A complex
booster coefficient $\alpha=\alpha_R+i\alpha_I$ updates a row by

$$
\begin{aligned}
y_R&\leftarrow y_R+\alpha_Rz_R-\alpha_Iz_I,\\
y_I&\leftarrow y_I+\alpha_Rz_I+\alpha_Iz_R.
\end{aligned}
$$

These are fused row loops rather than conversions to temporary complex
matrices. The central projection $CT_2$ or $C^TT_4$ applies real coefficients
to the two arrays. Its implementation zeros only the active cover shared by
the two panels, and then updates contiguous $\ell_{25}$ intervals. It can
reuse a stored central block directly or materialize the logical view of a
half-stored block in worker scratch.

The final complex dot is explicitly **unconjugated**:

$$
\begin{aligned}
\Re d&=\sum_h(a_{R,h}b_{R,h}-a_{I,h}b_{I,h}),\\
\Im d&=\sum_h(a_{R,h}b_{I,h}+a_{I,h}b_{R,h}).
\end{aligned}
$$

Both sums are accumulated in one SIMD reduction. Replacing this with a
Hermitian inner product would change the physical contraction. The active
coherent CPU tree uses these custom row kernels, not complex BLAS. Its
local dispatcher chooses the direction of the central projection; it does
not use the fixed-real path's separate tiny pairwise contraction kernel.

### Sharing work across the remaining intertwiner

The real tensors $S_B,S_C,S_D$ do not depend on $i_1$. At each $x$, their
supports and the descriptors for the $B_3/B_4$ row accesses can therefore
be prepared once and shared across a tile of active $i_1$ values. The
$i_1$-dependent $S_A,S_E,T_1,T_3$ remain separate for each tile member.
The tile size is bounded by scratch-memory and cache estimates; it is not
an extra physical cutoff.

In the normal accumulation path, OpenMP tasks flatten the pairs
$(i_1,\boldsymbol q)$. A task computes its local complex scalar, multiplies
by $C_1(i_1)$, and adds it to its worker's sum. This shares preparation and
wing data while leaving the local arithmetic scalar in $i_1$. It does not
form a full rank-five tensor or require a vectorized contraction across
all $i_1$ lanes.

Exactly zero $C_1$ entries and unsupported $(i_1,x)$ combinations can be
omitted. Optional relative pruning of small nonzero $C_1$ entries is a
separate approximation, described in
[Execution and storage](execution.md#numerical-approximations). Memory
pressure can reduce sharing to a streamed calculation with one $i_1$ at a
time. The coefficient contraction and complex arithmetic stay the same.

## Coherent amplitude on the GPU

### A different elimination order

The GPU eliminates $i_1$ before the principal pipeline. To match the device
notation, this section uses the aliases

$$
\begin{aligned}
(a,k,i,j,v)&=(i_1,k_2,k_3,k_4,k_5),\\
(u,q,p,r,s,t)&=(\ell_{23},\ell_{24},\ell_{25},\ell_{34},\ell_{35},\ell_{45}).
\end{aligned}
$$

Here the scalar $q$ means $\ell_{24}$; it is distinct from the CPU local
triple $\boldsymbol q$. Uppercase $N_a,N_k,\ldots$ below denote the extents
of these axes. For each $x$, form

$$
K(v,k,p)=\sum_a C_1(a)S_A(x,a,v,p)S_E(x,a,k,p).
$$

For fixed $q$, the subsequent stages are

$$
\begin{aligned}
H(k,u,i,p)&=S_D(x,i,k,u,q)\widetilde B_2(k,u,q,p),\\
X_2(v,p,u,i)&=\sum_kK(v,k,p)H(k,u,i,p),\\
X_3(v,p,i,r,s)&=\sum_uX_2(v,p,u,i)\widetilde B_3(i,r,s,u),\\
X_4(j,r,s,v,p)&=\sum_iS_C(x,i,j,q,r,s)X_3(v,p,i,r,s).
\end{aligned}
$$

The last contraction is tiled over a subset $\mathcal R$ of the $r$ axis:

$$
W_{xq\mathcal R}=
\sum_{r\in\mathcal R}\sum_{j,s,v,p,t}
X_4(j,r,s,v,p)\widetilde B_5(v,p,s,t)
S_B(x,v,j,s,t)\widetilde B_4(j,t,q,r).
$$

The amplitude is the sum of these tile scalars over $x,q,\mathcal R$.
The stages use the following concrete operations:

| Stage | Kernel or library operation | Reuse |
| --- | --- | --- |
| $S_A,S_B,S_D,S_E$ | Normalized device 6j recurrence | Once per $x$ |
| $K$ | Two real strided-batched DGEMMs, then interleave real/imaginary results | Shared by every $q$ and $r$ tile at that $x$ |
| $H$ | Pointwise real-times-complex kernel | Once per $q$ |
| $X_2$ | Binary complex cuTENSOR, reducing $k$ and retaining shared $p$ | Shared by the $r$ tiles |
| $X_3$ | Binary complex cuTENSOR, reducing $u$ and retaining shared $i$ | Current $r$ tile |
| $S_C,X_4$ | Tile-local device recurrence, then one real strided-batched DGEMM | Consumed by the tail |
| $W$ | Custom support-aware warp kernel and scalar reductions | Only scalar partials survive |

The $H$ kernel resolves one three-factor part of the network by preparing an
operand for a **binary** contraction. The cuTENSOR calls do not take three
input tensors simultaneously.

### Forming K with two real matrix products

For each $p$, weight and transpose $S_A$ into a real matrix of shape
$N_v\times N_a$, first using $\Re C_1$ and then $\Im C_1$. Multiply it by
the real $S_E$ matrix of shape $N_a\times N_k$. Each component uses
`cublasDgemmStridedBatched` with `NoTrans`/`NoTrans`, dimensions
$(N_v,N_k,N_a)$, $N_p$ batches, $\alpha=1$, and $\beta=0$.

The leading dimensions are $(N_v,N_a,N_v)$ and the batch strides are
$(N_vN_a,N_aN_k,N_vN_k)$ doubles. The two real outputs are interleaved into
complex $K[v,k,p]$, with $v$ contiguous. This keeps the 6j inputs real and
reuses the weighting scratch between the two passes.

### cuTENSOR layouts and plan reuse

$H$ is written directly with fast-to-slow axes $[k,u,i,p]$. The cuTENSOR
outputs use $[v,p,u,i]$ for $X_2$ and $[v,p,i,r,s]$ for $X_3$. In
particular, the flattened pair $(v,p)$ is contiguous in $X_3$.
The $B_3$ descriptor views the existing $[i,u,s,r]$ booster storage with
explicit strides, so selecting an $r$ tile does not require a fresh
transpose of the booster.

Plans and workspaces are prepared once for a device contraction and reused
across slices. There is an $X_2$ plan, an $X_3$ plan for a full tile, and,
when necessary, an $X_3$ plan for a shorter last tile. The operations request
FP64 computation and use identity operands, with no conjugation. The
application fixes this graph but does not prescribe the internal cuTENSOR
tile or vendor microkernel.

### X4 as one real DGEMM

The central recurrence writes $S_C[j,i,z]$ directly, with $j$ contiguous
and $z$ flattening the local pair $(r,s)$. For each $z$, the complex $X_3$
array has shape $(N_vN_p)\times N_i$. Its ordinary interleaved complex
storage can also be read as a real matrix with twice as many rows, with
row coordinate $c+2\widehat{vp}$, where $c=0,1$ selects the real or
imaginary component and $\widehat{vp}=\widehat v+N_v\widehat p$.

Thus both components are computed together as

$$
Y_z=S_{C,z}\left(X_{3,z}^{\mathrm{real\ view}}\right)^T,
\qquad
(N_j\times N_i)\,(2N_vN_p\times N_i)^T.
$$

The call is one `cublasDgemmStridedBatched` with `NoTrans`/`Trans`,
dimensions $(N_j,2N_vN_p,N_i)$, and one batch per $z$. Its leading
dimensions are $(N_j,2N_vN_p,N_j)$; its batch strides are
$(N_jN_i,2N_vN_pN_i,2N_jN_vN_p)$ doubles. Again $\alpha=1$, $\beta=0$.

The output is intentionally real storage with axes
$[j,c+2\widehat{vp},z]$. For a given $(j,v,p,z)$, its real and imaginary
values are separated by $N_j$ doubles. It is **not** an ordinary
interleaved complex array with $j$ fastest. This layout puts consecutive
$k_4$ values next to each other for the tail, so no output repacking is
needed. The transpose is ordinary, not conjugate.

### The fused tail microkernel

The tail combines the remaining $p,v,s,t$ sums and the $B_4$ contraction.
It does not allocate an $X_5$ tensor. One CUDA block is one warp of 32
threads, and the grid covers:

- 32 flattened $(j,r)$ positions per block, one owned by each lane;
- up to 128 flattened $(s,v)$ pairs per block;
- eight $t=\ell_{45}$ values per block.

Each lane holds eight complex dot accumulators for its $t$ values. Before
the $p$ loop, triangle-derived descriptors restrict the $p$ interval for
$k_5$, the live $i=k_3$ and $j=k_4$ intervals of the central block, and the
allowed $(k_5,\ell_{35},\ell_{45})$ and
$(k_4,j_{14},\ell_{45})$ combinations. The central $k_3$ interval
is checked for nonemptiness; its sum has already been performed in $X_4$.
Warp ballots skip a pair with no live lanes or no live $t$ values. These are
exact admissibility tests, not a threshold on the size of a tensor entry.

For each surviving $p$, a lane reads its own complex $X_4$ value. The first
up to eight lanes load the corresponding eight $\widetilde B_5$ values,
then warp shuffles broadcast them. Each live lane updates its eight complex
dots with FP64 fused multiply-add operations. After the $p$ sum, the result
is weighted by real $S_B$ and accumulated over the assigned $(s,v)$ pairs.

Shared memory holds a $32\times8$ real plane and a $32\times8$ imaginary
plane, 4096 bytes in total. At the end, each lane multiplies its eight
accumulated values by $\widetilde B_4$ and sums $t$. A shuffle-down
reduction sums the 32 lanes; lane zero writes one complex partial for the
whole block. The calculation keeps the ordinary bilinear complex product
throughout.

The operands are laid out for these accesses. $\widetilde B_5$ is packed once into
separate real and imaginary planes with $t$ contiguous, then $p$, then
$(s,v)$; the original device booster buffer is released after packing.
$S_B$ is repacked after each $x$ recurrence with $j$ contiguous, then
$(s,v)$, then $t$. The $X_4$ layout already matches the lanes and needs
no conversion.

If $N_r^{\mathrm{tile}}$ is the current $r$-tile size, the number of scalar
partials is

$$
\left\lceil\frac{N_jN_r^{\mathrm{tile}}}{32}\right\rceil
\left\lceil\frac{N_sN_v}{128}\right\rceil
\left\lceil\frac{N_t}{8}\right\rceil.
$$

Two additional GPU reductions collapse these partials to a tile scalar:
a 256-thread grid-stride reduction produces at most 1024 block sums, then
one 256-thread block finishes the sum. The tail and these reducers use no
floating-point atomics. A fixed launch has a fixed reduction tree; this
does not imply bitwise equality across different tile sizes, GPUs, or
vendor-library plans.

### Overlapping recurrence, middle, and tail

Real wing symbols are generated once per $x$ on the GPU. The central
$S_C$ recurrence generates only the current fixed-$q$ tile in $r$, in the
layout consumed directly by $X_4$. This avoids uploading full 6j tensor
families and retaining a dense central tensor for every lifted index.

The pipeline has four stream roles:

| Stream | Work |
| --- | --- |
| Preparation | Wing generation, $K$, and per-$q$ $H/X_2$ |
| Central recurrence | Generate the next $S_C$ tile |
| Middle | cuTENSOR $X_3$ and DGEMM $X_4$ |
| Tail | Fused tail and scalar reductions |

Each pipeline slot owns its central slab, $X_4$ slab, support descriptors,
and reduction buffers. A central-ready event allows the middle to read
$S_C$; an $X_4$-ready event allows the tail to start; a slot-free event
prevents the next tile from overwriting an unfinished tail. Additional
ordering protects the shared $X_2/X_3$ buffers and workspace when moving
to the next $q$.

The default three slots allow recurrence, middle, and tail to overlap on
different tiles. Memory planning chooses an $r$-tile size and includes
workspace, slot storage, and temporary booster-repacking peaks. A memory
failure can reduce the slot count from three to two to one, retaining the
same contraction graph. If the minimum tile cannot fit, the call fails
rather than changing the mathematical route.

With multiple GPUs, each worker owns its inputs and persistent workspaces,
and dynamically claims $x$ slices. The host stores each result by $x$
position and combines the scalars in increasing $x$ order, using
`long double` accumulators. No tensor intermediate crosses between GPUs.

## Full tensor on the GPU

### Device-generated wing operands

The full-tensor route retains the boundary axes and uses real arithmetic.
For fixed $x$ and a tile of $i_1$ values, it prepares

$$
\begin{aligned}
W_1(k_5,\ell_{45},i_5,k_4,\ell_{25},\ell_{35})
&=S_B(x,k_5,k_4,\ell_{35},\ell_{45})
B_5(i_5,k_5,\ell_{25},\ell_{35},\ell_{45}),\\
W_3(k_2,\ell_{23},i_2,k_3,\ell_{25},\ell_{24})
&=S_D(x,k_3,k_2,\ell_{23},\ell_{24})
B_2(i_2,k_2,\ell_{23},\ell_{24},\ell_{25}).
\end{aligned}
$$

Recurrence kernels multiply the normalized 6j sequences by the boosters
while those sequences are in shared memory. They write $W_1/W_3$ directly
in the layout required by cuTENSOR. Separate global $S_B$ and $S_D$ tensors
are not materialized for these wing products. The fused producers assign
structural zeros as well as allowed values, avoiding a separate pass to clear the operand.

Binary real cuTENSOR contractions of $W_1$ with $S_A$ and $W_3$ with $S_E$
reduce $k_5$ and $k_2$ and produce $T_1$ and $T_3$. In fast-to-slow order,
their outputs are

$$
\begin{aligned}
T_1 &: [\ell_{45},i_5,k_4,\ell_{25},\ell_{35},i_1],\\
T_3 &: [\ell_{23},i_2,k_3,\ell_{25},\ell_{24},i_1].
\end{aligned}
$$

The $W$ operands are independent of $i_1$ and have no $i_1$ axis. The
current execution nevertheless rebuilds them within each $(x,i_1\text{ tile})$
iteration. Their recurrences are also split over $\ell_{25}$ to
parallelize the larger product writes. This trades some repeated recurrence
work for direct, parallel production of the consumers' operands; it is not
a cache of every bare 6j family.

Left and right wing streams have separate cuTENSOR plans and workspaces.
They can generate their fused operands concurrently and wait for the
boundary $S_A/S_E$ data before contraction. The calls use FP64 computation,
$\alpha=1$ and $\beta=0$.

### Choosing which side receives C

Return to $\boldsymbol q=(\ell_{24},\ell_{34},\ell_{35})$ and $p=\ell_{25}$.
The raw panels $U_p=T_4$ and $V_p=T_2$ have the same meaning as in the full
CPU section: $U_p$ has $r_3$ internal rows and $M=I_2I_3$ boundary columns;
$V_p$ has $r_4$ internal rows and $N=I_4I_5$ boundary columns.

With the mathematical convention $C[k_3,k_4]$, either association is valid:

$$
\begin{array}{lll}
L_p=C^TU_p,&R_p=V_p,&\text{reduction rows per block: }N_{25}r_4,\\
L_p=U_p,&R_p=CV_p,&\text{reduction rows per block: }N_{25}r_3.
\end{array}
$$

Both give $\sum_pL_p^TR_p$. The GPU chooses by estimating the central
transform plus the final dense accumulation:

$$
\begin{aligned}
F_L&=2N_{25}r_3r_4M+2N_{25}r_4MN,\\
F_R&=2N_{25}r_3r_4N+2N_{25}r_3MN.
\end{aligned}
$$

Work common to producing raw $U,V$ is omitted from the comparison. The
right association is chosen only if $F_R<F_L$; ties select left. Thus the
choice depends on retained boundary dimensions as well as internal supports.

The GPU physically stores central blocks with $k_4$ contiguous, as a
$r_4\times r_3$ matrix. Its stored matrix is therefore the transpose of
the chapter's $C$. This explains why a source-level multiplication by the
stored central array can correspond to $C^TU$ in these equations.

Each nonempty local-block descriptor records its support ranges,
association, and row count. Rows are concatenated across descriptors into
left and right slabs. Empty blocks are omitted, and a descriptor stays
intact when a slab is scheduled. This removes unsupported outer blocks
while allowing regular dense inner products.

### FP64 panel-producing microkernels

The custom producers compute the $B_3T_3$ and $B_4T_1$ contractions and,
on the selected side, the central transform. Raw $U/V$ panels are local
working data; the complete global $U$ and $V$ tensors are not stored before
this transformation. The final values are written directly into the
column-major $L/R$ slabs consumed by cuBLAS.

Before these kernels run, $B_3,B_4,T_3,T_1$ are packed with the contracted
$\ell_{23}$ or $\ell_{45}$ coordinate contiguous, boundary intertwiner
axes tiled in groups of eight, and the contracted extent padded to a
multiple of four. Invalid boundary lanes are zero-filled. This arrangement
makes the arithmetic inside each tile regular.

On `sm_80` and newer architectures, a warp computes an $8\times8$
boundary tile with FP64 WMMA fragments. Each matrix instruction advances
through four contracted values: the fragment shape is $8\times8\times4$,
with double-precision inputs and accumulators. The raw $U$ tile multiplies
$B_3$ and $T_3$ panels; the raw $V$ tile multiplies $T_1$ and $B_4$ panels.
These are explicit double-precision matrix operations, without conversion
to TF32 or a lower-precision input format.

A block has eight warps, or 256 threads. For a plain producer, each warp
owns one internal intertwiner value and computes its 64 boundary-pair
values. A shared output tile reorganizes those values for row-contiguous
slab writes. The plain producer reserves 512 doubles, or 4 KiB, of dynamic
shared memory.

The transformed producer has a second stage:

1. Form a panel of up to 64 source intertwiner values by 64 boundary pairs
   in shared memory.
2. Stage a central-matrix tile with up to eight output intertwiner rows.
3. Multiply it by the shared source panel, then scatter the transformed
   values directly into the appropriate $L/R$ rows.

The eight warps divide the 64 boundary columns into groups of eight. The
shared-memory allocation comprises $64\times64$ source values,
$8\times64$ central values, and $8\times64$ output values: 5120 doubles,
or 40 KiB per block. When the source support fits within 64 values, the
source panel is reused for all output-intertwiner tiles. Larger supports
are processed in chunks of 64, accumulating their contributions. The
64-value capacity bounds scratch memory; it does not truncate the sum.

Below `sm_80`, the same layouts and supports feed an FP64 SIMT implementation.
Threads own individual output elements and perform register-accumulated
scalar dots over the contiguous lifted coordinate. Transformed producers
cooperatively build the shared source panel and then perform the
central-weighted dots, with the same 64-source and eight-output tiling.
The older-architecture path uses no FP64 WMMA instructions. cuBLAS and
cuTENSOR still select their own kernels for the separate library stages.

Producer launches use grid-stride work distribution with a default grid
ceiling proportional to eight times the device's multiprocessor count.
That setting limits queued blocks; it is not a promise of eight resident
blocks per multiprocessor. Shared-memory and register usage determine
actual occupancy.

### Materialized slabs and strided-batched DGEMM

Let $H$ be the selected slab row capacity and $K\leq H$ the number of
populated rows. For each $i_1$ in the current tile, the stored operands are
column-major $L[H,M]$ and $R[H,N]$. Let $a$ be the position of $i_1$ within its tile and $h$ a slab row. Their
addresses, with $m,n$ the boundary-pair positions defined above, are

$$
\begin{aligned}
\operatorname{offset}_L(a,h,m)&=aHM+h+Hm,\\
\operatorname{offset}_R(a,h,n)&=aHN+h+Hn.
\end{aligned}
$$

Thus the aggregated reduction rows are contiguous. The final
`cublasDgemmStridedBatched` call uses:

| Parameter | Value |
| --- | --- |
| Operand flags | `Trans`, `NoTrans` |
| Matrix dimensions | $(M,N,K)$ |
| Leading dimensions | $(H,H,M)$ |
| Batch strides, in doubles | $(HM,HN,MN)$ |
| Batch count | Current number of $i_1$ values |
| Product scale | $\alpha=1$ |

This performs one $L^TR$ update per $i_1$. Combining many local-block rows
turns numerous small contributions into fewer longer matrix products.
There is no additional global packing step between the custom producers
and DGEMM.

Two pairs of $L/R$ slabs support overlap: independent left and right
producer streams fill one pair while a GEMM stream consumes the other.
Ready events protect the consumer, and free events protect subsequent
writes. A slab flushes when it becomes full, the next descriptor does not
fit, or the current $x$ slab ends. Rows can accumulate across local work groups
and $x$ values within that slab.

For rectangular output, the first update uses $\beta=0$ and later updates
use $\beta=1$ in the persistent output matrix. For compact output, each
DGEMM uses $\beta=0$ in dense scratch, followed immediately by a reflection
fold into the compact result. The scratch does not need to be zeroed
before a $\beta=0$ call.

### Reflection folding and memory planning

The reflection gate validates boundary spins, actual channel-value maps,
and compatible central supports. When valid, it executes one
representative per local-block orbit. The dense update supplies both the
direct boundary coordinate $(i_2,i_3,i_4,i_5)$ and its reflected coordinate
$(i_5,i_4,i_3,i_2)$ to the compact output. A self-reflected block carries a
factor $1/2$ on the produced left operand to compensate for the two terms
in that fold.

This is a mapped reflection, not a blind transpose of the original
boundary flattening. After the right columns are mapped to the left-axis
order, the result occupies $M(M+1)/2$ values per $i_1$. General inputs use
every supported logical block and retain a rectangular $M\times N$ panel
through the same GPU route.

The planner chooses an $i_1$ tile, an $x$ slab, the $L/R$ row capacity,
and either resident or streamed central 6j storage. Resident $S_C$ can be
reused across $i_1$ tiles; streamed $S_C$ is generated for fixed
$(x,\ell_{24})$ and bounded $\ell_{34}$ tiles. Memory estimates include
both slab pairs, cuTENSOR workspaces, wing operands, output storage, and
temporary peaks during booster expansion and packing. Every selected
capacity must fit the largest indivisible descriptor.

The public default row-capacity request is 8192. Memory-pressure retries
can reduce it to 4096 and 2048 while preserving the graph. These are
capacity choices, not additional cutoffs in the lifted-spin sums. Plan
selection favors fewer GEMM flushes and central-generation calls, then
other measures of repeated work. General and reflection-compatible inputs
both keep their complete supported sums.

Multiple GPUs receive disjoint contiguous $i_1$ ranges. Each device owns
its booster inputs and contracts all required $x$ slices for its range.
Their rank-five output pieces can therefore be assembled without a
cross-device tensor reduction. Within a device, contributions from
successive $x$ slabs are combined into its output range.

## Arithmetic order and performance choices

All these trees compute the same finite-cutoff network with the stated
boundary contraction. Support intersections and validated reflection maps
remove mathematically redundant work. Padding, tiling, batching, and
memory-driven streaming change how the work is executed. Optional
coefficient pruning is a separate numerical approximation.

The BLAS thresholds and tile sizes above are current implementation
choices. They express the balance between dispatch overhead, arithmetic
reuse, and memory traffic; they are not universal performance guarantees.
In particular, outer OpenMP parallelism and any internal BLAS threading
both affect CPU resource use.

Floating-point association differs between a direct dot, a projected
panel, a tiled reduction, and a vendor matrix product. Dynamic CPU task
assignment can also change which partial sums are grouped together. A
fixed final reduction order alone does not make all routes bitwise
identical. Numerical comparisons should use the library's accuracy
criteria rather than require identical bits from different trees or
hardware.

## Implementation map

The source separates preparation, local kernels, scheduling, and library
calls. The following files contain the active implementations described
above:

| Part | Source |
| --- | --- |
| Fixed CPU wing products | [AmplitudePackedContraction.cpp](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/amplitude/AmplitudePackedContraction.cpp) |
| Fixed CPU general local tree | [AmplitudePackedContractionGeneral.cpp](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/amplitude/AmplitudePackedContractionGeneral.cpp) |
| Real row microkernels and selector | [AmplitudePackedContractionQLocal_ReflectionQOrbit.hpp](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/amplitude/AmplitudePackedContractionQLocal_ReflectionQOrbit.hpp) |
| Coherent CPU kernels and scheduling | [AmplitudePackedCoherentI1Batch.hpp](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/amplitude/AmplitudePackedCoherentI1Batch.hpp), [implementation](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/amplitude/AmplitudePackedCoherentI1Batch.cpp) |
| Full CPU BLAS and tile kernels | [CpuContraction.cpp](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/full_tensor/CpuContraction.cpp), [support kernels and options](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/full_tensor/CpuContraction.hpp) |
| CPU BLAS interface and tuning defaults | [CpuBlas.hpp](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/core/CpuBlas.hpp), [LowLevelTuning.hpp](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/core/LowLevelTuning.hpp) |
| Coherent GPU pipeline and library calls | [CoherentGpuBackend.cu](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/gpu/CoherentGpuBackend.cu) |
| Coherent GPU tail and support masks | [CoherentGpuSupportKernels.cuh](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/gpu/CoherentGpuSupportKernels.cuh), [packing and reductions](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/gpu/CoherentGpuKernels.cuh) |
| Full GPU planner, slabs, and DGEMM | [FullTensorSemiFusedGpuBackend.cu](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/gpu/FullTensorSemiFusedGpuBackend.cu) |
| Full GPU WMMA/SIMT producers and fold | [FullTensorSemiFusedGpuKernels.cuh](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/gpu/FullTensorSemiFusedGpuKernels.cuh), [tile constants](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/gpu/FullTensorSemiFusedGpuTypes.cuh) |
| Full GPU wing descriptors | [FullTensorSemiFusedCuTensor.hpp](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/gpu/FullTensorSemiFusedCuTensor.hpp) |
| Shared device recurrences and fused operands | [CoherentGpuRecurrence.cu](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/gpu/CoherentGpuRecurrence.cu) |
| Reflection validation | [FullTensorQReflectionOrbitPlan.cpp](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/full_tensor/FullTensorQReflectionOrbitPlan.cpp) |

The [weighted 6j generator](https://github.com/JMartinon/sl2cfoamxx/blob/main/src/amplitude/AmplitudePackedSJTensors_HalfSjC.cpp)
assigns the phases and dimension factors used in the scalar network.
