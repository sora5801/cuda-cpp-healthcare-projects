# THEORY — 6.20 Coronary Autoregulation & Microvascular Perfusion

> The deep dive. Written for a reader who knows C++ but is new to CUDA and new to coronary physiology.
> Read [`README.md`](README.md) first for the overview; read this for the *why*.
>
> _Educational only — not for clinical use._

---

### Catalog entry (for traceability)

### 6.20 Coronary Autoregulation & Microvascular Perfusion 🟡 · Active R&D

- **Deep dive:** Coronary blood flow is regulated by metabolic (adenosine), myogenic, and neural mechanisms
  operating across scales from capillaries (5 µm) to epicardial arteries (4 mm). GPU simulation of a
  microvascular network with 10⁴–10⁶ vessel segments requires solving a large sparse linear system (network
  Poiseuille flow) coupled to oxygen transport and auto-regulatory feedback ODEs. Real-time coronary
  perfusion models support fractional flow reserve (FFR) virtual assessment for stenosis evaluation.
- **Key algorithms:** network Poiseuille flow (sparse linear system), convection–diffusion oxygen transport,
  Green's-function tissue transport, myogenic/metabolic regulation ODE, 1-D structured-tree Windkessel
  outlet, FFR virtual computation, Fåhræus–Lindqvist effect.
- **CUDA pattern:** cuSPARSE for the network-flow linear system (sparse SPD); cuSPARSE SpMV for iterative
  CG; one thread per vessel segment for the transport update.

---

## 1. The science

The heart perfuses **itself** through the coronary arteries. Blood enters from the aortic root at
~100 mmHg, flows down epicardial arteries (~4 mm) that branch repeatedly into arterioles and finally
capillaries (~5 µm), where oxygen is delivered, and returns via venules and veins at a low back-pressure
(~5–20 mmHg). Two facts make this a rich modeling problem:

1. **Scale span.** Vessel radii span ~three orders of magnitude, and because hydraulic resistance scales as
   `1/r⁴`, the *arterioles* (not the big arteries) set most of the resistance — and therefore control flow.
2. **Autoregulation.** The myocardium demands roughly constant oxygen regardless of perfusion pressure. The
   arterioles achieve this by **actively changing their radius**: they dilate (metabolic signals such as
   adenosine, plus a myogenic response to wall tension) when flow is too low and constrict when it is too
   high. Because flow ∝ `r⁴`, a modest ±20% radius change swings flow by roughly ±2×.

A **stenosis** (a narrowing, e.g. from atherosclerotic plaque) raises the resistance of one branch, dropping
the pressure and flow downstream. Clinically this is quantified by **Fractional Flow Reserve (FFR)** — the
ratio of distal-to-lesion pressure to aortic pressure under maximal vasodilation; `FFR < 0.80` indicates a
flow-limiting lesion that may warrant a stent. *Virtual* FFR from an image-derived model is an active
research area (e.g. HeartFlow), and is the target read-out of this project's model class.

## 2. The math

**Poiseuille's law.** For steady laminar flow of a viscous fluid through a rigid cylinder, the volumetric
flow `Q` is proportional to the pressure drop:

```
Q = G · (p_a − p_b),      G = π r⁴ / (8 μ L)
```

`G` is the **conductance** (inverse resistance); `r`, `L` are radius and length; `μ` is viscosity. The `r⁴`
dependence is the crux — it is why arterioles dominate and why autoregulation is so effective.

**Non-constant viscosity (Fåhræus–Lindqvist).** In vessels below ~300 µm diameter, red cells stream in the
axial core leaving a cell-free plasma layer at the wall, so the *apparent* viscosity drops as the vessel
narrows (down to ~10 µm). We use a smooth, bounded, hematocrit-dependent multiplier on plasma viscosity
(`coronary.h::fahraeus_lindqvist_factor`) — a simplified stand-in for the classic Pries in-vivo fit.

**Flow conservation → a linear system.** At every interior junction node `i`, the flows in must equal the
flows out (Kirchhoff's current law for a resistor network):

```
Σ_{segments s at i}  G_s · (p_i − p_j)  =  0
```

Expanding, for interior rows this is the weighted **graph Laplacian** `L`:

```
L_ii = Σ (incident conductances),     L_ij = −G_ij  (for a segment between i,j),     L p = b
```

`L` is **symmetric**. Nodes with a prescribed (Dirichlet) pressure — the aortic inlet and the venous outlets
— are **eliminated**: we pin their rows to the identity and move their `G·p_fixed` contributions to the
right-hand side `b`. With at least one pinned node the interior block of `L` is **symmetric positive
definite (SPD)** and diagonally dominant, which is exactly the class **Conjugate Gradient** is built for.

**Autoregulation as a fixed-point iteration.** After solving for `p` and the per-segment flows `Q_s`, we
nudge each radius toward its metabolic target flow `Q*`:

```
err   = (Q* − |Q_s|) / Q*
r_new = clamp( r · (1 + k · err),  r_min,  r_max )
```

Re-solving with the new radii and repeating drives the network toward a self-consistent regulated state
(a fixed point). This is a deterministic proportional-feedback **surrogate** for the coupled metabolic +
myogenic ODEs (§7).

**Virtual FFR.** `FFR = (P_d − P_v) / (P_a − P_v)`, where `P_a` is aortic (inlet), `P_d` is distal to the
lesion, and `P_v` is venous (the lumped outlet). Healthy ≈ 1.0; flow-limiting `< 0.80`.

## 3. The algorithm

Outer loop over `n_autoreg` autoregulation steps; inner loop is Conjugate Gradient:

```
for k in 0 .. n_autoreg-1:
    G_s      = π r_s⁴ / (8 μ_eff(r_s,hct) L_s)        # O(S)  per-segment conductance
    (L, b)   = assemble boundary-eliminated Laplacian  # O(nnz)
    p        = CG(L, b, x0 = previous p)               # O(iters · nnz)
    Q_s      = G_s (p_a − p_b)                          # O(S)
    if not last: r_s = autoregulate(r_s, Q_s, Q*_s)     # O(S)
```

**Conjugate Gradient** (Hestenes–Stiefel / Shewchuk form), for SPD `L`:

```
r = b − L x ;  d = r ;  rr = rᵀr
repeat:
    Ld    = L d                      # the SpMV — the expensive step
    α     = rr / (dᵀ Ld)
    x    += α d ;  r −= α Ld
    rr'   = rᵀr ;  β = rr'/rr ;  d = r + β d ;  rr = rr'
until rr ≤ tol²·‖b‖²
```

**Complexity.** Let `N` = nodes, `S` = segments, `nnz ≈ N + 2S`. Each CG iteration is one SpMV (`O(nnz)`)
plus a few `O(N)` vector ops. CG converges in at most `N` iterations exactly, and in `O(√κ)` iterations to a
tolerance, where `κ` is the condition number. So one solve is `O(iters·nnz)`; the whole run is
`O(n_autoreg · iters · nnz)`. Warm-starting each solve from the previous `p` makes later solves converge in
~0 iterations. The serial CPU reference has the same asymptotic cost; the GPU parallelizes the `O(nnz)`
SpMV across nodes.

## 4. The GPU mapping

**Storage: CSR.** The GPU stores `L` in **Compressed Sparse Row** format — three arrays: `row_ptr[N+1]`
(where each row's nonzeros begin), `col_idx[nnz]` (the column/neighbor of each nonzero), and `val[nnz]`
(the value). One network node = one CSR row. Sparsity is fixed by topology (radii change *values*, not
*structure*), so we allocate the CSR once and only re-upload `val`+`b` each autoregulation step.

**The SpMV kernel** (`csr_spmv`) is the heart of the parallelism:

```
thread i  (i = blockIdx.x*blockDim.x + threadIdx.x)  owns row/node i
    acc = 0
    for k in [row_ptr[i], row_ptr[i+1]):   # this node's incident nonzeros
        acc += val[k] * x[col_idx[k]]        # gather x at the neighbor
    y[i] = acc
```

- **Thread → data mapping:** thread `i` computes output `y_i` for node `i`. Rows are independent →
  embarrassingly parallel. This is precisely what `cusparseSpMV` does internally (§7).
- **Memory hierarchy:** `val` and `col_idx` are read once, streamed from **global** memory; within a row the
  reads are contiguous (coalesced across the row). The `x[col_idx[k]]` load is a **gather** — the classic
  SpMV access pattern, and the reason SpMV is *memory-bandwidth bound*, not compute bound. On a real network
  a reordering (e.g. RCM) improves gather locality; we skip it for clarity. Dot-product reductions use
  **shared memory** (`__shared__ double s[TPB]`) for the intra-block tree reduction.
- **Launch config:** 256 threads/block (`TPB`), `⌈N/256⌉` blocks — a good occupancy default on sm_75…sm_89.

**Grid/block decomposition (the CG inner loop):**

```
   nodes:   0   1   2   3   4   5   6   7 ...            N-1
            |   |   |   |   |   |   |   |                 |
 csr_spmv  t0  t1  t2  t3  t4  t5  t6  t7  ...   (one thread per node/row)
            \______ block 0 (TPB threads) ______/  ...  block ⌈N/TPB⌉-1

 dot(a,b): [ block 0 → partial[0] ] [ block 1 → partial[1] ] ...   (tree-reduce in shared mem)
                         \___________ reduce_final (1 block, fixed order) → scalar ___________/
```

**Dot-products** (`dᵀLd`, `rᵀr`) are **reductions**, done in two deterministic stages: (1) `dot_partial`
reduces each block's products in shared memory (tree reduction) to one partial sum; (2) `reduce_final` sums
those partials in a *single block* in fixed index order. The scalars `α`, `β` are computed by tiny 1-thread
kernels and kept **in device memory**, so the AXPY kernels (`axpy`, `axmy`, `xpby`) read them by pointer
without a device→host round-trip on the hot path.

**Why not one kernel for the whole CG?** Each CG step has a data dependence (`α` needs the full `dᵀLd`
reduction before the AXPYs), and a global sync across all blocks is only available via separate kernel
launches (or cooperative groups). For teaching clarity we launch one kernel per BLAS-level operation; a
production solver would fuse them and/or use cooperative-group grid sync.

**Which library does what (no black boxes).** The catalog suggests **cuSPARSE** for the SpMV and the sparse
solve, and **Thrust** for the per-segment PDE. We hand-roll `csr_spmv` (equivalent to `cusparseSpMV` on a
CSR descriptor) and the reductions (equivalent to `thrust::inner_product` / `cub::DeviceReduce`) so every
FLOP is visible. To swap in cuSPARSE: create a CSR matrix descriptor with `cusparseCreateCsr`, a dense
vector descriptor with `cusparseCreateDnVec`, size a workspace with `cusparseSpMV_bufferSize`, then call
`cusparseSpMV(handle, OP_NON_TRANSPOSE, &one, matL, vecX, &zero, vecY, CUDA_R_64F, ALG_DEFAULT, buffer)`.

**A note on preconditioning.** Real coronary Laplacians are ill-conditioned (radii span 10³ ⇒ conductances
span 10¹²). A **Jacobi** (diagonal) preconditioner `M⁻¹ = diag(L)⁻¹` costs one extra `O(N)` kernel per
iteration and dramatically cuts iterations — a natural exercise (README §Exercises).

## 5. Numerical considerations

- **Precision: FP64.** Conductances span ~12 orders of magnitude (`r⁴` with `r` from 4→40 µm), so the
  Laplacian is stiff. Double precision keeps CG stable and the pressures accurate; FP32 would lose the
  small-vessel contributions.
- **Determinism (PATTERNS.md §3).** Floating-point addition is not associative, so a naive `atomicAdd`
  reduction over blocks would give run-varying sums and a non-reproducible CG trajectory. We instead reduce
  in a **fixed order** (per-block tree reduction → single-block final sum), so stdout is **byte-identical
  every run** and across Debug/Release (verified). No `atomicAdd` is used in the solve.
- **CPU vs GPU agreement.** Both sides run the *same* FP64 CG and call the *same* physics in `coronary.h`.
  They differ only in **summation order**: the CPU's matrix-free SpMV accumulates a node's contributions in
  *edge* order, while the GPU's CSR-SpMV accumulates them in *CSR-row* order. That reorders FMAs and diverges
  by ~`1e-14` per op, which can drift over the CG iterations. We verify pressures agree within **`1e-6 mmHg`**
  — in practice we see ~`5e-14 mmHg`, but the tolerance honestly reflects the reordering (PATTERNS.md §4).
- **Large intermediate, small headline.** The un-regulated network is grossly over-perfused, so the
  pre-autoregulation perfusion is ~`8e10` (units) and its low-order digits are FMA-sensitive; we print it in
  scientific notation at modest precision. The *regulated* perfusion is small and prints exactly — it is the
  headline number.

## 6. How we verify correctness

Three independent checks:

1. **CPU ≡ GPU.** `main.cu` solves the network on both paths and asserts `max_i |p_cpu[i] − p_gpu[i]| ≤ 1e-6
   mmHg`. An independent serial implementation agreeing with the parallel one is strong evidence: a shared
   bug would have to corrupt both the edge-order CPU assembly and the CSR-order GPU assembly *identically*,
   which is far less likely than either being individually wrong. This catches divergence in assembly, CG,
   or the physics core.
2. **CG residual.** Each solve reports its final residual `‖b − Lp‖`, confirming the linear system is
   actually solved (not just that two solvers agree on a wrong answer).
3. **Physical sanity.** Pressures fall monotonically from the 100 mmHg inlet to the 20 mmHg outlets; the
   stenosed branch's distal pressure (and hence virtual FFR) is depressed, flagging it as flow-limiting;
   autoregulation moves total perfusion toward the metabolic set-point. These are visible in the
   deterministic stdout and are the "science, not just CPU==GPU" check (PATTERNS.md §4).

The tolerance and the reasoning behind it are stated in `main.cu` (`TOLERANCE`) and §5 above.

## 7. Where this sits in the real world

- **cuSPARSE / cuSOLVER.** A production code would store `L` with `cusparseCreateCsr` and call
  `cusparseSpMV` (which our `csr_spmv` reimplements) inside a preconditioned CG — or hand the whole solve to
  a sparse direct/iterative library. We hand-roll the SpMV so the mechanism is visible; swapping in cuSPARSE
  is a README exercise and gives identical results.
- **Boundary conditions (SimVascular).** Real coronary models terminate each outlet with a **structured-tree**
  or **lumped-parameter Windkessel** (RCR) that captures downstream compliance and the impedance spectrum;
  our single fixed venous pressure is the zeroth-order version.
- **Autoregulation.** The literature couples a **metabolic** signal (tissue O₂ / adenosine) and a **myogenic**
  response (smooth-muscle tone vs. wall tension) as ODEs on each segment, often with time delays and a
  pressure–diameter set-point curve. Our proportional feedback is a deterministic teaching surrogate that
  shares the key qualitative behavior (flow driven toward a set-point via `r⁴`).
- **Oxygen transport.** Delivery is modeled by convection–diffusion along each segment coupled to a
  **Green's-function** method for tissue O₂ (Secomb et al.) — the natural next layer (README §Exercises), and
  another sparse linear solve, which is why the APBS electrostatics solver is a listed analogy.
- **Scale.** Image-derived coronary trees have 10⁴–10⁶ segments; that is where the GPU SpMV's parallelism
  dominates and where our tiny launch-bound demo is only a pedagogical stand-in.

---

## References

- Poiseuille (1840s); Pries, Secomb & Gaehtgens, *Blood viscosity in tube flow* — the Fåhræus–Lindqvist fit
  our viscosity model simplifies.
- Shewchuk, *An Introduction to the Conjugate Gradient Method Without the Agonizing Pain* (1994) — the
  clearest CG derivation; our solver follows its notation.
- Secomb et al., Green's-function methods for oxygen transport in microvascular networks — the O₂ extension.
- Taylor, Fonte & Min, *Computational FFR from coronary CTA* (2013) — the virtual-FFR clinical context.
- SimVascular / svFSI, HemeLB, APBS, OpenFOAM — see README "Prior art & further reading" for what to learn
  from each.
