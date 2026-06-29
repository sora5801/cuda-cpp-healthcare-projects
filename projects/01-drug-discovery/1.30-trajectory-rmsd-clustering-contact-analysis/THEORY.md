# THEORY — 1.30 Trajectory RMSD, Clustering & Contact Analysis

> The deep "why" behind the code. Read alongside `src/rmsd_core.h` (the math),
> `src/kernels.cu` (the GPU mapping), and `src/reference_cpu.cpp` (the baseline).
> Target reader: comfortable with C++, new to CUDA and to structural analysis.

---

## The science

A **molecular-dynamics (MD) simulation** integrates Newton's equations for every
atom in a molecule (a protein, a ligand–protein complex, a membrane…) and records
the positions every few picoseconds. The result is a **trajectory**: a long
sequence of *frames*, each a complete 3-D snapshot. A modern trajectory can hold
**millions of frames** of **thousands of atoms** — terabytes of coordinates.

The simulation is only half the work; the *insight* comes from **post-analysis**:

- **How much has the structure changed over time?** — measured by **RMSD** (root-
  mean-square deviation) of each frame against a reference (often the starting or
  the experimental structure). RMSD is the universal "distance between two
  conformations" in structural biology.
- **Which conformational states does the molecule visit?** — found by **clustering**
  frames so that members of a cluster are mutually similar (small pairwise RMSD).
  Clusters are the metastable states; the transitions between them are what a
  **Markov state model** later quantifies.
- **What holds the fold together, and when does it break?** — answered by
  **contact analysis**: which atom pairs are close, and what **fraction of the
  native contacts** survive in each frame (the coordinate **Q**, ≈1 when folded,
  →0 when unfolded).

These three — RMSD, clustering, contacts — are the bread-and-butter of trajectory
analysis (MDTraj, MDAnalysis, GROMACS `gmx cluster`). This project implements the
GPU-friendly heart of each on a small, controlled, **synthetic** example.

---

## The math

### RMSD after optimal superposition

Two structures `X = {x_i}` and `Y = {y_i}` (each `N` atoms, `i = 1..N`) should be
compared *only after removing rigid-body motion* — a molecule that merely tumbled
has not "changed". So we minimize over all rotations `R` (and the translation,
handled by centering on the centroid):

```
RMSD(X, Y) = sqrt( min_R  (1/N) Σ_i | R (x_i − x̄) − (y_i − ȳ) |² )
```

This is the **orthogonal Procrustes / Kabsch** problem. Let `x'_i = x_i − x̄`,
`y'_i = y_i − ȳ` be the centered coordinates. Define:

- `G = Σ_i (|x'_i|² + |y'_i|²)` — the two structures' inner products (their
  Frobenius norms squared), and
- `M = Σ_i x'_i (y'_i)ᵀ` — the **3×3 cross-covariance** matrix.

Kabsch solves it via the SVD `M = U Σ Vᵀ`, giving `R = V diag(1,1,d) Uᵀ` with
`d = sign(det(VUᵀ))` (the `d` fixes reflections). The minimized value is

```
RMSD = sqrt( (G − 2 Σ_k σ_k') / N ),   σ_k' = signed singular values of M.
```

### QCP: replacing the SVD with one eigenvalue

Theobald's **Quaternion Characteristic Polynomial** (QCP, 2005) shows that the
optimal `Σ σ_k'` equals the **largest eigenvalue `λ_max`** of the 4×4 symmetric
**key matrix** built from `M` (entries `S** = M**`):

```
      ┌ Sxx+Syy+Szz   Syz−Szy        Szx−Sxz        Sxy−Syx      ┐
  K = │ Syz−Szy       Sxx−Syy−Szz    Sxy+Syx        Szx+Sxz      │
      │ Szx−Sxz       Sxy+Syx       −Sxx+Syy−Szz    Syz+Szy      │
      └ Sxy−Syx       Szx+Sxz        Syz+Szy       −Sxx−Syy+Szz  ┘
```

so that

```
RMSD = sqrt( max(0, (G − 2 λ_max) / N) ).
```

`K` is symmetric and **traceless**, so its characteristic polynomial has no cubic
term: `p(λ) = λ⁴ + c₂ λ² + c₁ λ + c₀`. We obtain `c₂, c₁, c₀` *exactly* from
`K` with the **Faddeev–LeVerrier / Newton-identity** relations between traces of
powers `tr(Kᵏ)` and the elementary symmetric polynomials of the eigenvalues — only
matrix multiplies and traces, no factorization. Then `λ_max` is the largest root,
found by **Newton's iteration** started at `λ = G/2` (a rigorous upper bound, from
which Newton descends monotonically to `λ_max`).

### Native contacts and Q

A **contact** is a pair `(i, j)` with `|r_i − r_j| < r_cut`, excluding trivial
sequential neighbours (`j > i + sep`). The **native** set is the contacts of the
reference frame. For any frame,

```
Q(frame) = (# native pairs still in contact in `frame`) / (# native pairs) ∈ [0, 1].
```

`Q = 1` means every native contact survives (folded); `Q → 0` means the fold has
dissolved (unfolded).

---

## The algorithm

For each frame `f` (independently):

1. **Center** `f` and the reference on their centroids — `O(N)`.
2. **Accumulate** `G` and the 3×3 covariance `M` in one pass — `O(N)`.
3. **Build** `K(M)`, derive `c₂,c₁,c₀` via Faddeev–LeVerrier — `O(1)` (fixed 4×4).
4. **Newton** for `λ_max` (fixed 50 iterations) — `O(1)`.
5. **RMSD** `= sqrt((G − 2λ_max)/N)` — `O(1)`.
6. **Contacts:** sweep all pairs to compute `Q(f)` — `O(N²)`.

Per frame: `O(N + N²)`. Over `F` frames serially: **`O(F·N²)`**. The
**clustering** step then bins the `F` RMSD values into shells of fixed width — a
trivial `O(F)` reduction that, on this controlled data, recovers the metastable
states (three populated shells separated by empty ones).

> **Why a fixed iteration count?** Step 4 runs exactly 50 Newton iterations with
> *no data-dependent break*. A conditional break would let the CPU and GPU take
> different numbers of steps on borderline frames and diverge. A fixed schedule
> makes both sides execute the identical sequence of FP operations → identical
> bits. 50 iterations is far past double-precision convergence for this quartic.

### Complexity: serial vs. parallel

| | serial (CPU) | parallel (GPU) |
|---|---|---|
| total work | `O(F·N²)` | `O(F·N²)` |
| span (critical path) | `O(F·N²)` | `O(N²)` (all frames at once) |

The frames are mutually independent, so the *span* collapses from `F·N²` to `N²`:
the GPU does one frame's worth of work in the time the CPU does `F`.

---

## GPU mapping

This is the **independent-jobs** pattern (PATTERNS.md §1; the same shape as the
1.12 fingerprint search): **one GPU thread computes one frame**.

- **Thread-to-data map.** Thread `(blockIdx.x, threadIdx.x)` owns frame
  `f = blockIdx.x·blockDim.x + threadIdx.x`, then a **grid-stride loop** advances
  by the total thread count so a modest grid covers an arbitrarily long
  trajectory. See `analyze_frames_kernel` in `src/kernels.cu`.
- **Block size.** 128 threads/block — a multiple of the 32-lane warp, enough warps
  to hide global-memory latency, while each thread carries a fair amount of FP64
  work (a 4×4 eigenvalue + an `N²` sweep), so we keep the block modest.
- **Memory hierarchy.**
  - *Constant memory* holds the **reference frame**: every thread reads it, none
    writes it, and it is identical for the whole launch — the textbook case for the
    constant cache, which **broadcasts** one address to a whole warp in a single
    transaction (`__constant__ double c_ref[N_ATOMS*3]`, 384 bytes ≪ 64 KB bank).
  - *Global memory* holds all frames; each thread streams its own frame's
    coordinates. *Registers* hold the covariance accumulators, `K`, and the Newton
    scalar — the whole per-frame computation lives in registers.
  - **No shared memory, no atomics:** outputs `rmsd[f]`, `qnc[f]` are disjoint, so
    threads never coordinate. (Atomics would only enter if we reduced *across*
    frames, e.g. a pairwise-RMSD matrix — Exercise 1.)
- **The CPU/GPU parity trick (PATTERNS.md §2).** All per-frame math lives in
  `src/rmsd_core.h` as `__host__ __device__` inline functions. `reference_cpu.cpp`
  loops them; `kernels.cu` calls them from one thread. Same source, same FP ops →
  the verification is a real proof, not a fuzzy check.

```
 reference frame ──► [constant memory c_ref] ──broadcast──┐
                                                          ▼
 all frames ─► [global d_coords] ─► thread f ─► kabsch_rmsd(), frac_native_contacts()
                                                          │
                                              rmsd[f], qnc[f] ─► [global] ─► host
```

---

## Numerical considerations

- **Precision: FP64 throughout.** Coordinates, covariance, `K`, the eigenvalue,
  and the contact distances are all `double`. The covariance `G` and `M` are sums
  of products of similar magnitude; in FP32 the cancellation in `G − 2λ_max` would
  lose precision and RMSD libraries (MDTraj included) accumulate in double. We
  follow suit. *Exercise 5* makes the FP32 error visible.
- **Determinism.** Both sides walk atoms in the same order, run the same fixed
  50-iteration Newton, and divide `Q` by the **same precomputed integer**
  `native_total` (computed once on the host and passed to the kernel) — so the
  results are bit-reproducible and the demo's stdout never changes. Timings and
  the (tiny) error go to **stderr**.
- **Clamp.** `(G − 2λ_max)/N` can be a tiny *negative* number at the noise floor
  (a frame identical to the reference); we clamp to 0 before `sqrt` so RMSD is
  never `NaN`.
- **No atomics → no float-summation caveat.** Because each thread writes its own
  outputs, there is no non-associative atomic reduction to worry about (contrast
  5.01 / 11.09, which need integer/fixed-point accumulation for determinism).

---

## How we verify correctness

Three independent layers:

1. **CPU == GPU to ~machine epsilon.** `main.cu` runs both paths and reports
   `max_abs_err`. Because both call the same `__host__ __device__` core in FP64,
   the only difference is the order the compiler schedules the covariance adds;
   measured residual is `rmsd ≈ 5e-14`, `Q = 0` exactly. Tolerance **`1e-9`**
   (PATTERNS.md §4: machine-precision class — a *short* FP64 computation, not a
   long iterative solver, so no `1e-3` physical tolerance is needed).
2. **Against an analytic anchor.** Frame 0 *is* the reference, so its RMSD must be
   **exactly 0** and `Q = 1` — both hold in `expected_output.txt`. The QCP
   `λ_max`/`RMSD` was also cross-checked offline against a brute-force SVD
   superposition and against `numpy.linalg.eigvalsh(K)` over 2000 random structure
   pairs (max error `3e-12`).
3. **Known synthetic shape.** The trajectory is engineered (helix → unfolded
   through three states), so RMSD must rise monotonically and `Q` must fall
   `1 → 0.52 → 0`; the cluster histogram must show three populated shells. The
   demo reproduces exactly that.

---

## Where this sits in the real world

This is a **reduced-scope teaching version**. Production trajectory analysis
differs in scale and breadth:

- **Clustering.** Real GROMOS/DBSCAN/k-medoids clustering operates on the full
  **`F×F` pairwise-RMSD matrix** (the catalog's stated `O(N²)`-over-millions
  bottleneck), often built with cuBLAS via the outer-product distance formulation,
  then fed to RAPIDS cuML. Our 1-D RMSD-shell histogram is a deterministic stand-in
  that conveys the *idea* (group similar frames) without the quadratic matrix.
  Exercises 1–2 build the real thing.
- **More observables.** The catalog also lists **H-bond** networks (donor–acceptor
  distance + angle), the **radial distribution function** (RDF), and the **NMR
  order parameter S²** — all per-frame or per-pair and equally parallelizable; they
  are natural follow-on kernels.
- **I/O-bound reality.** At true scale the limiter is **trajectory-file I/O
  bandwidth** (XTC/DCD/HDF5 from disk), not the arithmetic — which is why RAPIDS
  cuDF and memory-mapped readers matter as much as the kernels.
- **Variable atom counts / selections.** Real tools align on a *selection* (e.g.
  Cα atoms) with per-system `N`; we fix `N_ATOMS` at compile time so the layout is
  static and the inner loops unroll — a didactic simplification.
- **Batched 3×3 SVD.** The catalog mentions custom batched-SVD Kabsch kernels;
  QCP (used here) is the standard *SVD-free* alternative MDTraj ships, chosen here
  precisely because its closed form is easy to make deterministic and CPU/GPU-
  identical.

Despite the simplifications, the **core lessons are production-real**: optimal-
superposition RMSD via QCP, the one-thread-per-frame mapping, constant-memory
broadcast of the reference, and FP64 for legible CPU/GPU agreement.
