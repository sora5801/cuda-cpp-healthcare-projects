# THEORY — 3.15 Hi-C / 3D Genome Contact Analysis

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

Two metres of DNA fit inside a nucleus a few micrometres across. It does not coil
randomly: the genome folds into a reproducible 3D architecture, and that
architecture controls which genes a cell can switch on. **Hi-C** is the assay that
reads this folding out at genome scale.

The wet-lab recipe (Lieberman-Aiden 2009): crosslink the chromatin so loci that are
physically touching get glued together; cut the DNA; ligate the glued ends so a
"touching" pair becomes one hybrid molecule; sequence those hybrids. Each read pair
tells you "locus A was near locus B in 3D". Binning the genome into fixed-size
windows (**bins**) and counting read pairs per bin pair gives the **contact matrix**
`M`: `M_{ij}` = how many times bin `i` and bin `j` were caught together. It is large
(genome_bins × genome_bins), **symmetric** (`M_{ij}=M_{ji}`), and **sparse** (most
distant pairs are never seen).

Two features dominate a Hi-C map:

- **TADs (Topologically Associating Domains):** square blocks along the diagonal —
  contiguous genomic regions that contact *within* themselves far more than across
  their borders. TAD borders often coincide with insulator proteins (CTCF) and gene
  regulatory boundaries.
- **A/B compartments:** a chequerboard at the chromosome scale separating open,
  active chromatin (A) from closed, inactive chromatin (B).

Before any of this biology is readable, the matrix must be **balanced**: raw counts
are inflated for bins that are more mappable, have more restriction sites, or were
simply sequenced deeper. **ICE** removes that per-bin bias. This project implements
**ICE balancing** and then **TAD-boundary detection via the insulation score** — the
two steps that best illustrate the GPU compute pattern.

## 2. The math

**Inputs.** A symmetric sparse matrix `M ∈ ℝ^{n×n}` of non-negative contact counts,
stored as upper-triangle nonzeros `(i, j, M_{ij})` with `0 ≤ i ≤ j < n`.

**ICE objective (matrix balancing).** Find a positive **bias** vector
`b ∈ ℝ^n` such that the corrected matrix

```
    M'_{ij} = M_{ij} / (b_i · b_j)
```

has **equal row sums**: `Σ_j M'_{ij} = s` for all occupied bins `i`, for some
constant `s`. Equivalently, with `D = diag(b)`, we seek `D` making `D⁻¹ M D⁻¹`
doubly stochastic up to scale. This is the classic **matrix-balancing** /
**Sinkhorn–Knopp** problem; for a symmetric matrix the two scalings coincide, so a
single vector `b` suffices.

ICE solves it by fixed-point iteration. Let `r_k = Σ_j M'_{kj}` be the current row
sum of bin `k` and `r̄` the mean row sum over occupied bins. The update is

```
    b_k  ←  b_k · (r_k / r̄)            (multiplicative correction)
```

Intuition: an over-represented bin (`r_k > r̄`) gets a *larger* bias, which divides
its contacts down next iteration; under-represented bins are boosted. Convergence is
measured by the **variance of the row sums** about their mean,
`Var(r) = (1/n_occ) Σ_k (r_k − r̄)²`, which → 0 as the matrix balances.

**Insulation score.** From the balanced matrix, the insulation of bin `k` over a
window of `w` bins is the **mean balanced contact in the diamond** that straddles the
diagonal at `k`:

```
    I(k) = mean over { M'_{ab} : k−w ≤ a < k ≤ b ≤ k+w−1 }
```

i.e. the average strength of contacts that *cross* position `k`. A strong domain
border lets few contacts cross, so `I(k)` **dips** there.

**TAD boundaries.** Bins where `I` is a **strict local minimum** within a search
radius `ρ`:

```
    boundary(k)  ⟺  I(k) < I(k±d)  for all 1 ≤ d ≤ ρ
```

## 3. The algorithm

```
ICE balancing:
  init  b_k = 1 for occupied bins, 0 for empty (masked) bins
  repeat ITERS times:
     (R)  for every nonzero (i,j): add M_{ij}/(b_i b_j) to rowsum[i] (and rowsum[j] if i≠j)
     (U)  r̄ = mean of occupied rowsums;  b_k *= rowsum[k]/r̄;  Var = mean((rowsum-r̄)^2)

Insulation + TADs (on the balanced matrix):
  for every nonzero (i,j), i≠j: add M'_{ij} to the diamond sums of the bins it crosses
  I(k) = sum[k]/count[k]   (NA near the edges)
  boundary(k) = strict local minimum of I within radius ρ
```

**Complexity.** Let `nnz` be the number of stored nonzeros.

| Step | Serial cost | Parallel work / depth |
|---|---|---|
| Row-sum reduction (R) | `O(nnz)` per iteration | work `O(nnz)`, depth `O(log)` via atomics |
| Bias update (U) | `O(n)` per iteration | trivially `O(n)` (kept on host) |
| ICE total | `O(ITERS · nnz)` | the (R) step is the hot loop offloaded to the GPU |
| Insulation | `O(nnz · w)` | work `O(nnz · w)` |

**Arithmetic intensity.** Step (R) is **memory/atomic-bound**, not compute-bound:
each nonzero does one divide, one multiply, one quantize, and 1–2 atomic adds. The
win is parallelism over `nnz`, which at genome scale is `10⁸–10⁹`. The data-access
pattern is a **scatter**: contiguous threads read contiguous entries (coalesced) but
write to scattered row-sum bins (the atomics).

## 4. The GPU mapping

We store the matrix as **struct-of-arrays COO**: device arrays `ei[]`, `ej[]`,
`ecount[]`, each length `nnz`. The hot loop is one kernel:

- **Thread-to-data mapping.** Thread `t = blockIdx.x·blockDim.x + threadIdx.x` owns
  nonzero `t`. It reads `(ei[t], ej[t], ecount[t])`, forms the corrected value, and
  `atomicAdd`s its fixed-point quantum into `acc[ei[t]]` (and `acc[ej[t]]` if
  off-diagonal).
- **Launch configuration.** Block = **256 threads** (a warp multiple; good occupancy
  on sm_75–sm_89). Grid = `ceil(nnz / 256)`. The last block is ragged; a
  `if (t >= nnz) return;` guard retires the extra lanes.
- **Memory hierarchy.**
  - *Global memory:* the COO arrays and the `acc[]` accumulators. The entry reads
    are **coalesced** (consecutive threads → consecutive addresses).
  - *Registers:* `i, j, c, corrected, q` — all per-thread scalars.
  - *Atomics:* the row-sum writes collide (many nonzeros share a row), so they must
    be atomic. We do **not** use shared memory here — the teaching version keeps the
    scatter simple; the per-block-partial-sums optimisation is left as an exercise.
- **No CUDA library.** A production tool would compute the row sum as a **cuSPARSE**
  sparse matrix-vector product, `rowsum = M' · 1` (`cusparseSpMV` on a CSR copy of
  `M'`). We **hand-roll** the SpMV so the reduction is fully visible (CLAUDE.md §6.1.6);
  the cuSPARSE call would compute the identical `rowsum` from a CSR `(row_ptr, col_ind,
  vals)` layout and a ones-vector — see §7.
- **Why the bias update stays on the host.** Step (U) is `O(n)`, negligible, and
  keeping it host-side guarantees the GPU and CPU apply **byte-identical** updates,
  which is what makes verification exact.

```
        COO nonzeros (length nnz)                 row-sum accumulators (length n)
   ei: [ 0  0  1  1  2 ...]   one thread per ↓        acc[0] ← Σ quanta touching bin 0
   ej: [ 0  1  1  2  2 ...]   nonzero                 acc[1] ← Σ quanta touching bin 1
        t0 t1 t2 t3 t4              atomicAdd(q)  ─▶   ...
   block 0  (256 threads)  block 1  ...  block ceil(nnz/256)-1
```

## 5. Numerical considerations

- **Precision: FP64 throughout.** Contacts and biases are `double`. Balancing
  multiplies and divides over dozens of iterations; FP32 would accumulate visible
  rounding. FP64 is the right default for a teaching reference.
- **Determinism — the central lesson.** The reduction (R) sums many contributions
  into each `acc[k]` from threads running in a **nondeterministic order**. Floating-
  point addition is **not associative**, so a *float* `atomicAdd` tally would differ
  run-to-run and would **not** match the serial CPU sum. We fix this exactly as
  flagships `5.01` and `11.09` do: **quantize each contribution to a 64-bit integer**
  (`q = round(M'_{ij} · 10⁹)`, see `hic.h`) and `atomicAdd` the integers. Integer
  addition **commutes**, so the integer tally is identical regardless of thread
  order — **deterministic and bit-identical to the CPU**, which sums the same
  integers. We divide back by `HIC_SCALE` at the end.
- **Fixed-point headroom.** `HIC_SCALE = 1e9` keeps ~9 fractional digits of each
  contribution; a `uint64` holds ~1.8e19, leaving >1e9 headroom for a row's sum —
  ample for matrices whose per-row corrected mass is `O(10³)`. For genome-scale
  data with very deep bins you would lower the scale or accumulate in 128-bit.
- **Masking.** Empty bins carry bias 0; `hic_corrected` returns 0 when either bias is
  0, so masked bins contribute nothing and never produce NaNs/divide-by-zero.

## 6. How we verify correctness

- **Independent CPU reference.** `reference_cpu.cpp` runs the *same* ICE — same
  initialisation, same shared per-element math (`hic.h`), same host bias update — but
  with a single serial loop instead of a parallel scatter. The GPU and CPU therefore
  differ *only* in the order of the reduction.
- **Tolerance.** We compare the per-bin biases and require
  `max |b_cpu − b_gpu| ≤ 1e-9`. In practice the difference is **0** (the fixed-point
  reduction makes both sides sum identical integers, and the host update is then
  deterministic). We still verify to `1e-9` rather than `== 0` because, in principle,
  host-compiler vs nvcc FMA contraction in the `O(n)` host update *could* drift the
  bias by ~`1e-12` over 30 iterations — far below any biological significance
  (`docs/PATTERNS.md §4`).
- **A stronger, scientific check.** Beyond CPU == GPU, the demo recovers the **planted
  biology**: the synthetic sample has TAD borders at bins 4 and 8, and the insulation
  caller reports exactly `boundary at bin 4` and `boundary at bin 8`. That validates
  the whole pipeline end-to-end, not just numerical agreement.
- **Edge cases:** empty/masked bins, the ragged last block (`t >= nnz` guard), and
  diamond windows that underflow the matrix edge (marked NA and skipped).

## 7. Where this sits in the real world

Production Hi-C tooling differs in scale and breadth:

- **cooler / cooltools (open2c)** store the matrix as CSR-like sparse pixels in HDF5
  (`.cool`/`.mcool`), balance with ICE/KR on the CPU (NumPy/SciPy), and compute the
  insulation score with the *exact* diamond definition we mirror. The natural GPU
  next step (the catalog's note) is to move the SpMV onto **cuSPARSE**:

  ```c
  // rowsum = M' · 1 , with M' in CSR (row_ptr, col_ind, vals) and a ones-vector.
  cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
               matM, vecOnes, &zero, vecRowsum, CUDA_R_64F,
               CUSPARSE_SPMV_ALG_DEFAULT, buffer);
  ```

  That call computes the identical `rowsum` our kernel does; we hand-roll it so the
  reduction (and the determinism trick) are visible. Note cuSPARSE uses FP atomics
  internally, so a fully reproducible pipeline still needs the fixed-point idea or a
  deterministic SpMV algorithm.

- **Knight–Ruiz (KR)** balancing (used by Juicer) is a Newton-style balancer that
  converges in fewer iterations than ICE; same goal, faster.

- **A/B compartments** need an **eigendecomposition** of the observed/expected
  correlation matrix (first eigenvector sign = A vs B) — a dense **cuSOLVER `Dsyevd`**
  job (cf. flagship `2.06`).

- **Loop calling (HiCCUPS, Juicer)** is a **2D convolution**: for every pixel, compare
  its enrichment against "donut"/"lower-left" background kernels — a GPU-accelerated
  custom convolution, the catalog's named GPU win.

- **ChromaFold (2024)** skips the contact matrix entirely and trains a **CNN (cuDNN)**
  to predict 3D contacts from 1D accessibility (ATAC-seq) signals.

This project deliberately ships the **ICE + insulation** core (a reduced-scope
teaching version, CLAUDE.md §13); the compartment/loop/CNN steps are described here so
the reader knows where this slots into a full pipeline.

---

## References

- Lieberman-Aiden et al. (2009), *Comprehensive Mapping of Long-Range Interactions
  Reveals Folding Principles of the Human Genome*, Science — the Hi-C assay.
- Imakaev et al. (2012), *Iterative correction of Hi-C data reveals hallmarks of
  chromosome organization*, Nat. Methods — **ICE balancing** (the algorithm here).
- Knight & Ruiz (2013), *A fast algorithm for matrix balancing*, IMA J. Numer. Anal.
  — the **KR** alternative (Exercise 1).
- Crane et al. (2015), Nature — the **insulation score** definition we follow.
- Rao et al. (2014), *A 3D Map of the Human Genome at Kilobase Resolution*, Cell
  (GSE63525) — high-resolution maps and the HiCCUPS loop caller.
- **cooler** https://github.com/open2c/cooler · **cooltools**
  https://github.com/open2c/cooltools — reference sparse storage + ICE/insulation.
- **Juicer / HiCCUPS** https://github.com/aidenlab/juicer — GPU loop caller (KR + 2D conv).
- **Higashi** https://github.com/ma-compbio/Higashi · **ChromaFold**
  https://www.nature.com/articles/s41467-024-53628-0 — single-cell + CNN approaches.
