# THEORY — 2.33 Structure-Based Pharmacophore Modeling from MD Ensembles

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

When a small-molecule **ligand** (a drug candidate) binds a protein **receptor**,
it does so by presenting the right chemical groups in the right places: a
hydrogen-bond donor where the protein offers an acceptor, a greasy hydrophobic
group against a greasy pocket wall, a positive charge near a negative one, and so
on. The 3-D pattern of those required interaction points — *not* the exact atoms,
but the abstract "donor here, hydrophobe there" geometry — is called a
**pharmacophore**. If you know the pharmacophore, you can screen millions of
molecules and keep only those that can place matching features in matching spots.

A pharmacophore is usually read off a single crystal structure of the
protein–ligand complex. But proteins **move**. The pocket breathes; side chains
rotate; transient "cryptic" pockets open and close. A single static snapshot can
miss interaction points that exist most of the time, or over-weight ones that are
fleeting. **Molecular dynamics (MD)** simulates the protein in motion, producing a
*trajectory* of thousands to millions of conformational **frames**. An **ensemble
pharmacophore** is the *consensus* of the features seen across those frames: a
donor that is present in 95% of frames is a strong, high-weight feature; one seen
in 10% is weak or discarded. This consensus captures **induced-fit** binding
(the pocket adapting to the ligand) and cryptic pockets that no single frame shows.

Once you have that ensemble pharmacophore, the money step is the **screen**: take a
library of 10⁶–10⁹ candidate molecules and rank each by how well its own features
can be overlaid on the query pharmacophore. The molecules that score highest are
the ones worth synthesizing and testing. **That ranking is what this project
computes on the GPU.**

> **Scope of this teaching version.** We take the query pharmacophore and each
> library molecule's typed feature points as *given* (synthetic input). We do
> **not** run MD, extract features from frames, or cluster them (DBSCAN) — those
> are described in §7. We implement the **3-D Gaussian-overlap scoring screen**,
> which is the core GPU pattern and the step that dominates at scale.

## 2. The math

A **feature** is a typed Gaussian sphere in 3-D space. Feature `i` has:

- a **type** `τᵢ ∈ {donor, acceptor, hydrophobe, aromatic, +charge, −charge}`,
- a **center** `rᵢ = (xᵢ, yᵢ, zᵢ)` in ångströms (Å),
- a **weight** `wᵢ ∈ [0,1]` (consensus confidence from the MD ensemble).

We model each feature as an isotropic Gaussian `g(r) = exp(−α‖r − rᵢ‖²)`. The
**overlap of two features** `i` and `j` is the product of their weights times the
overlap of their Gaussians, but **only if they are the same type**:

```
                 ⎧ 0                                   if τᵢ ≠ τⱼ   (donor ≠ acceptor)
overlap(i, j) =  ⎨
                 ⎩ wᵢ · wⱼ · exp(−α · ‖rᵢ − rⱼ‖²)      if τᵢ = τⱼ
```

`α` (units Å⁻²) sets how fast overlap decays with separation. We choose it so two
features exactly `r½ = 1 Å` apart still overlap at *half* the maximum:

```
exp(−α · r½²) = ½   ⟹   α = ln 2 / r½²  =  0.6931…   (with r½ = 1 Å)
```

For two **sets** of features Q (query) and L (a library molecule), define the
**total overlap** as the sum over all ordered pairs:

```
O(Q, L) = Σ_{i∈Q} Σ_{j∈L} overlap(i, j)
```

The raw cross-overlap `O(Q,L)` rewards big molecules (more features → more chances
to overlap). To remove that bias we use a **Tanimoto** (Jaccard-like) normalization
— exactly what ROCS calls the *color Tanimoto*:

```
            O(Q, L)
T(Q, L) = ───────────────────────────────       O_QQ = O(Q,Q),  O_LL = O(L,L)
          O_QQ + O_LL − O(Q, L)
```

`T ∈ [0, 1]`: it is **1** when L's features coincide exactly with Q's (then
`O_QL = O_QQ = O_LL` and the fraction is 1), and near **0** when they are far apart
or of the wrong types. The self-overlap `O_QQ` is the **same for every library
molecule**, so we compute it once.

**Inputs:** a query feature set Q and N library feature sets `L₁…L_N`.
**Output:** the N scores `T(Q, L_k)`, and their ranking.

## 3. The algorithm

```
precompute O_QQ = O(Q, Q)                         # once, depends only on the query
for each library molecule k = 1..N:               # ← the parallel loop
    O_QL = Σ_{i∈Q} Σ_{j∈L_k} overlap(i, j)        # cross term
    O_LL = Σ_{i∈L_k} Σ_{j∈L_k} overlap(i, j)      # self term (for Tanimoto)
    score[k] = O_QL / (O_QQ + O_LL − O_QL)
rank molecules by score, report top-K
```

**Complexity.** Let `q = |Q|` (a handful, ~5–10) and `m_k = |L_k|` (also small,
~4–10). One molecule costs `O(q·m_k + m_k²)` overlap evaluations, each a few
multiplies and one `exp`. Over the whole library the serial cost is
`Σ_k O(q·m_k + m_k²)` — **linear in the library size N** (since the per-molecule
work is bounded). There is no cross-molecule dependency, so the **parallel depth is
O(q·m + m²)** (one molecule's work) and the **parallel width is N**: this is the
textbook *independent-jobs* shape (PATTERNS.md §1).

**Data-access pattern.** Each thread reads the (tiny, shared) query and its own
molecule's contiguous block of features. Arithmetic intensity is modest — a short
double loop of `exp` per molecule — so on small inputs the kernel is **launch- and
occupancy-bound**, and the GPU's advantage only shows once N is large (PATTERNS.md
§7). The query reuse across all threads is what makes constant memory pay off.

## 4. The GPU mapping

**Thread-to-data mapping.** One thread scores one library molecule:

```
k = blockIdx.x * blockDim.x + threadIdx.x      // molecule index this thread owns
if (k >= N) return;                            // guard the ragged last block
```

**Launch configuration.** Block of `THREADS_PER_BLOCK = 256` (a multiple of the
32-lane warp; enough warps to hide latency; many resident blocks for occupancy),
grid of `ceil(N / 256)` blocks. Tunable per GPU.

**Variable-length library → flat CSR layout.** Molecules have *different* feature
counts, so we cannot use a fixed 2-D array. Instead all molecules' `Feature`
records are concatenated into one buffer `lib_feats`, with an `offset[]` array
(length N+1) marking where each molecule starts — the **Compressed-Sparse-Row**
idiom. Thread `k` reads molecule `k`'s features from
`lib_feats[offset[k] .. offset[k+1])`. One coalesced device allocation instead of N
tiny ones.

**Memory hierarchy and why:**

| Data | Lives in | Why |
|---|---|---|
| query features `Q` | **constant** memory (`__constant__ c_query`) | read by *every* thread, never written during the launch → the constant cache broadcasts one feature to a whole warp in a single cycle, with zero global-load traffic |
| library `lib_feats`, `offset` | **global** memory | large, read once per thread; coalesced across the flat buffer |
| running sums `O_QL`, `O_LL` | **registers** | per-thread scalars; no sharing needed |
| `score[k]` | **global** memory | one write per thread |

No shared memory, **no atomics** — the molecules are completely independent, so
each thread writes its own output slot. (Contrast project 11.09's k-means, which
*must* use atomics to accumulate shared centroids.)

```
   grid of blocks over the N library molecules
   ┌──────── block 0 ────────┐ ┌──────── block 1 ────────┐
   │ t0  t1  t2 ...  t255    │ │ t0  t1 ...               │   ← 256 threads/block
   └──┬───┬───┬──────────────┘ └─────────────────────────┘
      │   │   └ molecule 2  ─┐
      │   └─── molecule 1    │  each thread:  read c_query (constant, broadcast)
      └─────── molecule 0    │                read its features  lib_feats[offset[k]..]
                             ▼                score_molecule()  →  score[k]  (global)
   __constant__ c_query[ q ]  ── broadcast to all threads ──▶ (no global traffic)
```

**No CUDA library is used here.** The overlap scoring is hand-written so the lesson
is visible. The catalog mentions **cuML DBSCAN** for the *clustering* step that
builds the consensus pharmacophore — that is upstream of this screen and out of
scope (§7); were we to hand-roll DBSCAN we would need a parallel neighbor-graph and
union-find, a project of its own.

## 5. Numerical considerations

- **Precision.** Feature coordinates are **FP32** — MD positions carry ~0.01 Å
  noise, so double precision would be false precision, and FP32 halves the memory
  bandwidth the kernel is bound by. The **accumulators** (`O_QL`, `O_LL`, `O_QQ`)
  are **double**, giving the running sums headroom; the final Tanimoto is collapsed
  back to FP32. This `float data / double accumulate` split matches the CPU
  reference exactly.
- **The shared `__host__ __device__` core.** `overlap_pair()` and
  `score_molecule()` live in `src/pharmacophore.h`, decorated `__host__ __device__`,
  and are called *verbatim* by both the CPU reference and the GPU kernel
  (PATTERNS.md §2). Same operations, same order → the two sides agree to the last
  bit on this short computation. Here `max_abs_err = 0` in practice.
- **Determinism.** Each thread sums its *own* molecule's pairs in a fixed loop
  order — there is **no cross-thread reduction**, so there is no atomic reordering
  and no float-summation nondeterminism. Run-to-run, `stdout` is byte-identical
  (PATTERNS.md §3); only timings (on `stderr`) vary.
- **`exp` agreement.** The one place host and device *could* diverge is the
  transcendental `exp`. CUDA's `exp` is IEEE-754-faithful to <1 ulp and matches the
  host `std::exp` closely; with the double accumulators the difference stays far
  below the `1e-5` tolerance (and is exactly 0 for this sample).
- **Divide-by-zero guard.** If a feature set is empty/degenerate the Tanimoto
  denominator can vanish; `score_molecule()` returns 0 in that case.

## 6. How we verify correctness

`src/reference_cpu.cpp` is an independent, obviously-correct serial implementation:
a plain loop over molecules calling the same `score_molecule()`. `main.cu` runs
**both** the CPU reference and the GPU kernel and compares the per-molecule score
arrays with `util::max_abs_err`, asserting the largest discrepancy is `≤ 1e-5`.

Why `1e-5`? Both paths execute the *same* operations (PATTERNS.md §4: "same ops,
single precision"), so they should differ only by the GPU's fused-multiply-add vs.
the host's separate mul/add inside the `exp` argument — a ~1e-7 wobble on a score
in `[0,1]`. `1e-5` is a safe, honest absolute bound; in practice the error is
exactly `0` for the committed sample.

A second, **scientific** check (PATTERNS.md §4): the synthetic sample plants a
known answer — molecule 7 is a sub-ångström-jittered copy of the query — so we
verify the screen actually *recovers* it: mol[7] must rank **#1**, far above the
random decoys. Agreement of two independent implementations proves the *kernel* is
right; recovering the planted hit proves the *science* (the score discriminates).

**Edge cases exercised:** the ragged last block (`k ≥ N` guard), molecules with
differing feature counts (the CSR layout), a query with a repeated feature type
(the sample query has two `neg-charge` features), and the empty-overlap guard.

## 7. Where this sits in the real world

This teaching version implements the **overlap-scoring screen** only. A production
ensemble-pharmacophore pipeline (OpenEye ROCS/FastROCS, Pharmer, HTMD) adds:

- **Feature extraction from MD frames.** Real tools type features from atoms (SMARTS
  patterns: a hydroxyl is a donor, a carbonyl O an acceptor, a benzene ring an
  aromatic centroid) and locate them per frame. We take typed points as input.
- **Ensemble clustering (DBSCAN).** Across millions of frames the same feature
  appears as a cloud of points; **DBSCAN** (catalog: *cuML DBSCAN* on the GPU)
  clusters those clouds into consensus features and weights each by how persistent
  it is. We skip this; the query is given.
- **Per-molecule pose optimization.** The single biggest simplification: real ROCS
  *rotates and translates* each library conformer to **maximize** the overlap before
  scoring (a quaternion/Newton optimization per molecule, often from several
  starting orientations). We assume features are pre-aligned in a shared frame and
  skip the pose search. This is what makes real screening expensive — and what
  makes the GPU essential.
- **SMARTS matching, common-hits approach (CHA), water-displacement pharmacophores.**
  Refinements that combine multiple pharmacophore hypotheses (CHA) or add features
  where ordered waters are displaced — domain logic layered on top of the same
  overlap scoring.
- **Scale and top-K on device.** A billion-compound screen keeps the running top-K
  on the GPU (block reductions / CUB) and streams the library through; our host-side
  `partial_sort` is fine for the teaching N but would be the bottleneck at scale
  (see Exercises 1 and 5).

The kernel here is faithful to the **heart** of FastROCS — a typed Gaussian "color"
Tanimoto over an independent library — which is exactly the part a GPU transforms.

---

## References

- **Grant, Gallardo & Pickup (1996)**, *A fast method of molecular shape comparison*
  — the Gaussian-overlap formulation ROCS is built on.
- **OpenEye ROCS / FastROCS** — https://www.eyesopen.com/rocs — production GPU
  shape+color overlap; the tool this scoring imitates.
- **Pharmer** (Koes & Camacho) — https://github.com/dkoes/pharmer — open-source
  pharmacophore search; study feature typing and library indexing.
- **fpocket / MDpocket** — https://github.com/Discngine/fpocket — pocket detection
  across MD trajectories (the upstream "where are the features" step).
- **HTMD** — https://github.com/Acellera/htmd — ensemble pharmacophore from GPU MD;
  see how consensus features are clustered over frames.
- **Ester et al. (1996)**, *DBSCAN* — the density clustering used to build the
  ensemble consensus (catalog: cuML DBSCAN).
- Internal: `docs/PATTERNS.md` §1 (independent jobs / constant-memory query), §2
  (shared `__host__ __device__` core), §3 (determinism), §4 (tolerance).
