# THEORY — 1.4 Ultra-Large Virtual Screening

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See
> [README.md](README.md) for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

Drug discovery starts with a **target** — usually a protein whose activity we
want to block or boost (e.g. the SARS-CoV-2 main protease). A drug is a small
molecule (a **ligand**) that fits into a pocket on that protein and sticks there.
The first computational step in finding one is **virtual screening**: take a huge
library of candidate molecules and rank them by how likely each is to bind, so
chemists only synthesize and assay the most promising few.

Two facts make this a *systems* problem, not just a chemistry one:

- **The libraries are astronomically large.** "Make-on-demand" catalogs list
  molecules that don't physically exist yet but can be synthesized on request:
  **Enamine REAL** has **>6 billion** compounds, **ZINC** ~2 billion. You cannot
  run an expensive simulation on each.
- **The expensive step is docking.** Full docking searches for the best 3-D pose
  of a flexible ligand inside the pocket (a global optimization over translation,
  rotation, and every rotatable bond) and scores the binding energy. One docking
  is milliseconds-to-seconds of compute; a billion of them is supercomputer
  scale (the **Summit** COVID-19 campaign docked >1 billion compounds with
  AutoDock-GPU).

The practical answer is a **funnel**: cheap, parallel filters and **surrogate**
scores throw out the obviously-bad billions, so the expensive docking budget is
spent only on the plausible millions. This project implements that funnel's
cheap core — a **drug-likeness filter cascade** plus a fast **surrogate score** —
and maps it onto the GPU exactly the way a real campaign maps its per-ligand
work. We explicitly *do not* implement physics-based docking (§7 is honest about
the gap).

## 2. The math

We are given one target `T` and a library of `N` ligands `L_0 … L_{N-1}`. Each
ligand is a small **descriptor vector** plus a pharmacophore **feature bitmask**
(a deliberately reduced model — see §7):

```
L = (mw, logp, hbd, hba, rotb, psa, feat)
        mw    molecular weight (Da)            logp  logP·100 (signed int)
        hbd   H-bond donors                    hba   H-bond acceptors
        rotb  rotatable bonds                  psa   polar surface area (Å²)
        feat  32-bit pharmacophore bitmask
T = (mw_opt, logp_opt, psa_opt, feat_required)
```

**Stage 1 — filter cascade.** A ligand *passes* iff it satisfies all of
Lipinski's Rule of Five and the Veber rules:

```
pass(L) = (mw ≤ 500) ∧ (logp ≤ 500) ∧ (hbd ≤ 5) ∧ (hba ≤ 10)
                     ∧ (rotb ≤ 10) ∧ (psa ≤ 140)
```

(`logp` is stored ×100, so "logP ≤ 5" is the integer test `logp ≤ 500`.) These
are the classic empirical rules for oral bioavailability: too big, too greasy,
too many H-bonds, too floppy, or too polar → poor absorption.

**Stage 2 — surrogate dock score.** For a passing ligand we compute a single
integer:

```
score(L,T) = BASE
           + W · popcount(feat & feat_required)        ← pharmacophore overlap
           − |mw   − mw_opt|   / MW_SCALE               ← size mismatch penalty
           − |logp − logp_opt| / LOGP_SCALE             ← lipophilicity penalty
           − |psa  − psa_opt|  / PSA_SCALE              ← polarity mismatch penalty
```

clamped at 0, with `BASE=1000, W=60, MW_SCALE=10, LOGP_SCALE=50, PSA_SCALE=4`
(constants live once in `screen_core.h`). The first term rewards a ligand that
presents the pharmacophore features the pocket wants — exactly the
**`AND` + `popcount`** motif of Tanimoto similarity (project 1.12). The penalty
terms reward a ligand whose bulk properties complement the pocket. A *failed*
ligand gets the sentinel `REJECTED = −1`.

**Output:** the number of survivors and the indices of the **top-K** survivors by
score (ties broken by lower index → deterministic).

## 3. The algorithm

```
for each ligand i in 0 … N-1:          # fully independent across i
    if not pass(L_i): score[i] = REJECTED; continue
    score[i] = surrogate_dock_score(L_i, T)
survivors = count(score[i] != REJECTED)
hits      = indices of the K largest score[i]      # partial sort
```

- **Work / complexity.** Stage 1 + 2 are `O(1)` per ligand (a fixed number of
  comparisons, one 32-bit popcount, three integer divides), so scoring the whole
  library is `Θ(N)` work with **depth `O(1)`** in the parallel model — the ideal
  shape for a GPU. The top-K is `O(N log K)` with `std::partial_sort` on the
  host (negligible here; on-device alternatives in §7).
- **Arithmetic intensity.** Low: each ligand reads ~28 bytes (`sizeof(Ligand)`)
  and does ~15 integer ops, so the kernel is **memory-bandwidth-bound** at scale,
  not compute-bound. That is fine — at billion-ligand scale the win is reading
  the library once at full HBM bandwidth and scoring it in flight.
- **Data-access pattern.** Strictly streaming: thread `i` reads `ligands[i]`
  once, writes `score[i]` once. Consecutive threads read consecutive structs →
  **coalesced** global loads. No ligand depends on another → no synchronization.

## 4. The GPU mapping

**Thread-to-data map.** One logical thread per ligand. Thread
`i = blockIdx.x·blockDim.x + threadIdx.x` scores ligand `i`. A **grid-stride
loop** (`i += blockDim.x·gridDim.x`) lets a fixed, modest grid cover a library of
any size — the standard idiom for "more data than threads".

**Launch configuration.** `block = 256` threads (a multiple of the 32-lane warp;
8 warps give the scheduler enough in-flight work to hide global-memory latency,
and the kernel's tiny register/shared footprint means we hit full occupancy on
sm_75–sm_89). `grid = ceil(N/256)`, **capped at 1024 blocks**, with the
grid-stride loop handling the remainder.

**Memory hierarchy.**

- **Constant memory** holds the `Target` (`__constant__ Target c_target`). Every
  thread reads the same target and none writes it during the launch, so the
  constant cache **broadcasts** one address to a whole warp in a single
  transaction — far cheaper than each thread issuing a global load. (Same idea as
  the constant-memory query in project 1.12.)
- **Global memory** holds the ligand array (read, coalesced) and the score array
  (written, coalesced).
- **Registers** hold each thread's working ligand and accumulator. **No shared
  memory and no atomics** are needed — outputs are independent.

```
            constant cache (broadcast)
                  c_target  ──────────────┐
                                          ▼
   global:  ligands[0] ligands[1] ... ligands[N-1]   (coalesced reads)
               │           │                │
            thread0     thread1     ...   threadK      grid-stride loop
               │           │                │          i += blockDim*gridDim
               ▼           ▼                ▼
   global:  score[0]    score[1]   ...   score[N-1]    (coalesced writes)
```

**CUDA libraries.** This teaching kernel is hand-written (no library call) so the
mapping is fully visible. The *production* analogues use **Thrust/CUB** for the
top-K (`cub::DeviceRadixSort` over the scores, or `thrust::sort_by_key`), and the
real docking grids use **texture memory** for the trilinear grid-energy lookups —
both noted as exercises / §7 rather than hidden here.

## 5. Numerical considerations

- **Integer everywhere — by design.** `logp` is stored ×100 so logP comparisons
  are integer; the score is an integer sum of an integer reward and integer-
  divided penalties. There is **no floating-point arithmetic in the kernel**.
- **Determinism.** Integer addition is associative and order-independent, so the
  per-ligand score does not depend on thread scheduling, and there is no
  cross-thread reduction to reorder. The result is **bit-for-bit reproducible**
  run to run (PATTERNS.md §3). The only float in the program is cosmetic —
  `logp/100.0` printed for the human — and it is exactly representable.
- **No races.** Each thread writes a distinct `score[i]`; no atomics, no shared
  accumulators, so there is nothing to race on.
- **Overflow.** With the chosen constants the score is bounded well inside
  `int32` range (`BASE + W·32 ≈ 2920` before penalties), so no overflow is
  possible for any descriptor in the documented ranges.

## 6. How we verify correctness

The CPU reference (`src/reference_cpu.cpp::screen_cpu`) loops over the ligands
calling the **same** `score_ligand()` that the kernel calls — both `#include`
the shared `__host__ __device__` core in `src/screen_core.h` (the HD-macro idiom,
PATTERNS.md §2). `main.cu` runs both paths and counts element-by-element
mismatches.

- **Tolerance = 0 (exact).** Because the two sides execute identical *integer*
  operations, the correct outcome is `mismatches = 0` — the strongest kind of
  check (PATTERNS.md §4). Any nonzero count would signal a real bug (a struct-
  layout mismatch, a divergent code path), not floating-point noise, so we make
  it a hard failure.
- **A stronger, scientific check.** The synthetic sample embeds a **known
  answer**: four "designed binders" that present every rewarded feature and sit
  on the target's ideal size/logP/PSA. The demo confirms these four top the
  ranking — validating that the *scoring* recovers the planted hits, not merely
  that CPU == GPU.
- **Edge cases.** All-zero feature overlap (no match) scores by penalties only;
  the score clamps at 0 rather than going negative; ligands failing the cascade
  are excluded from the hit list (the `REJECTED` sentinel); the loader rejects a
  malformed file loudly.

## 7. Where this sits in the real world

This is a **reduced-scope teaching version** (CLAUDE.md §13). The honest gaps:

- **No real docking.** The expensive heart of a real campaign is **pose search**:
  AutoDock-GPU runs a **Lamarckian Genetic Algorithm + local search (Solis-Wets
  / ADADELTA / BFGS)** per ligand, evaluating a physics-based scoring function on
  a precomputed 3-D **grid** of the pocket (read through **texture memory** for
  fast trilinear interpolation), warp-parallel across the GA population. Our
  stage-2 "surrogate" is a transparent stand-in that exercises the *same GPU
  mapping* (score N independent ligands, keep the top-K) at a fraction of the
  cost. Uni-Dock and Vina-GPU batch thousands of such dockings concurrently.
- **The surrogate idea is real, though.** Cutting full-docking calls with a cheap
  predictor is exactly what **HASTEN** and **REINVENT** do: dock a small subset,
  train an ML model (random forest / GNN) on `(molecule → score)`, predict the
  rest, dock only the predicted-best, iterate (active learning / Bayesian
  optimization). They reach ~90% recall of the true top-1000 after docking ~1%
  of the library. Exercise 3 builds a miniature of this.
- **Filters are real and used first.** Lipinski/Veber/ADMET cascades and
  pharmacophore/shape pre-filters genuinely run before docking to shrink the
  funnel — our stage 1 is faithful (if strict; real cascades allow one Lipinski
  violation — Exercise 2).
- **Scale & memory.** Real campaigns shard libraries that exceed one GPU's
  memory, stream them, use **NVLink multi-GPU** and on-device top-K
  (`cub::DeviceRadixSort`). We load 64 ligands into memory and reduce on the host.

---

## References

- **Lipinski et al. (2001)**, *Adv. Drug Deliv. Rev.* — the "Rule of Five"; the
  basis of our stage-1 cascade.
- **Veber et al. (2002)**, *J. Med. Chem.* — rotatable-bond and PSA rules.
- **AutoDock-GPU** (Santos-Martins et al., 2021) — the CUDA/OpenCL docking engine
  behind the billion-compound campaigns; read it for the real stage-2.
- **Gorgulla et al. (2020), *Nature*** — the >1-billion-compound VirtualFlow
  screen; the scale this project abstracts.
- **HASTEN / REINVENT** — ML-surrogate active-learning screening; the funnel idea.
- **gpusimilarity** (Schrödinger) — GPU bit-overlap pre-screening; the closest
  analogue of our pharmacophore-overlap term and of project 1.12.
