# THEORY — 1.29 Kinase Selectivity Panel Scoring

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

**Protein kinases** are enzymes that transfer the γ-phosphate of ATP onto a
substrate — the master "on/off" switches of cell signaling. There are ~518 human
kinases (the *kinome*). Many cancers are driven by a single hyperactive kinase
(e.g. the BCR-ABL1 fusion in chronic myeloid leukemia), so kinase inhibitors are a
huge drug class (imatinib, dasatinib, …).

The catch: almost all small-molecule kinase inhibitors are **ATP-competitive** —
they sit in the ATP pocket. That pocket is **structurally conserved across the
kinome**, especially the **hinge region** (the backbone segment that normally
hydrogen-bonds the adenine of ATP). A compound that hydrogen-bonds the hinge of
your target kinase will tend to do the same for *dozens* of others. Those
off-target hits cause side effects and toxicity.

**Selectivity profiling** answers: *across the whole kinome, which kinases does
this compound bind, and how strongly?* Experimentally this is the **KINOMEscan**
assay (Karaman et al., *Nat. Biotechnol.* 2008), which measures binding of one
compound against hundreds of kinases and summarizes promiscuity with the
**S-score**. Computationally, you can *predict* the same profile by docking the
compound into every kinase pocket, featurizing each pose as a **kinase–ligand
interaction fingerprint (IFP)** (the KLIFS framework, Kooistra et al.), and scoring
affinity. This project models that scoring + selectivity step on the GPU.

## 2. The math

**Inputs.**
- A compound (the *query*) described by a feature vector of pharmacophore
  *offers* `L = (L_0, …, L_{F-1})`, `L_f ∈ ℤ≥0` = how much of interaction type *f*
  the ligand can form (count of H-bond donors, acceptors, hydrophobic contacts,
  aromatic π-stacks, ionic contacts, halogen bonds, and a special **hinge** motif).
- A panel of `N` kinases. Kinase *i* has a pocket *requirement* vector
  `R_i = (R_{i,0}, …, R_{i,F-1})`, `R_{i,f} ∈ ℤ≥0` = how much of type *f* the pocket
  wants, plus a scalar `bias_i` (a pocket-independent affinity offset).
- Fixed per-channel weights `w = (w_0, …, w_{F-1})`, `w_f ∈ ℤ>0` (importance of each
  interaction type). The **hinge channel has the largest weight** because hinge
  H-bonding dominates ATP-competitive binding.

Here `F = NFEAT = 8`.

**Per-kinase raw match score.** You can only realize the *overlap* of what the
ligand offers and what the pocket needs on each channel, so:

```
raw_i = bias_i + Σ_{f=0}^{F-1}  min(L_f, R_{i,f}) · w_f          (an integer)
```

`min(offer, need)` is the standard "you cannot make more H-bonds than either
partner allows" rule, identical in spirit to the saturating overlap used in
fingerprint scoring.

**Predicted affinity.** Map the raw score to a `pK`-like number (think `pKd` or
`pIC50`; higher = tighter). We use a simple affine map evaluated in **fixed-point
milli-units** (`pK × 1000`) so it stays an exact integer:

```
pK_milli_i = PK_BASE + raw_i · PK_PER_POINT      (= 4000 + 50·raw_i)
```

so `raw_i = 0 → pK 4.000` (a non-binder floor) and each raw point adds `0.050` pK.

**Hit and S-score.** A kinase counts as *bound* if its affinity clears a threshold
`x` (we use `pK ≥ 6.000`, i.e. `Kd ≲ 1 µM`, the common cutoff):

```
hit_i = [ pK_milli_i ≥ 6000 ]                    (1 if bound, else 0)
S(x)  = ( Σ_i hit_i ) / N                          (the selectivity S-score)
```

A **small S-score means a selective compound** (it binds few kinases). The numerator
`Σ_i hit_i` is the *S-count*; we report `S` as the exact integer ratio.

**Top-K.** Sort kinases by `pK_milli_i` descending (ties broken by lower index for
determinism) and report the K most potently bound.

## 3. The algorithm

```
load:    L, {(R_i, bias_i, name_i)}_{i<N}, w
score:   for each kinase i in 0..N-1:           # N independent iterations
             raw_i  = bias_i + Σ_f min(L_f, R_{i,f})·w_f
             pK_i   = 4000 + 50·raw_i
             hit_i  = (pK_i >= 6000) ? 1 : 0
reduce:  S_count = Σ_i hit_i                     # integer sum
rank:    top-K kinases by pK_i                   # partial sort
```

**Complexity.**
- Per kinase: `O(F)` work (a length-8 loop). Total scoring: `O(N·F)`.
- S-count reduction: `O(N)`.
- Top-K: `O(N log K)` with `partial_sort`.

**Serial vs. parallel.** The scoring loop has **zero cross-iteration dependency** —
kinase *i*'s result needs only `L`, `R_i`, `bias_i`. So the *work* is `O(N·F)` and
the *depth* is `O(F)` (constant in N): every kinase can be scored simultaneously.
The arithmetic intensity is low (a handful of integer ops per kinase, reading one
`KinasePocket` from global memory), so at small N the computation is **memory- and
launch-bound**, not compute-bound — see §5 and §7.

## 4. The GPU mapping

**Thread-to-data map.** One **thread per kinase**: thread
`i = blockIdx.x·blockDim.x + threadIdx.x` scores kinase *i*. A **grid-stride loop**
(`i += blockDim.x·gridDim.x`) lets a fixed, modest grid cover a panel of any size.

**Launch configuration.** `THREADS_PER_BLOCK = 256` (a multiple of the 32-lane
warp; 8 warps/block give the scheduler latency hiding and good occupancy on
sm_75–sm_89). Grid = `ceil(N/256)`, capped at 1024 blocks (the stride loop covers
the rest).

**Memory hierarchy — and *why*.**
- **Constant memory** holds the compound `L` (`__constant__ int32_t c_ligand[NFEAT]`).
  Every thread reads all of `L` but none writes it, and it is the same for the whole
  launch. Constant memory has a hardware **broadcast cache**: when a warp reads one
  constant address, it is served to all 32 lanes in a *single* transaction. Putting
  `L` in a global buffer instead would cost `NFEAT` global loads per thread.
- **Global memory** holds the `KinasePocket` array. Consecutive threads read
  consecutive structs (`pockets[i]`), so the access is reasonably coalesced. Each
  `KinasePocket` is a POD struct, so the entire host `std::vector` copies to the
  device in one `cudaMemcpy` (no per-field marshalling).
- **Registers** hold the per-thread accumulator and the loop temporaries; the
  `F = 8` loop is fully unrolled.
- **No shared memory, no atomics** in the kernel: outputs are independent. The
  S-count is reduced on the **host** from the returned integer `hit[]` flags
  (deterministic; see §5).

```
                 constant cache (broadcast)
                 ┌───────────────────────┐
   compound L →  │  c_ligand[0..NFEAT-1] │  (read by every thread, 1 txn/warp)
                 └───────────┬───────────┘
                             │
   global mem:   pockets[ ]  │   one KinasePocket per kinase
   ┌──────┬──────┬──────┬────┴─┬──────┐
   │ K0   │ K1   │ K2   │ ...  │ K_{N-1}
   └──┬───┴──┬───┴──┬───┴──────┴──┬───┘
      │      │      │             │
    t0     t1     t2     ...    t_{N-1}        (one thread per kinase)
      │      │      │             │
      ▼      ▼      ▼             ▼
   pK_milli[i], hit[i]   →  D2H  →  host: S_count=Σhit, top-K sort
```

**Libraries (no black boxes).** The catalog mentions **cuML** (for ML activity
prediction) and **Thrust** (top-K). This teaching version keeps the ranking on the
host with `std::partial_sort` (`O(N log K)`, trivial for a panel) so the data flow
is fully visible. To do it on-device you would use `thrust::sort_by_key` (or a CUB
`DeviceRadixSort` over the `pK` keys carrying kinase indices as values), then read
the top K. The S-count is a sum reduction — a one-liner `thrust::reduce` — but we do
it on the host because the array is already back and integer addition is exact
either way.

## 5. Numerical considerations

**Integer / fixed-point on purpose.** Every quantity in the scoring path — offers,
requirements, weights, the raw score, the `pK·1000` affinity, the hit flags, and the
S-count — is an **integer**. This is a deliberate determinism choice
(PATTERNS.md §3):

- **No floating-point reordering.** Floating-point addition is *not* associative, so
  a parallel float reduction can give an order-dependent (nondeterministic) sum.
  Integer addition **commutes and associates exactly**, so summing `hit[]` on the
  host gives the *same* `S_count` as any device-side order, and it matches the CPU
  bit-for-bit.
- **Exact CPU↔GPU agreement.** The per-kinase physics lives in **one**
  `__host__ __device__` function (`score_kinase` in `selectivity_core.h`). The host
  reference and the device kernel call the *same* source, so they execute identical
  integer arithmetic — the verification tolerance is **0**, not an epsilon.

**Precision/overflow.** `int32_t` is far more than enough: with `F = 8`, weights ≤ 6,
offers/needs ≤ ~15, and a small bias, `raw` is in the low hundreds and `pK·1000` is a
few thousand — orders of magnitude below the 2³¹ limit. No FP32/FP64 question arises
because there is no floating point in the result.

**Race conditions.** None: each thread writes only its own `pK_milli[i]` and
`hit[i]`. No two threads touch the same address, so no atomics or locks are needed.

## 6. How we verify correctness

**Independent serial reference.** `src/reference_cpu.cpp::score_panel_cpu()` loops
over kinases in plain C++ and produces `pK_milli[]`, `hit[]`, and the `S_count`.
`main.cu` runs it, runs the GPU path, and asserts **exact equality** of all three:

```cpp
bool pass = (pK_cpu == pK_gpu) && (hit_cpu == hit_gpu) && (s_count_cpu == s_count_gpu);
```

**Why a tolerance of 0 is justified.** Both sides call the identical
`__host__ __device__` integer functions; integer math is exact and order-independent;
there is no floating point in the computed result. So any mismatch would indicate a
real bug (a memory error, a bad index, a missed kinase), not benign FP drift. The
program prints `exact-match mismatches = 0 / N` to make this explicit.

**Embedded known answer (a second, stronger check).** The synthetic sample is
engineered (`scripts/make_synthetic.py`) so **ABL1 is the unique rank-1 hit** and the
**S-score is 1/16 = 0.062**. The demo recovering exactly that ranking validates the
*science wiring* (top-K, threshold, S-score), not just CPU==GPU agreement
(PATTERNS.md §6). Edge cases handled by the loader: missing file, `NFEAT` mismatch,
short rows — all throw with a clear message so the demo fails loudly.

## 7. Where this sits in the real world

This is a **reduced-scope teaching version** of a real selectivity pipeline. What
production does that we omit:

- **Docking generates the IFP.** Tools like **AutoDock-GPU** dock the compound into
  each kinase structure (or homology model), producing a 3-D pose; **KLIFS/`kissim`**
  then encode that pose as an interaction fingerprint over the **85 KLIFS pocket
  residues** × interaction types — hundreds of bits, not 8 channels. We start from
  pre-baked feature vectors; the GPU mapping (one thread per kinase) is unchanged.
- **Learned scoring.** Our affine `pK = 4 + 0.05·raw` is a stand-in for a fitted
  scoring function. **KinoML** trains ML models (graph/structure-based) on ChEMBL /
  KINOMEscan data to predict activity; **MM-GBSA** rescoring adds physics-based free
  energy. cuML would train such models on-GPU.
- **Richer selectivity metrics.** Beyond the S(x)-score, the field uses the **Gini
  coefficient** and **selectivity entropy** (Uitdehaag & Zaman, 2011) to summarize a
  profile in a concentration-independent way — good exercises (README §Exercises).
- **Scale.** Real sweeps are thousands of compounds × hundreds of kinases × many
  docked poses — exactly where the GPU's flat per-kinase cost crushes the linear CPU
  cost. On our 16-kinase toy the kernel is **launch/copy-bound** and the timing is a
  teaching artifact only (CLAUDE.md §12), never a benchmark claim.

---

## References

- **Karaman, M.W. et al. (2008)** "A quantitative analysis of kinase inhibitor
  selectivity." *Nat. Biotechnol.* 26:127–132. — The KINOMEscan assay and the
  S-score we compute.
- **Uitdehaag, J.C.M. & Zaman, G.J.R. (2011)** "A theoretical entropy score as a
  single value to express inhibitor selectivity." *BMC Bioinformatics* 12:94. — The
  Gini / entropy selectivity metrics (exercise material).
- **Kooistra, A.J. et al. / KLIFS** (https://klifs.net) — the kinase–ligand
  interaction fingerprint framework and the 85-residue pocket definition our toy IFP
  abstracts.
- **AutoDock-GPU** (https://github.com/ccsb-scripps/AutoDock-GPU) — GPU docking; the
  pose-generation step upstream of IFP scoring.
- **KinoML** (https://github.com/openkinome/kinoml) — ML for kinase activity
  prediction; the modern replacement for our hand-set affinity map.
- **HTMD** (https://github.com/Acellera/htmd) — GPU kinome docking workflows; how the
  panel sweep is orchestrated at scale.
- **Flagship `1.12`** (Tanimoto fingerprint search) — the same "score one query vs N
  independent items, query in constant memory" GPU pattern, with `__popcll` bit IFPs.
