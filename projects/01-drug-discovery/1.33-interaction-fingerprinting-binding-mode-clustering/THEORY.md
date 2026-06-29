# THEORY — 1.33 Interaction Fingerprinting & Binding-Mode Clustering

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

When a small-molecule drug binds a protein, it does not just "stick" — it makes a
specific set of **non-covalent interactions** with particular residues lining the
binding pocket:

- **Hydrophobic contacts** — non-polar surfaces packing against each other.
- **Hydrogen bonds** — a donor (N–H, O–H) and an acceptor (O, N) at ~2.5–3.5 Å.
- **Aromatic / π-stacking** — two aromatic rings stacking face-to-face or edge-to-face.
- **Ionic / salt bridges** — oppositely charged groups (e.g. ligand carboxylate
  against a lysine ammonium).
- (and **halogen bonds**, **π-cation**, etc. in fuller schemes).

The *pattern* of these interactions — "this pose H-bonds Ser87 and π-stacks
Phe123" — is the chemical fingerprint of a **binding mode**. Two questions drive a
lead-optimization campaign:

1. **Docking validation.** A docking program emits many candidate poses per
   ligand. Do they agree on *how* the ligand sits, or do they scatter across
   several modes? Clustering the interaction patterns answers this directly.
2. **SAR by interaction.** Across an MD trajectory (thousands of frames) or across
   many analogs, which interactions are *conserved* (load-bearing) and which are
   transient? Conserved interactions are the ones medicinal chemists protect.

An **interaction fingerprint (IFP)** encodes that pattern as a fixed-length
**bit-string**, exactly so we can compare and cluster modes with the same fast
bit-vector math used for chemical fingerprints (project 1.12). This project builds
IFPs from poses and clusters them into binding modes — both on the GPU.

> Schemes in the literature: **SIFt** (Structural Interaction Fingerprint, Deng et
> al. 2004) lays out a fixed block of interaction-type bits per binding-site
> residue — the layout we use. **PLIF** (MOE) and **PLEC** (Wójcikowski et al.,
> the ODDT/extended-connectivity variant) are richer descendants.

## 2. The math

**Inputs.**
- A binding pocket of `R = NUM_RESIDUES` residues. Residue `j` has an interaction
  center `c_j ∈ ℝ³` (Å) and chemistry flags `χ_j = (hbond_j, aromatic_j, ionic_j) ∈ {0,1}³`.
- `P` ligand poses. Pose `i` has a representative atom `p_i ∈ ℝ³` (Å) and ligand
  chemistry flags `λ_i = (donor_i, aromatic_i, charge_i) ∈ {0,1}³`.

**Stage A — the fingerprint.** For interaction type `t ∈ {hydrophobic, hbond,
aromatic, ionic}` with squared-distance cutoff `D_t` and chemistry gate
`g_t(χ_j, λ_i) ∈ {0,1}`, the bit for (pose i, residue j, type t) is

```
b(i, j, t) = 1   iff   g_t(χ_j, λ_i) = 1   AND   ‖p_i − c_j‖² ≤ D_t
```

(hydrophobic has `g ≡ 1`; the others require both partners to carry the chemistry).
Pose `i`'s fingerprint is the concatenation over residues then types:

```
IFP_i ∈ {0,1}^B,   B = R · T,   bit index = j·T + t      (T = NUM_ITYPES = 4)
```

so here `B = 24·4 = 96` bits, packed into `⌈96/64⌉ = 2` 64-bit words.

**Stage B — Tanimoto distance.** Between two fingerprints `A, B ∈ {0,1}^B`,

```
Tanimoto(A,B) = |A ∧ B| / |A ∨ B|         (Jaccard on bits, in [0,1])
d(A,B)        = 1 − Tanimoto(A,B)          (a distance; d=0 ⇔ identical bits)
```

where `|·|` is popcount. (`|A ∨ B| = 0`, two empty fingerprints, is defined `d=0`.)

**Stage B — clustering objective.** Partition the `P` poses into `K` clusters with
labels `ℓ_i ∈ {0,…,K−1}` and **consensus** centroids `μ_k ∈ {0,1}^B`, minimizing

```
J = Σ_i  d(IFP_i, μ_{ℓ_i})
```

We minimize `J` by Lloyd's algorithm. The twist vs. ordinary k-means: a centroid
of bit-vectors is itself a bit-vector chosen by **per-bit majority vote**,

```
μ_k[b] = 1   iff   2 · (#members of k with bit b set)  ≥  (#members of k)
```

(ties → set). This is the binary analogue of "the mean is the point minimizing
summed squared distance": the majority bit minimizes summed Hamming distance to
the members, and it tracks Tanimoto closely while staying a valid fingerprint.

## 3. The algorithm

```
build_ifps:                                   # STAGE A   — O(P · R)
  for each pose i:
    IFP_i = 0
    for each residue j:
      nibble = bits b(i,j,·)                   # 4 distance tests + chemistry gates
      OR nibble into IFP_i at offset j·T

ifp_cluster (Lloyd, fixed ITERS):              # STAGE B
  init centroids by farthest-first seeding     # O(K · P)
  repeat ITERS times:
    ASSIGN : ℓ_i = argmin_k d(IFP_i, μ_k)      # O(P · K · W) popcounts
    TALLY  : per cluster, count members & per-bit set-counts   # O(P · B)
    UPDATE : μ_k[b] = majority vote            # O(K · B)
  return J
```

**Complexity.** With `W = FP_WORDS` words per fingerprint:

| step | serial work | parallel depth | notes |
|---|---|---|---|
| build IFP | `O(P·R)` | `O(R)` | independent per pose |
| assign | `O(P·K·W)` | `O(K·W)` | independent per pose |
| tally | `O(P·B)` | `O(log P)` (atomics) | scatter-reduce |
| update | `O(K·B)` | `O(1)` | tiny; done on host |

The pose loop (`P`, the big dimension) is fully parallel in build + assign; that is
where the GPU wins as `P` grows into the millions. `ITERS` is fixed (12) so the run
is deterministic — no convergence test whose stopping point could differ between
CPU and GPU.

**Why farthest-first init.** Naive init (first K poses) can place two seeds in the
same mode and miss another entirely. Farthest-first (the greedy core of k-means++)
puts seed 0 anywhere, then each next seed at the pose farthest (max Tanimoto
distance) from all chosen seeds — for well-separated modes this drops exactly one
seed per mode (same lesson as flagship 11.09).

## 4. The GPU mapping

The mapping is **one thread per pose** in every kernel — the pose dimension `P` is
the parallelism, residues/centroids/bits are small inner loops.

```
        P poses (up to millions)                 grid.x = ceil(P / 256) blocks
   ┌────────────────────────────────┐            block  = 256 threads
   │ t0  t1  t2  ...           t_{P-1}│           thread i  ->  pose i
   └────────────────────────────────┘
 build_ifp_kernel : thread i scans R residues  -> writes IFP_i (its own row; no races)
 assign_kernel    : thread i popcounts vs K μ  -> writes label_i (its own slot)
 tally_kernel     : thread i scatters its bits -> atomicAdd into K·B integer counters
```

- **Block size 256** — a multiple of the 32-lane warp; gives the scheduler 8 warps
  to hide global-memory latency, with good occupancy on sm_75…sm_89.
- **Registers / global memory.** `build_ifp_kernel` accumulates its `IFP_i` in
  **registers** (`uint64_t row[FP_WORDS]`) and writes the row once — no repeated
  global traffic. The residue and centroid arrays are small and read-only; they
  live in global memory and ride the L1/L2 cache (every thread reads the same
  residues, so they stay hot). *Exercise 3* moves the K centroids into
  `__constant__` memory, whose broadcast cache is ideal when all threads read the
  same address — the trick flagship 1.12 uses for its query.
- **The TALLY reduction needs atomics.** Many poses share a cluster, so their
  contributions to that cluster's per-bit counters **collide**. We `atomicAdd(…,1u)`
  into `unsigned int` counters. Crucially these are **integers** — see §5.
- **No CUDA library.** The catalog suggests cuML k-means / RAPIDS; we hand-roll so
  nothing is opaque. Writing it "by hand" is exactly these three kernels plus a
  host majority-vote — and that *is* the teaching content. cuML would fuse + tune
  these and add k-means++ on-device, but hide the popcount/atomic lesson.

## 5. Numerical considerations

This pipeline is **entirely integer / bit logic**, which buys us *exact*
reproducibility — the strongest possible correctness story.

- **Geometry → bits is exact.** We compare **squared** distances to squared
  cutoffs (`ifp_sqdist` never calls `sqrt`). The same `float` subtractions and
  multiplies run on host and device via the shared `ifp.h` (the `IFP_HD` idiom).
  A bit is a `≤` comparison of two identically-computed floats → identical on both.
  (The *only* place float order could matter is `a+b+c`; here each `‖p−c‖²` is a
  single fixed expression, not a reduction, so there is nothing to reorder.)
- **Popcount is exact.** `__popcll` on the GPU and the Kernighan loop on the host
  count the set bits of the *same* integer — same answer by definition.
- **The atomic reduction is deterministic — because it is integer.** Float
  `atomicAdd` is *not* associative: summing the same values in a different thread
  order gives a different last bit, so a float tally would be irreproducible
  (PATTERNS.md §3). We add **`1u`** instead. Integer addition commutes and
  associates exactly, so the per-bit counts are **independent of thread order** →
  the GPU tally equals the CPU's serial tally, every run. This is the same
  determinism trick as flagship 11.09 (`km_to_fixed`) and 5.01 (energy quanta).
- **Majority vote is integer.** `2·count ≥ size` is an integer comparison; the
  consensus centroid is therefore identical on CPU and GPU.
- **Precision.** Coordinates are `float` (FP32) — plenty for Å-scale geometry, and
  the distance test's outcome is robust because the synthetic modes sit well inside
  the cutoffs (no pose is balanced on a cutoff boundary). Tanimoto ratios are
  computed in `double` for the argmin comparison, but only from exact integer
  popcounts, so the comparison is stable.

## 6. How we verify correctness

`src/reference_cpu.cpp` is an independent, obviously-correct serial implementation
of the *same* computation (it shares only the per-element math in `ifp.h`, not the
loop structure). `main.cu` runs CPU and GPU and checks **three** things:

1. **IFPs match bit-for-bit** — `fps_cpu[i] == fps_gpu[i]` for every word.
2. **Cluster labels match** — `label_mismatch == 0`.
3. **Consensus centroids match bit-for-bit** — every centroid word equal.

Because the whole pipeline is integer/bit logic (§5), the tolerance is **exact
equality (`== 0`)**, not a floating-point slack — the honest tolerance for this
class of computation (PATTERNS.md §4). Agreement between two independent
implementations that nonetheless produce *bit-identical* output is strong evidence
the GPU code is right.

A second, **scientific** check validates the *result*, not just CPU==GPU: the
synthetic sample plants 4 well-separated modes (30 poses each), and the demo
reports **`mode recovery = 100.00%`** — cluster purity against the ground-truth
`true_mode` labels (a deterministic integer majority metric). So we know clustering
rediscovered the planted biology, and each cluster's printed consensus contacts
fall in the expected residue neighborhood. Edge cases handled: empty clusters keep
their old centroid (no divide-by-zero); empty∧empty fingerprints define `d=0`.

## 7. Where this sits in the real world

Production interaction-fingerprinting tools differ in three ways this teaching
version deliberately simplifies:

- **Richer interaction geometry.** Real detectors (**ProLIF**, **PLIP**) use full
  *angle* criteria — donor–H···acceptor angle for H-bonds, ring-normal angles and
  centroid offsets for π-stacking (face-to-face vs. edge-to-face), charge **signs**
  for salt bridges — and many atoms per residue, not one center. Our distance-only,
  one-center predicate is the *teachable skeleton*; swapping in the richer
  `g_t`/cutoffs is local to `ifp_residue_nibble()` and is *Exercise 1*.
- **Fingerprint flavors.** We implement the **SIFt** layout (fixed type-bits per
  residue). **PLEC** (ODDT) instead hashes *environments* of interacting atom pairs
  into a long folded bit-vector (ECFP-style), capturing more context at the cost of
  per-bit interpretability. **PLIF** (MOE) and **kissim** (KLIFS, for kinases) are
  other production schemes.
- **Scale + tuned clustering.** Real runs cluster `10⁴–10⁶` poses/frames; **cuML**
  provides GPU k-means with on-device k-means++ init and **RAPIDS cuDF** streams MD
  frames. They would fuse our three kernels and tune occupancy, but the popcount +
  integer-atomic structure is exactly what they do under the hood — which is the
  point of hand-rolling it here.

The clustering itself (Tanimoto k-means with consensus centroids) is a faithful,
full-scale-ready algorithm; only the per-bit *interaction predicate* and the data
scale are reduced for teaching.

---

## References

- **Deng, Chuaqui & Singh (2004)**, *Structural Interaction Fingerprint (SIFt)*,
  J. Med. Chem. — the per-residue interaction-bit layout this project uses.
- **Wójcikowski, Kukiełka, Stepniewska-Dziubinska & Siedlecki (2019)**, *PLEC
  fingerprints*, Bioinformatics — extended-connectivity interaction fingerprints.
- **ProLIF** — <https://github.com/chemosim-lab/ProLIF> — read its interaction
  definitions to see the real distance+angle criteria; the gold standard for IFPs
  from MD trajectories.
- **ODDT** — <https://github.com/oddt/oddt> — IFP/PLEC inside a docking pipeline.
- **kissim / KLIFS** — <https://github.com/volkamerlab/kissim> — fixed-length
  kinase fingerprints clustered across the kinome.
- **Lloyd (1982)**, *Least squares quantization in PCM* — the k-means iteration;
  **Arthur & Vassilvitskii (2007)**, *k-means++* — the seeding our farthest-first
  init approximates.
- Project **1.12** (Tanimoto similarity) and **11.09** (GPU k-means, integer
  atomics) in this repo — the two patterns this project combines.
