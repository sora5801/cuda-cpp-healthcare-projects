# THEORY — 1.13 Pharmacophore & 3D Shape Screening

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

Two molecules that look alike in 3D tend to bind the same protein pocket — even
when they share **no** common substructure and would look unrelated to a 2D
fingerprint (project 1.12). This is the principle behind **shape-based virtual
screening**: take a known active molecule (the *query*), and search a huge library
for molecules whose 3D **shape** (and, optionally, the spatial arrangement of
chemical features — the *pharmacophore*) matches it. The high-scoring hits become
candidates for the more expensive docking step that follows.

The dominant production tool, OpenEye's **ROCS** (Rapid Overlay of Chemical
Structures), measures shape similarity by treating each molecule as a smooth
**volume** and asking how much two volumes overlap when superimposed. The elegant
trick — due to Grant & Pickett (1996) — is to build that volume from **Gaussian**
functions, because the overlap of Gaussians has a closed-form integral. No grid,
no Monte Carlo: just arithmetic over atom pairs. That makes the score cheap,
smooth (differentiable, so the overlay can be *optimized*), and — crucially for us
— **embarrassingly parallel** across library molecules.

A real molecule also carries a *pharmacophore*: hydrogen-bond donors/acceptors,
hydrophobic groups, aromatic rings, ionizable centers. ROCS scores those as a
parallel "color" overlap. This teaching project implements the **shape** half (the
geometry); the color half is Exercise 1 in the README — the math is identical, you
just restrict the sum to atom pairs of the same feature type.

## 2. The math

**Atom as a Gaussian.** Model heavy atom *i*, centered at position **R**ᵢ, as a
single isotropic Gaussian density

$$ \rho_i(\mathbf r) = p \, \exp\!\big(-\alpha_i \, |\mathbf r - \mathbf R_i|^2\big), $$

where *p* = 2.7 is a fixed "partial weight" and αᵢ (units **Å⁻²**) sets the width.
We pick αᵢ so the Gaussian's integrated volume equals the atom's hard-sphere
volume `(4/3)πrᵢ³`. Carrying out that match gives

$$ \alpha_i = \pi \left(\frac{3p}{4\pi}\right)^{2/3} \frac{1}{r_i^{2}}
            \;=\; \frac{\kappa}{r_i^{2}}, \qquad \kappa = \pi\Big(\tfrac{3p}{4\pi}\Big)^{2/3}. $$

So a **bigger** atom (larger radius *rᵢ*) is a **fatter, lower-α** Gaussian. This
is `atom_alpha()` in `src/shape_overlap.h`.

**Overlap of two atoms.** The product of two Gaussians is another Gaussian, and
its integral over all space is closed-form:

$$ V_{ij} \;=\; \int \rho_i(\mathbf r)\,\rho_j(\mathbf r)\,d^3\mathbf r
   \;=\; p^2 \left(\frac{\pi}{\alpha_i+\alpha_j}\right)^{3/2}
         \exp\!\left(-\frac{\alpha_i \alpha_j}{\alpha_i+\alpha_j}\, d_{ij}^2\right), $$

where `d²ᵢⱼ = |Rᵢ − Rⱼ|²` is the squared center-to-center distance. This is
`pair_overlap()`. Note the structure: a constant prefactor times **one** `exp()`
that decays as the atoms separate. Two coincident atoms (d = 0) give the maximum
overlap; far-apart atoms contribute ≈ 0.

**Overlap of two molecules (first order).** Sum over all atom pairs:

$$ O_{AB} \;=\; \sum_{i \in A}\ \sum_{j \in B} V_{ij}. $$

This is `molecule_overlap()`. It is *first order* — the exact union volume would
subtract triple-overlap corrections (inclusion–exclusion), but ROCS and most tools
score with this first-order sum because the higher terms are small and explode
combinatorially (README Exercise 5).

**The score.** The **Shape Tanimoto** normalizes the cross-overlap by the union of
the two self-volumes:

$$ \mathrm{ShapeTanimoto}(A,B) \;=\; \frac{O_{AB}}{O_{AA} + O_{BB} - O_{AB}} \in [0,1]. $$

This is exactly intersection-over-union, but of continuous **volumes** instead of
the discrete **bit sets** of project 1.12. `1.0` = identical shape in space, `0.0`
= disjoint. This is `shape_tanimoto()`.

**Symbols:** *p* = 2.7 (partial weight, dimensionless); αᵢ (Å⁻²); *rᵢ* (Å, van der
Waals radius); **R**ᵢ (Å, atom position); *Vᵢⱼ* (Å³, pair overlap up to *p²*);
*O* (overlap volume); ShapeTanimoto (dimensionless, [0,1]).

## 3. The algorithm

```
load query Q and library {B_0 ... B_{N-1}}          # parse + radius -> alpha
O_AA = molecule_overlap(Q, Q)                        # query self-overlap (once!)
for each library conformer B_k:                      # the parallel dimension
    O_AB = molecule_overlap(Q, B_k)                  # M*K Gaussian evals
    O_BB = molecule_overlap(B_k, B_k)                # K*K Gaussian evals
    score[k] = O_AB / (O_AA + O_BB - O_AB)
report top-K by score
```

**Complexity.** Let N = library size, M = query atoms, K = atoms per conformer.

- **Serial work:** `O(N · (M·K + K²))` Gaussian evaluations (one `exp()` each).
  `O_AA` is hoisted out of the loop, so it costs `O(M²)` once, not per conformer.
- **Parallel depth:** the N conformers are independent, so with N processors the
  *depth* is just one conformer's `O(M·K + K²)` — the screen is **embarrassingly
  parallel** in N.
- **Arithmetic intensity:** each thread reads one small conformer (K atoms × 32 B)
  and the query from constant cache, then does `O(M·K)` flops including a
  transcendental `exp()` per pair. That is a high compute-to-memory ratio — this
  kernel is **compute-bound**, not bandwidth-bound, which is the comfortable regime
  for a GPU.

**Data-access pattern:** the query is read by every thread (→ constant memory); each
conformer is read by exactly one thread (→ plain global load, naturally coalesced
within a warp because consecutive threads read consecutive `Molecule` structs).

## 4. The GPU mapping

**Thread-to-data map.** One thread owns one library conformer:
`k = blockIdx.x * blockDim.x + threadIdx.x`, with a **grid-stride loop**
(`k += blockDim.x * gridDim.x`) so a fixed, modest grid covers an arbitrarily large
library. Thread *k* computes `score[k]` and nothing else — no cooperation between
threads, no atomics, no shared-memory reduction. This is the cleanest parallel
pattern there is, shared with `1.12` (Tanimoto) and `12.01` (spectral search).

**Launch configuration.** `THREADS_PER_BLOCK = 128`. This kernel is register-hungry
(each thread keeps a conformer's worth of coordinates in flight and runs nested
loops of `exp()`), so a block of 128 (vs. the usual 256) keeps register pressure
down while still giving the scheduler four warps per block to hide instruction
latency. Blocks = `ceil(N / 128)`, capped at 1024 (the grid-stride loop mops up any
remainder).

**Memory hierarchy — and why:**

| Data | Lives in | Why |
|---|---|---|
| query molecule | **constant** (`__constant__ Molecule c_query`) | read by *every* thread, never written, fixed for the launch → the constant cache **broadcasts** one address to a whole warp in a single transaction. ~2 KB, far under the 64 KB bank. |
| library conformers | **global** (`d_lib[k]`) | each read by one thread; consecutive threads read consecutive structs → coalesced. |
| `O_AA` | **register** (kernel argument) | identical for all threads; computed once on the host and passed by value — never recomputed. |
| per-thread sums | **registers** | the inner-loop accumulators; no shared memory needed. |

```
            __constant__ c_query  (broadcast to all warps)
                       |
   grid ──> block 0 ─ block 1 ─ ... ─ block G-1     (grid-stride over N)
              |          |               |
           [t0..t127] [t0..t127] ...   each thread t  ->  conformer k
              |
              +-- molecule_overlap(c_query, d_lib[k])  = O_AB   (M*K exp's)
              +-- molecule_overlap(d_lib[k], d_lib[k])  = O_BB   (K*K exp's)
              +-- score[k] = O_AB / (O_AA + O_BB - O_AB)
```

**Occupancy story.** With 128 threads/block and a per-thread register footprint
driven by the `Molecule` POD, a Turing/Ampere/Ada SM holds several blocks
concurrently — enough warps to keep the `exp()` pipeline busy. Because the kernel is
compute-bound, we do not chase maximum occupancy; we just need enough warps to hide
the latency of the transcendental units.

**No CUDA library needed here.** The catalog suggests cuBLAS for the *rotation-matrix
application* in rigid-body alignment. Our teaching version scores a **pre-aligned**
overlay, so there is no GEMM to do — adding the alignment search (README Exercise 2)
is exactly where cuBLAS (batched `R·x` over many conformers) would earn its place. We
keep it out so the core idea — the closed-form Gaussian overlap — stays in the open,
no black box (CLAUDE.md §6.1.6).

## 5. Numerical considerations

**Precision: FP64 throughout.** `shape_overlap.h` accumulates in `double`. We could
use `float` (the scores are in [0,1]), but double has two payoffs: (a) the
intermediate overlap volumes span a wide dynamic range before normalization, and (b)
it makes CPU/GPU agreement essentially exact, which is the whole point of the shared
core. On the RTX 2080 Super (consumer Turing) FP64 is throttled, but this kernel is
not FP64-bound at teaching scale, so the clarity wins.

**Determinism.** The overlap sum is a fixed loop order — `i` outer, `j` inner —
**identical** on CPU and GPU because both call the same `molecule_overlap()`. There
are **no atomics** and **no cross-thread reductions**, so there is no
order-of-summation nondeterminism (contrast `5.01`/`11.09`, which must accumulate in
integers/fixed-point to stay deterministic). Floating-point addition is not
associative, but here the order is pinned, so the result is bit-stable run to run
(verified: stdout is byte-identical across repeated runs and between Debug and
Release).

**The one residual difference: FMA contraction.** nvcc may fuse `a*b + c` into a
single fused-multiply-add (one rounding instead of two); the host compiler may not.
This is a ~10⁻¹⁶ relative effect per operation and is the *only* reason CPU and GPU
are not bit-for-bit identical. Our tolerance covers it with enormous margin (§6).

**Stability.** The `exp()` argument is always ≤ 0 (it is `−(positive)·d²`), so `exp`
never overflows; the worst case is underflow to 0 for very distant atoms, which is
the physically correct "no overlap". The Tanimoto denominator is guarded against
zero (empty molecule → score 0, not NaN).

## 6. How we verify correctness

`src/reference_cpu.cpp` provides an independent serial implementation,
`shape_tanimoto_cpu()`. `main.cu` runs **both** the CPU reference and the GPU kernel
on the same input and reports `max_abs_err = max_k |score_cpu[k] − score_gpu[k]|`.

**Tolerance: `1e-9` (absolute, on scores in [0,1]).** Because both paths execute the
*same* double-precision `molecule_overlap()` / `shape_tanimoto()` from the shared
header, in the *same* loop order, the only divergence is FMA contraction (~1e-16 per
op accumulated over a short sum). The measured error is `≈ 3.3e-16` — seven orders of
magnitude under the floor. We set the floor at `1e-9` (not `0`) to be honest that FMA
*could* differ, while still being a stringent agreement test (PATTERNS.md §4: "the
same exact operations on both sides" → near-exact tolerance).

**A second, stronger check — recovering a known answer.** Verifying CPU == GPU only
proves the two implementations agree; it does not prove the *science* is right. So the
synthetic sample (`make_synthetic.py`) embeds a ground truth: `lib_00_self` is an
exact copy of the query, which **must** score exactly `1.000000`, and the graded
perturbations must rank in the physically sensible order (exact > jitter > small
shift > rotation > grown > … > far/different ≈ 0). The demo's top-5 shows precisely
that, so a reader can confirm the model behaves correctly by eye.

**Edge cases handled:** empty/degenerate molecule (guarded denominator), atom count
over `MAX_ATOMS` (loader throws), missing file (clean error to stderr, exit 2),
ragged last block (grid-stride guard `k < n`).

## 7. Where this sits in the real world

Production shape screening (ROCS and friends) does several things this teaching
version deliberately omits:

- **It optimizes the overlay.** The single biggest difference: ROCS does not assume
  the molecules are pre-aligned. It *maximizes* `O_AB` over rigid-body transforms
  (translation + rotation, often via quaternions) — usually from several starting
  orientations — because the overlap is a smooth, differentiable function of the
  pose. We score one fixed superposition. (README Exercise 2; this is where the
  catalog's cuBLAS rotation-matrix application lives.)
- **Color/pharmacophore on top of shape.** ROCS adds a "color" overlap over chemical
  feature types (HBD/HBA/hydrophobic/aromatic/anion/cation) and reports a Combo
  score. Same Gaussian math, restricted to like-typed atom pairs (Exercise 1).
- **Conformer ensembles.** A flexible molecule is screened as *many* conformers
  (generated by tools like OMEGA/RDKit); the molecule's score is the best over its
  conformers. We treat each library entry as a single rigid shape.
- **Scale & engineering.** Real screens run over **billions** of conformers with
  multi-Gaussian atom models, element-specific radii, second-order volume
  corrections, and heavy GPU/cluster parallelism. The independent-jobs structure we
  show is exactly how that scales — just with N in the billions instead of 9.

What this project *does* faithfully capture is the mathematical heart: the
closed-form Gaussian volume overlap and the Shape Tanimoto, computed identically on
CPU and GPU, mapped onto the GPU with the constant-memory broadcast pattern that the
production tools also rely on.

---

## References

- **Grant, J. A.; Gallardo, M. A.; Pickett, S. D. (1996).** "A fast method of
  molecular shape comparison: A simple application of a Gaussian description of
  molecular shape." *J. Comput. Chem.* 17(14):1653–1666. — The Gaussian-volume model
  and the overlap integral this project implements.
- **Rush, Grant, Mosyak & Nicholls (2005).** ROCS-based virtual screening case study
  (ZipA–FtsZ) — why shape similarity finds novel scaffolds 2D methods miss.
- **OpenEye ROCS docs** — <https://www.eyesopen.com/rocs> — ShapeTanimoto,
  ColorTanimoto, Combo score, and the overlay optimization we leave as an exercise.
- **RDKit shape tools** — <https://github.com/rdkit/rdkit> — the closest open-source
  reference implementation of Gaussian shape alignment/scoring; study `rdShapeAlign` /
  `rdMolAlign`.
- **Pharmer** — <https://github.com/dkoes/pharmer> — open pharmacophore (feature-
  point) search, the complementary "color" view of the same screening problem.
