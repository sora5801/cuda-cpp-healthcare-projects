# THEORY — 2.10 Protein Design / Inverse Folding Inference

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use. This ships a **reduced-scope teaching
> model** (CLAUDE.md §13); §7 below is explicit about what the real tool does._

---

## 1. The science

A protein is a chain of **amino-acid residues**. Each residue contributes one
unit to the **backbone** (the repeating N–Cα–C=O main chain) plus a distinctive
**side chain** that gives it its chemistry. The backbone folds into a 3-D shape;
the side chains pack against each other and against water to stabilise that shape.

Two inverse problems sit at the heart of structural biology:

| Problem | Given | Find | Famous tool |
|---|---|---|---|
| **Structure prediction** ("folding") | a sequence | its 3-D structure | AlphaFold2 |
| **Inverse folding** ("design") | a 3-D backbone | a sequence that folds into it | **ProteinMPNN** |

This project is the second one. You hand it a fixed backbone — just the chain of
**Cα atom positions**, one per residue — and it answers: *which amino acid should
sit at each position so that the whole thing folds back into this shape?* This is
how de-novo proteins are designed: a backbone is invented (e.g. by RFdiffusion),
inverse folding writes a sequence for it, and structure prediction checks the
sequence really folds back. ProteinMPNN recovers ~50% of the wild-type residues
of natural proteins and its designs succeed in the wet lab at high rates.

**The single dominant signal.** Why does any particular residue belong at a
particular position? The strongest rule in all of protein biophysics is the
**hydrophobic effect**: oily (hydrophobic) side chains hide from water in the
protein's **buried core**, while polar/charged side chains sit on the
water-exposed **surface**. A position's "buriedness" is well captured by how many
neighbours surround it. That one geometric quantity — **local neighbour density**
— is what our teaching model uses, and it is also the first thing any learned
model rediscovers.

## 2. The math

**Input.** A backbone of `L` residues. Residue `i` has a Cα coordinate
`r_i = (x_i, y_i, z_i)` in ångström (Å). We also carry a **native sequence**
`s_i ∈ {0..19}` (the amino acid nature used) purely to *score* our design — it is
never used to produce the design.

**Step 1 — burial as a contact count.** Two residues are *in contact* when their
Cα atoms are within a cutoff `R = 10 Å` (a standard residue-contact radius). The
**burial** of residue `i` is the number of contacts:

```
n_i = #{ j ≠ i : ||r_i − r_j||² ≤ R² }            (we compare squared distances to avoid a sqrt)
```

A large `n_i` means a packed, buried position; a small `n_i` means an exposed one.

**Step 2 — score every amino acid at every position.** Each amino acid `a`
(`a ∈ {0..19}`) has a **preferred burial** `b_a` — a small integer encoding the
neighbour density it likes best (hydrophobic Phe/Ile like `b ≈ 23–24`; charged
Glu/Asp like `b ≈ 2–4`; amphipathic residues sit in between). The score of
placing amino acid `a` at residue `i` is a **quadratic well** centred on `b_a`:

```
score(a, i) = −(n_i − b_a)²
```

The best possible score is `0` (perfect match); the further the position's burial
is from what the amino acid prefers, the more negative the score.

**Step 3 — design = per-position argmax.** The designed residue at `i` is the
amino acid with the best score:

```
design_i = argmax_a  score(a, i)
```

Ties are broken by lowest index `a` so the answer is unique and deterministic.

**Output metric — native sequence recovery.** The headline number is the fraction
of positions where the design matches the native:

```
recovery = (100 / L) · #{ i : design_i = s_i }            (rounded to an integer %)
```

Everything is **integer arithmetic** (burial counts, the squared-distance compare
uses floats but only for a `≤` test, the score and argmax are pure `int`). That is
a deliberate determinism choice (see §5).

> **Why a quadratic well and not a single hydrophobicity scalar?** A bilinear
> energy `buriedness × hydrophobicity` always selects the single *most* extreme
> residue at every buried position and the single most polar one at every exposed
> position — the whole core collapses to one amino acid. The per-amino-acid
> *preferred burial* + quadratic well makes the design **graded and diverse**:
> different burial levels select different residues. This both is more realistic
> and makes a better demo. (You can see the contrast in `src/inverse_folding.h`.)

## 3. The algorithm

```
load backbone (L residues, Cα coords, native sequence)
# Step 1: burial — all pairs
for i in 0..L-1:
    n_i = count of j≠i with ||r_i - r_j||² ≤ R²        # O(L) per i  ->  O(L²) total
# Step 2+3: per-position argmax over 20 amino acids
for i in 0..L-1:
    design_i = argmax over a in 0..19 of  -(n_i - b_a)²  # O(20) per i  ->  O(20·L) total
recovery = matches(design, native) / L
```

**Complexity.**

| Pass | Serial work | Parallel depth | Note |
|---|---|---|---|
| Burial (step 1) | `O(L²)` | `O(L)` (per output) | the dominant, all-pairs cost |
| Design (step 2+3) | `O(20·L)` | `O(20)` | tiny, fully independent |

The burial pass is the expensive one and is the analog of **message passing over
the protein contact graph** in a real GNN: every node (residue) aggregates
information from its spatial neighbours. The design pass is the per-position
**decode**. Both passes are *independent across residues* — there is no data
dependence between different `i` — which is exactly what makes the GPU mapping
trivial and correct.

## 4. The GPU mapping

This is the "score N independent items" pattern (`docs/PATTERNS.md` §1, exemplar
`1.12` Tanimoto): **one thread per residue**. Two kernels, one per pass.

```
            backbone: L residues (Cα x,y,z)
                         │
     ┌───────────────────┴───────────────────┐
     │  KERNEL 1  neighbor_kernel             │  one thread per residue i
     │  thread i loops over all L residues,   │  reads coords through a SHARED
     │  counts contacts within R  ->  n_i     │  MEMORY tile (see below)
     └───────────────────┬───────────────────┘
                         │  neighbors[0..L-1]   (global memory)
     ┌───────────────────┴───────────────────┐
     │  KERNEL 2  design_kernel               │  one thread per residue i
     │  thread i argmaxes -(n_i - b_a)² over  │  calls the SAME score core as
     │  a=0..19  ->  design_i, score_i        │  the CPU (inverse_folding.h)
     └───────────────────┬───────────────────┘
                         │
              designed[], score[]  ->  recovery
```

**Thread-to-data map.** In both kernels, residue `i = blockIdx.x * blockDim.x +
threadIdx.x`. **Block size 256** (a multiple of the 32-lane warp; 8 warps to hide
latency; good occupancy on sm_75…sm_89). **Grid** `= ceil(L / 256)`; the ragged
last block is guarded by `if (i < L)`.

**Memory hierarchy — why shared memory in kernel 1.** The naïve burial kernel has
every one of the `L` threads read all `L` coordinates from **global memory** →
`L²` global loads, each costing hundreds of cycles. Instead we **tile**: the block
cooperatively stages a tile of 256 residue coordinates into **shared memory**
(on-chip, ~100× faster than global), every thread compares its residue against
that whole tile from shared memory, then we advance to the next tile. Each global
coordinate is read **once per block** instead of once per thread. Two
`__syncthreads()` barriers bracket the tile use (one after the cooperative load so
all data is present before anyone reads it; one after the reads so no thread
overwrites the tile while another is still using it). This is the same staging
trick as a tiled matrix multiply or an N-body force kernel.

Kernel 2 needs no shared memory or atomics: each thread does a private 20-way
argmax over registers and writes one independent output.

**No CUDA library here.** This project deliberately hand-rolls both kernels (the
all-pairs neighbour count and the per-position argmax) because they *are* the
lesson. A production system would instead push this through PyTorch Geometric /
cuDNN GNN layers — see §7. The only library linked is the CUDA runtime
(`cudart`), used for `cudaMalloc`/`cudaMemcpy`/kernel launches.

## 5. Numerical considerations

**Precision.** Coordinates are `float` (PDB precision is ~0.01 Å; `float` has ~7
significant digits — plenty). The contact test is a single `float` compare
`d2 <= R²`; with well-separated synthetic coordinates no pair sits within a
rounding error of the cutoff, so CPU and GPU classify every pair identically.
Everything downstream — the burial **count**, the score `−(n−b)²`, and the argmax
— is **pure integer**.

**Determinism (PATTERNS.md §3).** The result must be byte-identical every run so
`demo/run_demo` can diff stdout. We get that for free because:

- the neighbour count is an **integer increment**, order-independent;
- the score and argmax are **integer**; integer addition/comparison is associative
  and commutative, so neither thread-scheduling order nor compiler FMA contraction
  can change the answer;
- the argmax **tie-break is explicit** (strict `>` keeps the lowest index), so a
  position whose burial is equidistant between two amino acids always picks the
  same one on CPU and GPU.

No atomics are used (each thread owns one output), so there is no floating-point
reduction-order issue at all.

**Range / overflow.** Burial counts are bounded by `L`; for any realistic problem
`|n − b|` is at most a few dozen, and its square is at most a few thousand — far
inside a 32-bit `int`. No overflow is possible at this scale.

## 6. How we verify correctness

`src/reference_cpu.cpp` contains `design_cpu()`: an obviously-correct serial
implementation of the *same* two steps. Crucially, **both** the CPU reference and
the GPU kernel `#include "inverse_folding.h"` and call the **same**
`score_aa_at_residue()` — the shared `__host__ __device__` core (the HD-core
idiom, PATTERNS.md §2). Because the math is identical and integer, the two sides
must agree **exactly**.

`main.cu` therefore verifies with **exact equality** (`==`, no tolerance) across
all three arrays: `neighbors`, `designed`, and `score`. This is the strongest
verification class in the repo (PATTERNS.md §4), justified because the operations
are integer and the scoring function is literally shared. A mismatch would mean a
real bug — an out-of-bounds index, a missing `__syncthreads()`, a divergent
formula — not a rounding difference.

A second, *scientific* check is the **native sequence recovery** (87% on the
committed sample). The synthetic backbone is built so that most native residues
genuinely fit their burial (with ~25% deliberately "mutated" to random residues),
so a recovery near ~75–90% confirms the model is doing the intended thing — not
just that CPU==GPU.

## 7. Where this sits in the real world

This is a **reduced-scope teaching model**. The real tool, **ProteinMPNN**
(Dauparas et al., *Science* 2022, Baker Lab), differs in every important way:

- **It is a learned graph neural network**, not a hand-written energy. It builds a
  k-nearest-neighbour graph over residues and runs **message-passing** layers that
  encode backbone geometry as *learned* edge features: not just Cα–Cα distance but
  virtual Cβ positions, backbone dihedrals (φ, ψ, ω), and inter-residue
  orientations. Our single "preferred burial" scalar is the crudest shadow of what
  those layers learn.
- **It decodes autoregressively.** Real ProteinMPNN predicts residues one at a
  time, conditioning each on the residues already chosen (order-agnostically), so
  neighbouring choices are *coupled*. Our per-position argmax assumes positions are
  independent — the zero-temperature, single-pass limit with no coupling.
- **Temperature-controlled sampling** yields *diverse* sequences (raise the
  temperature to sample, lower it toward argmax); **tied decoding** enforces
  symmetry for oligomers; **LigandMPNN** adds small-molecule context. Our model has
  none of these — it always returns the one argmax sequence.
- **GPU implementation.** Production inference runs the GNN with cuDNN/cuBLAS
  kernels, FP16 mixed precision, and a KV-cache across the autoregressive steps,
  batched over many backbones. We hand-roll two tiny kernels instead, to *teach the
  pattern* (independent per-residue work + a shared-memory all-pairs gather).

What our toy gets right is the **shape of the computation** — a per-residue
neighbour gather (message passing) followed by a per-residue decode — and the
**dominant biophysical signal** (hydrophobic burial). That is enough to learn the
GPU mapping honestly without pretending to be a design tool. To go further, study
the references below and the exercises in the README.

---

## References

- **ProteinMPNN** — J. Dauparas et al., "Robust deep learning–based protein
  sequence design using ProteinMPNN", *Science* 378:49–56 (2022).
  <https://github.com/dauparas/ProteinMPNN> — the model this project is a toy of;
  read `model_utils.py` for the message-passing + autoregressive decoder.
- **LigandMPNN** — Dauparas et al. (2023). <https://github.com/dauparas/LigandMPNN>
  — adds small-molecule/ligand context to the design graph.
- **ESM-IF1** — C. Hsu et al., "Learning inverse folding from millions of predicted
  structures", ICML (2022). <https://github.com/facebookresearch/esm> — an
  alternative inverse-folding model trained on predicted structures.
- **RFdiffusion** — J. Watson et al., *Nature* (2023).
  <https://github.com/RosettaCommons/RFdiffusion> — generates the backbones that
  ProteinMPNN then designs sequences for; the upstream half of the pipeline.
- **CATH / PDB** — <https://www.cathdb.info>, <https://www.rcsb.org> — where real
  backbones and native sequences come from (see `data/README.md`).
