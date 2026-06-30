# THEORY — 2.16 ΔΔG Stability Prediction

> A deep, didactic walk from the biology of protein stability to the CUDA mapping
> of a saturation-mutagenesis scan. This project ships a **reduced-scope teaching
> version**: the per-mutation model is a transparent, physics-inspired scoring
> function, not a trained predictor. The CUDA lesson — evaluating an `L × 20` grid
> of independent mutation scores in parallel — is identical to what a production
> model (ThermoMPNN, ProteinMPNN-ddG, ESM-1v) does. Read this with `src/ddg_model.h`
> open; that header is the single source of truth for the math below.

---

## The science

A folded protein sits in an equilibrium between its compact **folded** state and
an unfolded **denatured** ensemble. The thermodynamic stability of the fold is the
free-energy difference

```
ΔG_fold = G_unfolded − G_folded   (kcal/mol, positive for a stable protein)
```

A **single-point mutation** (replacing one amino acid with another) perturbs this
balance. The change in stability is

```
ΔΔG = ΔG_fold(mutant) − ΔG_fold(wild-type)        (kcal/mol)
```

with the sign convention used throughout this project (the ThermoMPNN / "mutant
minus wild-type" convention):

- **ΔΔG > 0** → the mutation is **stabilising** (the mutant folds more tightly);
- **ΔΔG < 0** → the mutation is **destabilising** (the common case — a well-evolved
  protein is near a local optimum, so most random changes hurt);
- **ΔΔG = 0** → the trivial "mutation" of a residue to itself.

Why anyone cares: ΔΔG drives **protein engineering** (designing stabilising
mutations for therapeutic antibodies, industrial enzymes), and it explains
**disease variants** (a destabilising missense mutation can unfold a protein and
cause loss of function). A **saturation-mutagenesis scan** asks the question
exhaustively: *for every position, what does each of the 20 substitutions do?*
The output is a deep-mutational-scan heatmap, `L × 20` cells.

### The structural intuition our model encodes

We do not model atoms. We capture four well-known physical drivers of stability,
each gated by how **buried** the residue is (core vs. surface), because the core
is where packing is tight and mutations bite hardest:

1. **Hydrophobic burial.** Burying a hydrophobic side chain is favourable
   (the hydrophobic effect); burying a polar/charged one is not.
2. **Packing / volume strain.** In the tightly packed core, changing side-chain
   volume either over-packs (steric clash) or under-packs (creates a cavity) —
   both cost energy, scaling with the squared volume change.
3. **Backbone disruption (Pro/Gly).** Proline locks the backbone dihedral and
   cannot donate a backbone H-bond; glycine is uniquely flexible. Introducing
   either into a structured core is destabilising.
4. **Buried-charge desolvation.** Burying a net charge with no solvent to screen
   it is expensive.

---

## The math

Let a mutation be `(wt → mut)` at a position with burial fraction `b ∈ [0,1]`
(`1` = fully buried core, `0` = fully exposed). Using the per-residue property
tables in `src/ddg_model.h` — hydropathy `h(·)` (Kyte-Doolittle), volume `V(·)`
(Å³), formal charge `q(·)` (e), and a Pro/Gly indicator `pg(·)` — the **raw**
score is a sum of four terms (all kcal/mol):

```
ΔΔG_raw(wt, mut, b) =  w_h · b · ( h(mut) − h(wt) )                # hydrophobic burial
                     − w_V · b · ( V(mut) − V(wt) )²               # packing strain
                     − w_P · b · pg(mut)                           # Pro/Gly backbone penalty
                     − w_q · b · ( |q(mut)| − |q(wt)| )            # buried-charge desolvation
```

with the fixed weights `w_h = 0.10`, `w_V = 0.0008`, `w_P = 2.50`, `w_q = 1.80`.
Note every term is either a **difference** of wild-type and mutant properties or
multiplied by a mutant-only factor, so the self-mutation `mut == wt` gives exactly
`ΔΔG_raw = 0`.

The raw score is then passed through a smooth **bounded squashing** so the output
stays in a physically plausible window (about ±8 kcal/mol, matching how real ΔΔG
measurements cluster):

```
ΔΔG(wt, mut, b) = S · tanh( ΔΔG_raw / S ),   S = 8 kcal/mol
```

For small `|raw|` this is ≈ linear (the physics reads through directly); for large
`|raw|` it saturates smoothly so no single term produces an absurd value.

> These weights and tables are **physical priors, not fit to any dataset.** That
> is deliberate: it keeps every number readable. A real model replaces this whole
> expression with a learned function — see "Where this sits in the real world."

---

## The algorithm

```
load protein  ->  (L residues; wt_code[p], buried[p] for each position p)
for each position p in 0..L-1:            # outer
    for each amino acid a in 0..19:       # inner
        score[p*20 + a] = ddg_predict(wt_code[p], a, buried[p])
report: top-K most negative scores; aggregate statistics
verify: max |score_cpu - score_gpu| <= tolerance
```

**Complexity.** The scan is `L × 20` cells, each `O(1)` (a handful of table lookups
and floating-point ops). So the work is `Θ(L)` serially (20 is a constant), and the
grid is **embarrassingly parallel**: no cell depends on another.

- **Serial (CPU):** one thread walks all `20L` cells → `Θ(L)` time.
- **Parallel (GPU):** `20L` threads, each computing one cell → `Θ(1)` depth
  (ignoring launch/transfer overhead), `Θ(L)` total work. This is the
  textbook "map" pattern.

---

## GPU mapping

This is the "score one query vs N items, each independent" pattern
(`docs/PATTERNS.md` §1; exemplar flagship `1.12` Tanimoto). The structure is fixed
and every `(position, mutant-AA)` query is scored in parallel — exactly the
"batched masked prediction" a real saturation scan performs.

### Thread → data mapping

A 2-D thread block tiles the `(amino-acid, position)` grid:

```
block = (AA_LANES = 32, POS_PER_BLK = 8)     # 256 threads/block
a = blockIdx.x * 32 + threadIdx.x            # amino-acid column (0..19 used)
p = blockIdx.y * 8  + threadIdx.y            # residue position (grid-strided in y)
out[p*20 + a] = ddg_predict(c_wt[p], a, c_buried[p])
```

- `threadIdx.x` spans amino acids. We use **32 lanes** (one warp) even though we
  only need `NUM_AA = 20`, so the x-dimension is warp-aligned; lanes 20–31 are
  masked off by `if (a >= NUM_AA) return;`. (Wasting 12/32 lanes is the price of
  warp alignment for a tiny, clear kernel — an exercise suggests packing instead.)
- `threadIdx.y` spans positions, and a **grid-stride loop** over `p` lets a
  fixed-height grid cover any length `L`.

### Memory hierarchy — why constant memory

The per-residue features `wt_code[p]` and `buried[p]` are:

- **read-only** during the launch,
- **tiny** (a few KB for any teaching-sized protein), and
- **reused by many threads** — all 20 amino-acid threads for position `p` read the
  *same* two values.

That is the textbook case for **constant memory**: its hardware broadcast cache
serves one address to an entire warp in a single transaction. So we place them in
`__constant__ int c_wt[MAX_RESIDUES]` and `__constant__ float c_buried[MAX_RESIDUES]`
(32 KB total at `MAX_RESIDUES = 4096`, comfortably inside the 64 KB constant bank).
The output grid is the only global-memory traffic: one coalesced write per thread.

There is **no shared memory and no atomics** — the cells are fully independent, so
there is nothing to reduce or synchronise. (Contrast with the k-means flagship
`11.09`, where many threads accumulate into shared centroids and need atomics.)

### Why the property tables are accessor functions

A namespace-scope `constexpr float[]` has **host linkage**; nvcc rejects reading it
from `__device__` code ("identifier undefined in device code"). The portable
single-source-of-truth idiom (used in `src/ddg_model.h`) wraps each table in a
`__host__ __device__` inline accessor whose `static constexpr` local array is
materialised in whichever address space the caller compiles to. Alternatives —
a duplicate `__constant__` device copy, or passing the tables as kernel arguments —
are heavier and less DRY for a fixed 20-entry table.

---

## Numerical considerations

- **Precision.** All arithmetic is single-precision `float`. ΔΔG is an inherently
  approximate quantity (experimental error bars are ~0.5 kcal/mol), so FP32 is more
  than enough; FP64 would buy nothing here.
- **The shared core guarantees parity.** The CPU reference and the GPU kernel call
  the **same** `ddg_predict()` from `ddg_model.h` (the HD-macro idiom,
  `docs/PATTERNS.md` §2). The four additive terms are identical integer-indexed
  table math, so they agree bit-for-bit.
- **The one wobble: `tanhf`.** The bounded squashing uses `tanhf`. Under nvcc this
  is the device transcendental intrinsic; under cl.exe it is `<cmath>`'s float
  `tanh`. Those two implementations — plus **fused-multiply-add (FMA)** contraction
  of `raw / S` on the device — can differ in the last bit. In practice the CPU/GPU
  ΔΔG values differ by `~5e-7` kcal/mol (see the demo's `max_abs_err`), which is
  ~six orders of magnitude below experimental error: **physically negligible**.
- **Determinism.** Each cell is computed by exactly one thread from read-only
  inputs; there is no atomic accumulation and no reduction order to vary. The
  stdout report (top-K with deterministic tie-breaks on `(position, mutant)`, and
  integer counts) is therefore byte-identical every run. Timing and the floating
  `max_abs_err` go to **stderr** so they never perturb the diffed stdout
  (`docs/PATTERNS.md` §3).

---

## How we verify correctness

Two independent checks:

1. **CPU == GPU.** `main.cu` runs `ddg_scan_cpu()` and `ddg_scan_gpu()` over the
   same protein and computes `max_abs_err` across all `L × 20` cells. We require it
   `≤ 1e-3` kcal/mol — a small **physical** tolerance chosen to absorb the `tanhf`/
   FMA divergence above while still catching any real bug (`docs/PATTERNS.md` §4).
   The observed error (`~5e-7`) sits far inside it.

2. **Known-answer sanity (the science, not just the plumbing).**
   - The **self-mutation** cell `(p, a == wt)` is exactly `0` by construction — a
     built-in invariant the model cannot violate.
   - The synthetic protein plants buried bulky hydrophobic anchors (Leu7, Phe13)
     and a hydrophobic core stripe; the scan's **most destabilising** hits are
     exactly those buried aromatic/hydrophobic residues mutated to **glycine**
     (W17G, Y9G, F13G, …), which is the physically correct answer (destroying core
     packing). Recovering the planted signal validates the model end-to-end.

---

## Where this sits in the real world

This project is a **teaching stand-in**. Production ΔΔG prediction does not use a
hand-weighted four-term formula; it **learns** the map from data:

- **ThermoMPNN** and **ProteinMPNN-ddG** take a fixed protein backbone, run a
  graph neural network (ProteinMPNN) to produce a per-residue structural
  *embedding*, and predict ΔΔG for every mutation from that embedding — the same
  `L × 20` saturation scan this project performs, but with a learned per-residue
  representation in place of our single `buried` scalar.
- **ESM-1v / EVmutation** are sequence-only: they score a mutation by the change in
  a protein language model's log-likelihood (a "zero-shot" stability proxy), no
  structure required.
- **FoldX / Rosetta Cartesian-ddG** are physics-based empirical force fields that
  repack side chains and minimise energy — slower but mechanistic.

What stays the same in all of them is the **GPU pattern**: a structure (or
sequence) is fixed, and the scan evaluates a large batch of independent
`(position, mutant)` predictions. To make this project closer to real practice you
would (a) replace `buried` with a real per-residue feature vector (relative solvent
accessibility from DSSP/`freesasa`, secondary-structure class, contact number),
(b) replace `ddg_predict` with a small trained MLP/GNN whose weights load from a
file, and (c) keep the exact same kernel launch geometry. The data the model is
trained and benchmarked on — Protherm/ProThermDB, the Megascale set, ProteinGym,
S669 — is listed in `data/README.md`.

> **Honesty.** Our scores are *not* validated ΔΔG predictions and must never be
> used to choose real mutations. They exist to make the parallel scan runnable and
> the result interpretable. See the README "Limitations & honesty".
