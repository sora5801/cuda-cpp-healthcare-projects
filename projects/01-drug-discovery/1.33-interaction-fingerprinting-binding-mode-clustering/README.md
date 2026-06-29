# 1.33 — Interaction Fingerprinting & Binding-Mode Clustering

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.33`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A docking run or a molecular-dynamics trajectory throws hundreds to millions of
candidate **ligand poses** into one binding pocket. Which of them actually bind
the *same way*? This project answers that in two GPU stages. **Stage A** turns
each pose into an **interaction fingerprint (IFP)** — a bit-string that records,
residue by residue, whether the ligand makes a hydrophobic contact, a hydrogen
bond, an aromatic (π) contact, or an ionic (salt-bridge) contact. **Stage B**
clusters those bit-strings into **binding modes** with Tanimoto k-means, so poses
that light up the same residues land in the same group. It is the structural-
biology cousin of chemical-fingerprint similarity (project 1.12): same bit-vector
+ popcount machinery, but the bits now describe *geometry of interaction* rather
than *molecular substructure*.

## What this computes & why the GPU helps

Protein-ligand interaction fingerprints (IFPs) encode which residues form HBs, hydrophobic contacts, π-stacking, salt bridges, and halogen bonds with a ligand. IFPs enable rapid clustering of thousands of docking poses or MD trajectory frames into distinct binding modes, analogous to chemical fingerprints but for structural biology. GPU-parallel distance/angle evaluation over millions of frame-residue pairs makes real-time IFP generation from MD trajectories feasible. Applications include binding-mode prediction validation and SAR-IFP correlation for lead optimization.

**The parallel bottleneck:** both stages are millions of *independent* little
computations. Stage A is a `poses × residues` grid of distance tests — every cell
is independent, so it maps to one GPU thread per pose (each scanning the residue
list). Stage B's per-iteration **ASSIGN** step is one Tanimoto popcount of every
pose against every centroid — again independent per pose. With 10³–10⁶ poses ×
hundreds of residues, this is exactly the embarrassingly-parallel shape GPUs
crush; the serial CPU loop is what real tools (ProLIF on long MD trajectories)
get stuck on.

## The algorithm in brief

- **IFP generation (SIFt/PLIF-style):** for each (pose, residue) pair, test
  squared distances against per-type cutoffs gated by chemistry (a residue that
  cannot H-bond never sets the H-bond bit). One bit per interaction type per
  residue → a fixed-length IFP per pose.
- **Tanimoto IFP similarity:** distance between two IFPs is `1 − popcount(A&B)/popcount(A|B)`
  — the bit-vector Jaccard distance, evaluated with hardware popcount.
- **Consensus-bit k-means** clusters the IFPs into K binding modes. The cluster
  centroid is a **consensus fingerprint**: bit b is set iff a majority of members
  set it (an integer majority vote → fully deterministic, see THEORY §Numerics).
- Key algorithm names from the catalog: PLEC, PLIF, SIFt, Tanimoto IFP similarity,
  GPU distance/angle kernels, GPU k-means on IFP bit-vectors.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/interaction-fingerprinting-binding-mode-clustering.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/interaction-fingerprinting-binding-mode-clustering.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\interaction-fingerprinting-binding-mode-clustering.sln /p:Configuration=Release /p:Platform=x64
```

This project links only `cudart_static.lib` (it hand-rolls IFP geometry, popcount,
and k-means for teaching — no cuML/Thrust black box).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/ifp_sample.txt`, prints the
per-mode consensus contacts, shows the GPU-vs-CPU agreement check (IFPs + labels +
centroids all match), and prints a timing line.

## Data

- **Sample (committed):** `data/sample/ifp_sample.txt` — a tiny, offline,
  **synthetic** input (24-residue pocket, 120 poses from 4 planted modes) so the
  demo runs with zero downloads and recovers a *knowable* answer.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print pointers to the real
  structure sources (none are bypassed/credential-gated).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: PDB-bind complex structures (http://www.pdbbind.org.cn); KLIFS (https://klifs.net); ChEMBL bioactivity with structures (https://www.ebi.ac.uk/chembl/); BindingDB (https://www.bindingdb.org).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): four
clusters of 30 poses each, each cluster's consensus interaction pattern listed,
`cost = 2.7484`, `mode recovery = 100.00%`, and `RESULT: PASS`. The program builds
the IFPs and clusters them on **both** the GPU (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts they agree **bit-for-bit** — the
whole pipeline is integer math, so the tolerance is *exact equality*, not a float
slack (see THEORY §How we verify correctness).

## Code tour

Read in this order:

1. [`src/ifp.h`](src/ifp.h) — the shared `__host__ __device__` core: geometry→bits,
   Tanimoto distance, nearest-centroid. The single source of truth both compilers use.
2. [`src/main.cu`](src/main.cu) — loads data, runs both stages CPU + GPU, verifies, reports.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline (loader, STAGE A, k-means).
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
5. [`src/kernels.cu`](src/kernels.cu) — the four kernels (build IFP, assign, tally) + host wrappers.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

ProLIF (https://github.com/chemosim-lab/ProLIF) — protein-ligand interaction fingerprints from MD trajectories; ODDT (https://github.com/oddt/oddt) — open drug discovery toolkit with IFP; Pharmit (https://pharmit.csb.pitt.edu) — pharmacophore + shape screening; KLIFS Python / `kissim` (https://github.com/volkamerlab/kissim) — kinase IFP features.

- **ProLIF** — the reference for *which* interactions to detect and the canonical
  per-residue bit layout; study its interaction definitions (distance + angle).
- **ODDT** — shows IFP generation inside a full docking/scoring pipeline.
- **kissim (KLIFS)** — how kinase binding sites become fixed-length feature
  fingerprints you can cluster across the kinome.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Two patterns, both from the cookbook ([docs/PATTERNS.md](../../../docs/PATTERNS.md)):
**independent jobs** (one thread per pose builds an IFP / scores it against
centroids — same shape as flagship 1.12 Tanimoto), and **parallel-assign +
integer atomic reduce** for k-means (the determinism lesson of flagship 11.09:
float atomics reorder and drift, integer counts commute and stay exact). GPU
popcount (`__popcll`) powers the Tanimoto distance. The catalog also mentions cuML
k-means / RAPIDS cuDF — we hand-roll instead so nothing is a black box.

## Exercises

1. **Halogen bonds & π-stacking angles.** Add a 5th interaction type (halogen
   bond) and make the aromatic bit require a ring-normal *angle*, not just a
   distance — the real geometric criterion. How does `IFP_BITS` / `FP_WORDS` grow?
2. **Scale it up.** Run `python scripts/make_synthetic.py --per-mode 2000` and
   watch the stderr timings: at what pose count does the GPU stop being launch-bound?
3. **Bit-pack the centroids into constant memory.** The K centroids are tiny and
   read by every thread in ASSIGN — move them to `__constant__` (as flagship 1.12
   does for its query) and measure the difference.
4. **k-means++ vs. farthest-first.** Replace the deterministic farthest-first
   seeding with random k-means++ (seeded RNG) and compare the final cost over
   several seeds — does better init find a tighter clustering?
5. **A real IFP.** Use ProLIF to emit `ifp_sample.txt` rows from an actual PDBbind
   complex's docked poses, and cluster those instead of the synthetic data.

## Limitations & honesty

- **Synthetic data, labeled synthetic.** No coordinates come from a real complex;
  the sample is a geometric toy designed so the four modes are cleanly separable
  and recoverable (100% purity). Real docking poses are noisier and modes overlap.
- **Simplified interaction geometry.** We use one interaction center per residue
  and *distance-only* criteria (squared distances vs. cutoffs). Production IFPs
  (ProLIF/PLIP) check **donor–H–acceptor angles** for H-bonds and **ring-normal
  angles** for π-stacking, track many atoms per residue, and handle charge signs.
  This is a deliberate teaching reduction — the bit layout and clustering are the
  same; only the per-bit predicate is richer in practice (THEORY §Real world).
- **Consensus centroid ≠ a real molecule.** The cluster "centroid" is a majority-
  vote bit-vector, a summary, not a synthesizable pose.
- **Timing is a teaching artifact, never a benchmark** (CLAUDE.md §12). On this
  tiny sample the GPU is launch/copy-bound and slower than the CPU; the point is
  *correctness you can see* and the *parallel structure*, not the wall-clock.
- **Not for clinical or real binding-mode decisions.**
