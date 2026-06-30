# 2.33 — Structure-Based Pharmacophore Modeling from MD Ensembles

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.33`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A **pharmacophore** is the 3-D arrangement of chemical features a ligand must
present to bind a receptor — hydrogen-bond donors and acceptors, hydrophobic
blobs, aromatic rings, charged groups. An **ensemble** pharmacophore is the
consensus of those features taken over many molecular-dynamics (MD) frames, so it
captures the receptor's flexibility instead of one frozen crystal snapshot. This
project takes one such query pharmacophore and **screens a library of candidate
molecules** against it, scoring each candidate with a ROCS-style **3-D Gaussian
overlap** ("color" Tanimoto). Every candidate's score is independent, so the GPU
gives each library molecule its own thread and broadcasts the query from constant
memory — the same "one query vs. N items" pattern as the Tanimoto fingerprint
search (project 1.12). It is a **reduced-scope teaching version**: the feature
points are given (we do not extract them from raw trajectories or do the DBSCAN
clustering), so the lesson is the **GPU overlap-scoring screen** at its heart.

## What this computes & why the GPU helps

Static pharmacophore models miss receptor flexibility; ensemble pharmacophore
modeling derives features from MD trajectory frames, capturing induced-fit and
cryptic-pocket binding geometries. GPU-accelerated MD generates the conformational
ensemble; GPU-parallel feature extraction (H-bond donor/acceptor, hydrophobic
contact maps) across millions of frames clusters into a consensus pharmacophore.
The resulting ensemble pharmacophore is used for 3-D similarity screening with GPU
ROCS/FastROCS against billion-compound libraries, bridging MD insights with
ultra-large-scale screening.

**The parallel bottleneck:** the **screen** — scoring one pharmacophore against an
enormous library (10⁶–10⁹ conformers). Each molecule's score is an independent sum
of Gaussian overlaps, so the screen is embarrassingly parallel: **one GPU thread
per library molecule**, with the small read-only query held in **constant memory**
(its cache broadcasts a feature warp-wide for free). That is the step this project
puts on the GPU; the feature extraction and clustering that precede it are
described in `THEORY.md` but not implemented here.

## The algorithm in brief

- **Typed feature points.** Query and library features each carry a type (donor,
  acceptor, hydrophobe, aromatic, ±charge), a 3-D center (Å), and a weight.
- **Gaussian overlap.** Two same-type features overlap by
  `w_q·w_l·exp(-α·r²)`; different types never overlap (a donor cannot satisfy an
  acceptor). `α = ln 2` so features 1 Å apart still overlap at half strength.
- **ROCS "color" Tanimoto.** Per molecule, `T = O_ql / (O_qq + O_ll − O_ql)` — the
  cross-overlap normalized by the self-overlaps, so molecule size cannot inflate
  the score.
- **GPU mapping.** One thread per library molecule; the query in constant memory;
  variable-length library feature sets stored in a flat **CSR** (offset) layout.

The full pipeline named in the catalog (dynamic feature extraction from MD,
DBSCAN ensemble clustering, SMARTS matching, common-hits / water-displacement
pharmacophores) is discussed in [THEORY.md](THEORY.md); this teaching version
implements the **overlap-scoring screen** and takes the features as input.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/structure-based-pharmacophore-modeling-from-md-ensembles.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/structure-based-pharmacophore-modeling-from-md-ensembles.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\structure-based-pharmacophore-modeling-from-md-ensembles.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/pharmacophore_sample.txt`, prints
the top hits, shows the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/pharmacophore_sample.txt` — a tiny,
  **synthetic**, offline input (one 5-feature query + 512 library molecules) so the
  demo runs with zero downloads. A near-perfect match is planted at molecule 7.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print pointers and
  instructions only (the real sources need registration + a feature-extraction
  pipeline; nothing is redistributed here).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: GPCRmd trajectory archive (https://gpcrmd.org); DUD-E
actives/decoys for validation (https://dude.docking.org); PDB structures of target
classes (https://www.rcsb.org); ZINC drug-like library for screening
(https://zinc20.docking.org).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): the
planted target **mol[7]** ranks **#1** with a score far above the random decoys,
and the run ends `RESULT: PASS`. The program computes every molecule's score on
both the **GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`)
and asserts they agree within `1e-5` — that agreement is the correctness guarantee.
Both sides call the **same** `score_molecule()` (in `src/pharmacophore.h`), so here
they agree **exactly** (`max_abs_err = 0`, shown on stderr).

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the screen, runs CPU + GPU, verifies, reports the top-K.
2. [`src/pharmacophore.h`](src/pharmacophore.h) — the **shared `__host__ __device__` scoring core** (the one true formula both sides call).
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-molecule idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel (query in constant memory) and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the loader + trusted serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **Pharmer** (https://github.com/dkoes/pharmer) — open-source pharmacophore
  search; study its feature typing and the way it indexes a library for fast
  screening.
- **fpocket / MDpocket** (https://github.com/Discngine/fpocket) — pocket detection
  across MD trajectories; the upstream step that finds *where* the pharmacophore
  features should live.
- **HTMD** (https://github.com/Acellera/htmd) — building an ensemble pharmacophore
  from GPU MD; see how consensus features are clustered over frames.
- **OpenEye ROCS / FastROCS** (https://www.eyesopen.com/rocs) — the production
  Gaussian shape+color overlap engine this project's scoring imitates.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Score one query vs. N independent items**, query in **constant memory**
(PATTERNS.md §1; shared with 1.12 Tanimoto and 12.01 spectral search). One thread
scores one library molecule via a Gaussian-overlap "color" Tanimoto; the
variable-length library is stored in a flat **CSR** layout so the kernel reads each
molecule's features from one coalesced buffer. The per-molecule physics lives in a
single **`__host__ __device__`** header so the CPU and GPU produce identical
scores. No atomics or shared memory are needed — the molecules are fully
independent.

## Exercises

1. **Scale it up.** Run `python scripts/make_synthetic.py --N 1000000` and rerun
   the demo (pass the new file as an argument). Watch the GPU kernel time barely
   move while the CPU time grows — the screen is where the GPU earns its keep.
2. **Shared-memory query.** The query is in constant memory; try staging it into
   `__shared__` at block start instead and compare. Which is faster for a 5-feature
   query, and why? (Hint: constant cache already broadcasts.)
3. **A stricter score.** Add a distance cutoff so features more than 3 Å apart
   contribute exactly zero (an early-exit). Does the ranking change? Does it speed
   up the inner loop?
4. **Vector overlap (shape).** Add an untyped "shape" Gaussian term (sum over all
   feature pairs regardless of type) and blend it with the color score — that is
   ROCS's actual `shape + color` combo.
5. **Top-K on the GPU.** Right now top-K is computed on the host. Implement a
   block-level reduction (or use CUB) to keep the best K on the device, the way a
   billion-compound screen must.

## Limitations & honesty

- **Synthetic data.** Every molecule here is generated, not real. There are no
  actual ligands, receptors, or MD trajectories; the planted "hit" is a jittered
  copy of the query. Labeled synthetic everywhere (CLAUDE.md §8).
- **Reduced scope.** This implements the **scoring screen only**. The upstream
  pipeline — extracting typed features from MD frames, DBSCAN-clustering them into
  a consensus pharmacophore, SMARTS substructure matching, the common-hits and
  water-displacement refinements — is described in `THEORY.md` but **not coded**.
- **Simplified overlap.** Real ROCS optimizes the rigid-body **alignment** of each
  conformer to maximize overlap (a Hungarian-style or quaternion optimization per
  molecule). Here features are pre-aligned in a shared frame, so we skip the
  per-molecule pose search — the single biggest simplification.
- **Not for any real screening or clinical decision.** It is study material for the
  GPU pattern, nothing more.
