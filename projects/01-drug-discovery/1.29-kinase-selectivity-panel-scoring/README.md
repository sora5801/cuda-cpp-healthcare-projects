# 1.29 — Kinase Selectivity Panel Scoring

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.29`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A kinase inhibitor is only useful if it hits the kinase you *want* and leaves the
~500 others alone — otherwise you get toxicity. This project profiles **one
compound against a panel of N kinases at once**: for each kinase it predicts a
binding affinity (a pKd/pIC50-like number) from a **kinase–ligand interaction
fingerprint (IFP)**, then summarizes the whole panel with the canonical kinome
**S-score** (the fraction of kinases the compound binds). Every kinase is scored
**independently from the same compound**, so the work maps perfectly onto the GPU:
**one thread per kinase**, with the compound held in **constant memory**. The
result is deterministic and verified bit-for-bit against a plain-C++ reference.

## What this computes & why the GPU helps

Kinases share highly similar ATP-binding pockets, which is exactly why selectivity
is hard: a scaffold that hydrogen-bonds to the hinge of one kinase tends to do so
for many. Production pipelines profile a compound across 500+ kinase structures by
(1) docking it against every kinase model, (2) featurizing each pose as a KLIFS
interaction fingerprint, and (3) scoring/rescoring affinity — turning days of work
into minutes on a GPU.

**The parallel bottleneck:** the *panel sweep*. Scoring kinase *i* needs only the
compound and kinase *i*'s pocket — there is **no dependency between kinases**. With
hundreds of kinases (and, in practice, thousands of compounds × thousands of docked
poses), this is an embarrassingly parallel map: assign **one kinase to one GPU
thread**. Because the compound is identical for every thread, it lives in
**constant memory** (a hardware-broadcast cache), so reading it costs one
transaction per warp instead of one global load per thread. This is the same
"score one query vs N items" pattern as flagship `1.12` (Tanimoto search).

## The algorithm in brief

- **Interaction fingerprint (IFP) scoring.** The compound carries integer
  pharmacophore *offers* (donors, acceptors, hydrophobic/aromatic contacts,
  halogen, hinge motif). Each kinase pocket carries integer *requirements*. The
  per-kinase raw score is `bias + Σ_f min(offer_f, need_f) · weight_f` — you only
  score the overlap you can actually form.
- **Affinity mapping.** An affine map turns the raw score into a predicted
  `pK = 4.000 + 0.050 · raw`, in fixed-point milli-units (exact integers).
- **Selectivity S-score** (Karaman et al. 2008): `S = (#kinases with pK ≥ 6.000) / N`.
  Smaller S ⇒ more selective.
- **Top-K ranking** of the most potently bound kinases (deterministic, ties broken
  by index).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/kinase-selectivity-panel-scoring.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/kinase-selectivity-panel-scoring.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\kinase-selectivity-panel-scoring.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/kinase_panel_sample.txt`, prints
the selectivity summary and top-5 hits, shows the GPU-vs-CPU agreement check, and
prints a timing line.

## Data

- **Sample (committed):** `data/sample/kinase_panel_sample.txt` — a tiny,
  **synthetic** 16-kinase panel + one compound so the demo runs with zero
  downloads. ABL1 is engineered to be the single strongest hit.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print where to get the real
  KLIFS / KINOMEscan / ChEMBL kinase data (registration required for some).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: KLIFS — kinase-ligand interaction fingerprinting database
(https://klifs.net); KINOMEscan — 468-kinase selectivity data; ChEMBL kinase
activity (https://www.ebi.ac.uk/chembl/); DTC Drug-Target Commons kinase panel
(https://drugtargetcommons.fimm.fi).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
1.29 -- Kinase Selectivity Panel Scoring
panel: 1 compound vs 16 kinases (8-feature interaction fingerprint)
S-score(pK>=6.000) = 1/16 = 0.062  (lower = more selective)
top-5 most potently bound kinases:
  #1  ABL1        pK = 6.050  [HIT]
  #2  SRC         pK = 5.950
  #3  PDGFRA      pK = 5.750
  #4  KIT         pK = 5.650
  #5  LCK         pK = 5.500
RESULT: PASS (GPU matches CPU exactly: per-kinase pK, hit flags, S-count)
```

The program computes the result on both the **GPU** (`src/kernels.cu`) and a
**CPU reference** (`src/reference_cpu.cpp`). Because both call the *same*
`__host__ __device__` integer scoring physics, agreement is **exact** (tolerance
= 0): every predicted pK, every hit flag, and the S-count match bit-for-bit. That
agreement is the correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the panel, runs CPU + GPU, verifies, reports.
2. [`src/selectivity_core.h`](src/selectivity_core.h) — **the shared
   `__host__ __device__` scoring physics** (the one true formula both sides call).
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel (one thread per kinase) and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline + loader.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **AutoDock-GPU** (https://github.com/ccsb-scripps/AutoDock-GPU) — GPU docking; the
  per-kinase docking step that would feed the IFPs we score here. Learn how a real
  scoring function and pose search work on the GPU.
- **KLIFS / `kissim`** (https://github.com/volkamerlab/kissim) — the real kinase
  structural interaction fingerprints our toy IFP abstracts. Learn the 85-residue
  KLIFS pocket definition and bit-level IFP encoding.
- **KinoML** (https://github.com/openkinome/kinoml) — machine-learned kinase
  activity prediction; the modern replacement for our hand-set affine affinity map.
- **HTMD** (https://github.com/Acellera/htmd) — GPU kinome docking workflows; how the
  panel sweep is orchestrated at scale.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Score one query vs N independent items, with the query in constant memory** — one
GPU thread per kinase, grid-stride loop to cover any panel size, and an
**integer/fixed-point** computation so the GPU and CPU results are deterministic and
bit-identical (the S-count reduction uses integer addition, which commutes). The
catalog also mentions cuML (ML training) and Thrust top-K; this teaching version
keeps the ranking on the host (`std::partial_sort`) and points at those libraries
in THEORY's "real world" section.

## Exercises

1. **Bigger panel.** Edit `scripts/make_synthetic.py` to emit 512 kinases and rerun
   `make_synthetic.py`; watch the GPU kernel time stay flat while the CPU grows.
   (Update `demo/expected_output.txt` from a real run if you commit it.)
2. **Sharpen the affinity map.** Replace the affine `pK = 4 + 0.05·raw` with a
   logistic curve (still integer fixed-point) and discuss how it changes the S-score.
3. **A second selectivity metric.** Add the **Gini coefficient** or **selectivity
   entropy** (Uitdehaag & Zaman 2011) over the per-kinase affinities, computed with a
   deterministic integer reduction.
4. **Weighted Tanimoto IFP.** Swap the `min(offer,need)` overlap for a bitwise
   weighted Tanimoto over real KLIFS-style bit IFPs (reuse the `__popcll` idea from
   project `1.12`).
5. **Profile the constant-memory win.** Move the compound from constant memory into a
   global buffer and measure the difference in Nsight Compute on a large panel.

## Limitations & honesty

- **Synthetic data, toy IFP.** The committed panel is **synthetic and labeled as
  such**; the 8-channel fingerprint, integer requirements, and the affine pK map are
  teaching simplifications, **not** a fitted QSAR model and **not** real
  pharmacology. The kinase *names* are real, but their pocket vectors here are
  invented to make ABL1 the clear hit.
- **No docking, no structure.** Real pipelines *dock* the compound to generate the
  IFP per pose; we start from pre-baked feature vectors. The GPU mapping (one thread
  per kinase) is the same; only the per-kinase work changes.
- **Affinity is ordinal, not absolute.** The predicted pK values are monotonic in the
  match score; treat them as a ranking, not a measured Kd.
- **Not for any decision.** Nothing here may inform a real medical or
  drug-development decision (CLAUDE.md §8).
