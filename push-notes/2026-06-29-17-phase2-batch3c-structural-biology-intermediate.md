# Push 2026-06-29 #17 -- phase2 batch3c structural-biology intermediate

> Push-note (CLAUDE.md section 7.1). Third domain-2 batch: 6 Intermediate structural-biology
> projects, each worker-built and independently lead-verified.

## 1. Summary

Six **domain-2 (structural biology) Intermediate** projects are complete, taking the
collection to **60 -> 66 / 301 (21.9%)** and domain 2 to **19/35**. This batch is the
"modern structural ML + network analysis" cluster: flexible fitting into density, a
profile-HMM database search, diffusion+attention co-folding, antibody-library screening,
a saturation ΔΔG scan, and a dynamical cross-correlation allosteric-network engine. Two of
them implement **self-attention / FlashAttention-shaped** kernels, deepening the transformer
thread started by 2.01. Every project was built in its own folder by one worker and
re-verified by the lead.

## 2. What changed

Six new fully-implemented projects under `projects/02-structural-biology/`:

- [`2.12` Flexible Fitting / MDFF](../projects/02-structural-biology/2.12-flexible-fitting-mdff)
- [`2.13` MSA Generation Acceleration](../projects/02-structural-biology/2.13-msa-generation-acceleration)
- [`2.14` Protein-Ligand Co-Folding](../projects/02-structural-biology/2.14-protein-ligand-co-folding)
- [`2.15` Antibody Structure Prediction](../projects/02-structural-biology/2.15-antibody-structure-prediction)
- [`2.16` Stability Prediction (ΔΔG)](../projects/02-structural-biology/2.16-g-stability-prediction)
- [`2.17` Allosteric Network Analysis](../projects/02-structural-biology/2.17-allosteric-network-analysis)

`docs/STATUS.md` -> these 6 marked **done** (66/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **2.12 Flexible Fitting / MDFF** — fits an atomic model into a cryo-EM density map: per-atom
  **trilinear density-gradient force** + harmonic restraint, steepest-descent, one thread per
  atom (shared `mdff.h` core, CPU==GPU ~1.8e-15). RMSD drops 1.41 -> 0.03 A. Pairs naturally
  with the cryo-EM reconstruction (2.03) and density validation (2.22) projects.
- **2.13 MSA Generation Acceleration** — the inner loop of MSA building as a **profile-HMM
  Viterbi** database search: one block per database sequence, a shared integer DP recurrence
  (`hmm_core.h`) with shared-memory ping-pong rows and a constant-memory emission table.
  Exact CPU==GPU (max|diff|=0). A bioinformatics DP kernel cousin of Smith-Waterman (3.01).
- **2.14 Protein-Ligand Co-Folding** — a **DDIM reverse-diffusion** loop whose every step is
  a block-per-token **FlashAttention-shaped** self-attention pass (online shared-memory
  softmax) over a joint protein+ligand token sequence, with the trained score replaced by an
  analytic geometric-attention score so CPU==GPU to ~9e-16. Pose RMSD 1.55 -> 0.012 A. The
  most advanced kernel in the batch — read it after 2.01.
- **2.15 Antibody Structure Prediction** — GPU antibody-library screening by **CDR-weighted
  BLOSUM62** similarity (one thread per library antibody, query in constant memory, integer
  scoring core -> exact CPU==GPU). Recovers the planted near-copy and the shared-CDR-H3 hit.
- **2.16 Stability Prediction (ΔΔG)** — a **saturation-mutagenesis** scan scoring an L×20
  grid of (position, mutant) mutations, one thread per cell, with a transparent four-term
  physics-inspired model shared host+device. GPU==CPU within ~5e-7 kcal/mol. A clean
  "score a huge independent grid" kernel.
- **2.17 Allosteric Network Analysis** — a **Dynamical Cross-Correlation** engine: a 2-D-grid
  kernel builds the NxN residue correlation matrix (one thread per entry, shared `dcc_core.h`
  -> bit-identical), then **Floyd-Warshall** on -log|C| weights recovers the allosteric path
  and bottleneck hop from a synthetic trajectory with a planted hinge.

All six are clearly-labeled **reduced-scope teaching versions** (synthetic structures/maps/
trajectories, labeled synthetic), with the production tools (MDFF/VMD, HHblits/HMMER,
RoseTTAFold-AA/Chai, IgFold, FoldX/Rosetta-ddG, MD-TASK/Bio3D) named in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/02-structural-biology/2.14-protein-ligand-co-folding   # (or any of the six)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

No new CUDA libraries this batch — all custom kernels (shared-memory attention/DP, constant
memory, 2-D grids).

## 5. What to study here

Reading path: **2.16** (independent L×20 grid) -> **2.15** (per-item scoring + recovery) ->
**2.12** (gradient-descent fit with trilinear forces) -> **2.13** (HMM Viterbi DP, ping-pong
rows) -> **2.17** (matrix build + graph shortest-path) -> **2.14** (diffusion loop of
FlashAttention steps — the deep end). Exercise: in **2.14**, change the number of DDIM steps
and watch the folded RMSD vs. step count; in **2.17**, move the planted hinge and confirm
Floyd-Warshall finds the new bottleneck.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 6 project folders changed; no shared/root file; no artifacts.
- ✅ **Clean rebuild** (`/t:Rebuild`, fat arch list) of all 6 in both `Release|x64` and
  `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (12/12 builds).
- ✅ All 6 **demos PASS**: GPU==CPU (HMM/antibody/DCC exact; MDFF 1e-4; co-folding 1e-3;
  ΔΔG ~5e-7).
- ✅ `verify_project.py` -> **DONE** for all 6 (comment ratios **0.98–1.13**).
- **Workflow:** 6 agents, ~1.08M agent tokens, 496 tool uses.
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All six are **reduced-scope teaching versions**: analytic (untrained) attention scores in
  2.14, a small profile HMM in 2.13, a transparent physics ΔΔG model in 2.16, synthetic
  trajectories in 2.17. Labeled synthetic; production methods are described in each THEORY.md.

## 8. Next push preview

Continue **domain-2 Intermediates** (`2.18` NMR refinement, `2.20` heterogeneous cryo-EM,
`2.21` protein-nucleic-acid docking, `2.23` interaction-energy decomposition, `2.24` SAXS,
`2.25` coevolution) in ~6-project batches, then `2.26`–`2.29`, `2.31`, `2.33`, then the 4
Advanced (`2.30`, `2.32`, `2.34`, `2.35`) to finish domain 2. Same workflow, lead-verified.
