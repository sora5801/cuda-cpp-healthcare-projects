# Push 2026-06-29 #12 -- phase2 batch2c drug-discovery intermediate

> Push-note (CLAUDE.md section 7.1). Fourth Phase-2 batch and the first **Intermediate**
> tier: 6 domain-1 projects, each built by a worker agent and independently lead-verified.

## 1. Summary

Six **domain-1 (drug discovery) Intermediate** projects are complete, taking the
collection to **31 -> 37 / 301 (12.3%)**. This batch moves past the Beginner tier into
methods that read more like real cheminformatics / ML pipelines — generative SMILES
models, a 3-D convolutional affinity scorer, Markov State Models, combinatorial library
enumeration, knowledge-graph link prediction, and route-yield scoring — while keeping
each one a single, fully-verifiable GPU teaching kernel. Every project was built in its
own folder by one worker, then the lead independently re-verified it (boundary check,
clean Release+Debug rebuild with zero warnings, demo PASS, `verify_project.py` DONE,
and a comment-marker spot-check of the multi-stage MSM project).

## 2. What changed

Six new fully-implemented projects under `projects/01-drug-discovery/`:

- [`1.10` De Novo Generative Molecular Design](../projects/01-drug-discovery/1.10-de-novo-generative-molecular-design)
- [`1.15` Protein-Ligand Binding Affinity Scoring (ML)](../projects/01-drug-discovery/1.15-protein-ligand-binding-affinity-scoring-ml)
- [`1.17` Markov State Models from MD](../projects/01-drug-discovery/1.17-markov-state-models-from-md)
- [`1.18` Fragment / Combinatorial Library Enumeration](../projects/01-drug-discovery/1.18-fragment-combinatorial-library-enumeration)
- [`1.19` Network / Polypharmacology Modeling](../projects/01-drug-discovery/1.19-network-polypharmacology-modeling)
- [`1.20` Reaction Yield / Retrosynthesis Scoring](../projects/01-drug-discovery/1.20-reaction-yield-retrosynthesis-scoring)

`docs/STATUS.md` -> these 6 marked **done** (37/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **1.10 De Novo Generative Design** — a first-order Markov SMILES language model that
  generates 4096 candidate molecules *in parallel* (one thread per molecule, per-thread
  SplitMix64 RNG, transition table in constant memory) and scores them. A shared
  `__host__ __device__` core (`generator.h`) makes GPU==CPU *exact* (4096/4096 identical).
  The most interesting idea: parallel stochastic generation that is still bit-reproducible
  because each thread owns an independent, seeded counter-based RNG stream.
- **1.15 Protein-Ligand Affinity (3-D CNN)** — a reduced 3-D convolutional scorer:
  voxelize the pocket -> Conv3D+ReLU -> global-average-pool -> dense -> logistic pKd. One
  block per pose, a deterministic gather voxelizer (no atomics), and a shared-memory tree
  reduction for the pooling. CPU/GPU agree to ~2e-15 via the HD-macro core. A first taste
  of CNN inference as a GPU kernel; look at the shared-memory reduction in the pool stage.
- **1.17 Markov State Models** — a four-stage pipeline on one set of MD-like samples:
  k-means microstate clustering -> **integer-atomic** transition counting -> MLE row-
  normalized transition matrix -> stationary distribution + slowest implied timescale.
  Every stage is matched CPU-vs-GPU *exactly* by using integer/fixed-point atomics (the
  determinism rule from `PATTERNS.md §3`). The headline is how a real analysis *pipeline*
  decomposes into independently-verifiable GPU stages.
- **1.18 Fragment / Combinatorial Enumeration** — one GPU thread per Cartesian-product
  molecule: decode a **mixed-radix index** into a fragment choice per site, sum additive
  group-contribution descriptors (MW/cLogP/TPSA/HBD/HBA) from constant-memory tables, and
  apply the Lipinski+Veber filter. Exact CPU==GPU (130/216 pass the filter). Teaches the
  "enumerate a huge combinatorial space by mapping the flat index to a thread" pattern.
- **1.19 Network / Polypharmacology (TransE)** — a knowledge-graph link-prediction model:
  one thread scores the query drug against each candidate protein tail by TransE energy
  (negative squared L2 of head+relation-tail, head/relation in constant memory), ranks
  top-K, and recovers 3/3 planted synthetic targets. Documents the small FMA-contraction
  divergence (~9.5e-7) honestly (verified within 1e-5).
- **1.20 Reaction Yield / Retrosynthesis Scoring** — GPU-batched scoring of candidate
  synthesis routes (one thread per route, grid-stride, logistic yield model in constant
  memory), with a shared `route_score()` core so CPU==GPU to ~6e-8. A planted-winner
  sample (route[0] ranks #1) makes the result legible.

All six are clearly-labeled **reduced-scope teaching versions** (toy models / synthetic
data, labeled synthetic) with the production approach described in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/01-drug-discovery/1.17-markov-state-models-from-md   # (or any of the six)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

No new CUDA libraries this batch — all six are custom kernels (constant memory, integer
atomics, shared-memory reductions, grid-stride loops).

## 5. What to study here

Reading path: **1.18** (clearest "flat index -> thread" enumeration) -> **1.20** / **1.19**
(per-item batched scoring, constant-memory models) -> **1.10** (parallel *stochastic*
generation that stays reproducible) -> **1.15** (CNN inference + shared-memory reduction)
-> **1.17** (a full multi-stage analysis pipeline, each stage exactly verified). Exercise:
in **1.10**, raise the SMILES Markov order from 1 to 2 and observe how the generated set's
validity changes; in **1.17**, vary the number of microstates k and watch the slowest
implied timescale converge.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 6 project folders changed; no shared/root file; no artifacts.
- ✅ **Clean rebuild** (`/t:Rebuild`, committed fat arch list) of all 6 in both
  `Release|x64` and `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (12/12 builds).
- ✅ All 6 **demos PASS**: rebuilt exe reproduces `expected_output.txt`, GPU==CPU
  (generative/enumeration/MSM exact; affinity 2e-15; TransE 1e-5; route 6e-8).
- ✅ `verify_project.py` -> **DONE** for all 6 (comment ratios **0.82–1.15**).
- ✅ **Spot-check** of 1.17: 96 teaching-marker hits (HD-core, integer atomics,
  determinism, launch config) across all 8 source files; shared `msm.h` core present.
- **Workflow:** 6 agents, ~1.01M agent tokens, 450 tool uses.
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All six are **reduced-scope teaching versions**: toy Markov SMILES model, a tiny
  untrained 3-D CNN, synthetic MD samples, additive descriptor tables, a small synthetic
  knowledge graph, a logistic yield surrogate. Labeled synthetic throughout; real tools
  (REINVENT, Gnina, PyEMMA, RDKit/Enamine, DeepPurpose, ASKCOS) are named in each THEORY.md.
- **Process note (honest):** batch 2c's first launch hit the account session limit mid-run
  and killed all 6 workers; the partial work was discarded (`git restore` + `git clean`,
  tree clean at 31/301) and the batch was re-run cleanly after the usage window reset. This
  is the second window-exhaustion this session — handled the same way each time.

## 8. Next push preview

Continue the **domain-1 Intermediate** tier: `1.21` (polarizable/AMOEBA MD), `1.22`
(constant-pH MD), `1.23` (QM/MM MD), `1.25` (GaMD), `1.26` (steered MD), `1.28` (covalent
docking), `1.29` (kinase selectivity) — in ~6-project batches — then the 4 Advanced domain-1
projects, then on to domain 2 (structural biology). Same workflow, lead-verified, one
push-note per batch.
