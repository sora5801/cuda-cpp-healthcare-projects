# Push 2026-07-01 #08 -- phase2 batch6e domain6-complete + medical-ai start

> Push-note (CLAUDE.md section 7.1). A **cross-domain** batch: the last 2 physiology projects
> (completing **Domain 6, 27/27**) plus the first 4 medical-AI projects (starting **Domain 7**).
> Each worker-built; independently lead-verified.

## 1. Summary

This batch **completes Domain 6 — Computational Physiology & Systems Biology (27/27)** — the
**sixth of 14 domains** finished — and opens **Domain 7 (Medical AI & Clinical Deep Learning)**,
taking the collection to **181 -> 187 / 301 (62.1%)**. The two closing physiology projects are
statistical-workflow tools (global sensitivity, data assimilation); the four opening AI projects
are inference kernels (CNN classifiers, a transformer encoder, a GNN). Each was built in its own
folder by one worker and re-verified by the lead.

## 2. What changed

Two new projects under `projects/06-physiology-systems-biology/` (completing domain 6):

- [`6.26` Virtual Population Generation & Sensitivity Analysis](../projects/06-physiology-systems-biology/6.26-virtual-population-generation-sensitivity-analysis)
- [`6.27` Parameter Estimation & Data Assimilation](../projects/06-physiology-systems-biology/6.27-parameter-estimation-data-assimilation-for-physiological-models)

Four new projects under `projects/07-medical-ai/` (starting domain 7):

- [`7.01` Diagnostic Imaging Classifier](../projects/07-medical-ai/7.01-diagnostic-imaging-classifier)
- [`7.02` Drug-Target Interaction Prediction (GNN)](../projects/07-medical-ai/7.02-drug-target-interaction-prediction-gnn)
- [`7.03` Clinical NLP over Notes & Records](../projects/07-medical-ai/7.03-clinical-nlp-over-notes-records)
- [`7.18` Retinal Fundus AI Screening](../projects/07-medical-ai/7.18-retinal-fundus-ai-screening)

`docs/STATUS.md` -> 6 marked **done** (187/301; **domain 6 = 27/27**, domain 7 = 5/19).
`CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **6.26 Virtual Population / Sensitivity** — a **Sobol/Saltelli** global-sensitivity study over
  a virtual PK population: one thread per Saltelli evaluation (N*(k+2)=24,576 AUC integrations,
  Halton sampling), then Saltelli/Jansen index reduction. Verifies against the analytic result
  (AUC=F*Dose/CL forces S(CL)+S(F)~1). Variance-based sensitivity on the GPU.
- **6.27 Parameter Estimation (EnKF)** — an **Ensemble Kalman Filter** estimating two-element
  Windkessel (R, C) from a noisy pressure waveform: GPU-parallel ensemble RK4 forecast + host-side
  EnKF analysis. Recovers R to 0.14%, C to 2.17% from a wrong prior. The "ensemble forecast +
  host analysis" data-assimilation pattern.
- **7.01 Diagnostic Imaging Classifier** — a CNN forward pass (conv -> ReLU -> maxpool -> dense ->
  softmax) over synthetic normal/lesion patches (per-output gather, constant-memory weights,
  shared core -> exact CPU==GPU). The canonical medical-image classification kernel.
- **7.02 Drug-Target Interaction (GNN)** — a message-passing **GNN**: gather-over-CSR-edges message
  passing (ping-pong) + graph pooling + protein encoding + one-thread-per-pair DxP scoring
  (constant-memory weights, ~6e-8). Recovers the planted top drug-protein pair.
- **7.03 Clinical NLP (Transformer)** — a **multi-head self-attention** encoder block over a
  tokenized clinical-note batch, using **cuBLAS `DgemmStridedBatched`** for the QK^T/AV matmuls +
  a hand-written stable-softmax kernel (shared core -> ~1e-16). Recovers a planted coreference link.
- **7.18 Retinal Fundus Screening** — a DR-screening CNN: shared-memory-**tiled** 2-D conv+ReLU,
  max-pool, global-average-pool, FC + softmax, plus a **Grad-CAM** lesion-localization heatmap
  (shared `cnn_core.h`, ~7e-9). A 2-D extension of the 1-D-conv flagship 7.10.

All six are clearly-labeled **reduced-scope teaching versions** (synthetic data / fixed untrained
weights, labeled synthetic), with production tools (SALib, filterpy/DA, MONAI/nnU-Net, DeepPurpose,
ClinicalBERT, EyePACS/Messidor DR models) named in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/07-medical-ai/7.03-clinical-nlp-over-notes-records   # (or any of the six)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

7.03 links **cuBLAS**. The others are pure custom kernels (RK4 ensembles, CNN gather/tiling,
GNN message passing).

## 5. What to study here

Domain 6 is now a **complete tour** of physiology on the GPU (ensemble ODEs, reaction-diffusion
stencils, Monte Carlo, sparse CG, Green's functions, agent-based hybrids, sensitivity/DA
workflows). For the new domain 7, reading path: **7.01** (CNN classifier) -> **7.18** (tiled
2-D CNN + Grad-CAM) -> **7.02** (GNN message passing) -> **7.03** (transformer attention + cuBLAS).
The three AI inference styles (CNN, GNN, transformer) now each have a domain-7 example alongside
the structural-biology attention projects (2.01, 2.14) and the genomics ones (3.18).

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 6 project folders (2 in `projects/06`, 4 in `projects/07`) changed;
  no shared/root file; no artifacts.
- ✅ **Clean rebuild** (`/t:Rebuild`, fat arch list) of all 6 in both `Release|x64` and
  `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (12/12 builds), incl. the cuBLAS link in 7.03.
- ✅ All 6 **demos PASS**: GPU==CPU (classifier exact; NLP ~1e-16; sensitivity/EnKF 1e-6..1e-9;
  GNN 6e-8; retinal 1e-3).
- ✅ `verify_project.py` -> **DONE** for all 6 (comment ratios **0.79–1.06**).
- ✅ **Domain-6 sweep:** 27/27 markers `done`.
- **Workflow:** 6 agents, ~1.06M agent tokens, 440 tool uses.
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All six are **reduced-scope teaching versions**: fixed (untrained) network weights, synthetic
  cohorts/notes/graphs. Labeled synthetic; the training pipelines and production models are
  described in each THEORY.md.

## 8. Next push preview

Continue **domain 7** (medical AI) Intermediates — `7.4` GAN image synthesis, `7.5` federated
learning, `7.6` survival analysis, `7.7` multi-omics, `7.8` RL treatment policies, `7.9` edge
inference, `7.11`–`7.14`, `7.17`, `7.19`, then Advanced `7.15`, `7.16` — in ~6-project batches to
complete **domain 7 (19/19)**. Then domain 8 (neuroscience/BCI). Same workflow, lead-verified.
