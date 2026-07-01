# Push 2026-07-01 #00 -- phase2 batch5e medical-imaging analysis

> Push-note (CLAUDE.md section 7.1). Fifth domain-4 batch: 6 Intermediate imaging-analysis
> projects, each worker-built and independently lead-verified.

## 1. Summary

Six more **domain-4 (medical imaging) Intermediate** projects are complete, taking the
collection to **135 -> 141 / 301 (46.8%)** and domain 4 to **31/33** — only 2 projects from
finishing the domain. This batch is the **quantitative analysis / restoration** cluster: ASL
perfusion fitting, CT/MRI super-resolution, cross-scanner harmonization, vessel segmentation,
radiomics, and light-sheet deconvolution. It's notable for the **three real numerical bugs**
the CPU/GPU cross-check surfaced and the workers fixed (documented as teaching notes). Each was
built in its own folder by one worker and re-verified by the lead.

## 2. What changed

Six new fully-implemented projects under `projects/04-medical-imaging/`:

- [`4.23` Arterial Spin Labeling & Perfusion Imaging](../projects/04-medical-imaging/4.23-arterial-spin-labeling-perfusion-imaging)
- [`4.24` CT/MRI Super-Resolution](../projects/04-medical-imaging/4.24-ct-mri-super-resolution)
- [`4.25` Image Harmonization Across Scanners/Sites](../projects/04-medical-imaging/4.25-image-harmonization-across-scanners-sites)
- [`4.26` Vessel Segmentation & Centerline Extraction](../projects/04-medical-imaging/4.26-vessel-segmentation-centerline-extraction)
- [`4.27` Radiomics Feature Extraction](../projects/04-medical-imaging/4.27-radiomics-feature-extraction)
- [`4.29` Light-Sheet Microscopy Reconstruction](../projects/04-medical-imaging/4.29-light-sheet-microscopy-reconstruction)

`docs/STATUS.md` -> these 6 marked **done** (141/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **4.23 ASL Perfusion** — multi-delay **Buxton** kinetic-model fit: one thread per voxel runs a
  shared **Levenberg-Marquardt** solver (PLDs in constant memory) recovering CBF + arterial
  transit time. CPU==GPU ~7e-15. **Bug fixed:** naive Gauss-Newton failed to converge on the
  CBF/ATT ~1000x scale mismatch; Marquardt diagonal scaling fixed it (teaching note).
- **4.24 CT/MRI Super-Resolution** — the **ESPCN sub-pixel-convolution** upsampler (feature conv +
  ReLU, then pixel-shuffle reconstruction conv) with fixed synthetic weights, one thread per HR
  pixel. Exact CPU==GPU; 2x super-resolves a phantom, +1.23 dB PSNR over nearest-neighbour.
- **4.25 Image Harmonization** — **ComBat** statistical harmonization (NeuroComBat-faithful),
  chosen over a ~100-GPU-hour CycleGAN per §13: an ensemble of per-feature OLS + empirical-Bayes
  shrinkage (one thread per feature). CPU==GPU 7e-15. **Bug fixed:** a rank-deficient design with
  intercept broke determinism; dropping the intercept fixed it (teaching note).
- **4.26 Vessel Segmentation (Frangi)** — the **Frangi vesselness** filter: per-voxel
  finite-difference Hessian + closed-form symmetric 3x3 eigenvalues + Frangi score, one thread per
  voxel (shared `frangi.h` -> exact CPU==GPU). A synthetic vessel gives a clean single-peak ridge.
- **4.27 Radiomics** — first-order + **GLCM** texture features from a 3-D ROI: one thread per voxel
  scatters co-occurrences into a block-private shared-memory GLCM via integer atomics
  (deterministic, exactly CPU-matching), then shared count->feature reductions. The PyRadiomics core.
- **4.29 Light-Sheet Microscopy** — 2-D single-view **Richardson-Lucy** deconvolution in the
  Fourier domain with **cuFFT** (double-precision D2Z/Z2D), CPU reference via direct DFT. CPU==GPU
  ~1e-15; flux conserved; beads sharpen. **Bug fixed:** an adjoint argument-order bug that had made
  RL diverge (teaching note). A third RL/cuFFT project (cf. 4.30) with a different geometry.

All six are clearly-labeled **reduced-scope teaching versions** (synthetic phantoms/cohorts,
labeled synthetic), with production tools (BASIL/oxford_asl, ESPCN/ESRGAN, NeuroComBat, Frangi/
VMTK, PyRadiomics, RLdeconv/clij) named in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/04-medical-imaging/4.26-vessel-segmentation-centerline-extraction   # (or any)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

4.29 links **cuFFT**. The others are pure custom kernels (LM fit, sub-pixel conv, per-feature OLS,
Hessian eigen, GLCM atomics).

## 5. What to study here

Reading path: **4.26** (per-voxel Hessian eigen) -> **4.27** (GLCM atomics) -> **4.24** (sub-pixel
conv) -> **4.29** (cuFFT RL) -> **4.23** (LM fit + the scale-mismatch lesson) -> **4.25** (ComBat +
the rank-deficiency lesson). The three bug-fix teaching notes (4.23, 4.25, 4.29) are worth reading
as a set — each is a distinct way GPU/CPU divergence exposes a real algorithmic flaw.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 6 project folders changed; no shared/root file; no stray artifacts.
- ✅ **Clean rebuild** (`/t:Rebuild`, fat arch list) of all 6 in both `Release|x64` and
  `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (12/12 builds), incl. the cuFFT link in 4.29.
- ✅ All 6 **demos PASS**: GPU==CPU (radiomics GLCM exact; ASL/harmonization 7e-15; light-sheet
  1e-15; super-res/vessel 1e-6).
- ✅ `verify_project.py` -> **DONE** for all 6 (comment ratios **0.77–1.09**).
- **Workflow:** 6 agents, ~1.09M agent tokens, 487 tool uses (relaunched after a window reset;
  first attempt was killed mid-run by the usage limit).
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All six are **reduced-scope teaching versions**: fixed (untrained) super-res weights, ComBat
  instead of a deep harmonizer, small ROIs/volumes. Labeled synthetic; production scale and the
  deep-learning alternatives are described in each THEORY.md.

## 8. Next push preview

The **last 2 domain-4 projects** (`4.32` landmark detection, `4.33` real-time MRI) — completing
**domain 4 (33/33)** — then on to **domain 5 (radiation therapy & medical physics, 15 projects)**.
Four of 14 domains will be complete. Same workflow, lead-verified.
