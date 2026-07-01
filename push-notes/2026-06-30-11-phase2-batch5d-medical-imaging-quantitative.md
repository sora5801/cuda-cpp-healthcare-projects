# Push 2026-06-30 #11 -- phase2 batch5d medical-imaging quantitative

> Push-note (CLAUDE.md section 7.1). Fourth domain-4 batch: 6 Intermediate quantitative-imaging
> projects, each worker-built and independently lead-verified.

## 1. Summary

Six more **domain-4 (medical imaging) Intermediate** projects are complete, taking the
collection to **129 -> 135 / 301 (44.9%)** and domain 4 to **25/33**. This batch is the
**quantitative / functional imaging** cluster: task-fMRI GLM, image-guided-surgery registration,
4D-CT, dual-energy CT, MR fingerprinting, and QSM. It exercises a nice spread of numerical
methods on the GPU — per-voxel OLS/Newton solves, ICP + Kabsch, and both **cuBLAS SGEMM**
(4.21) and **cuFFT** (4.22). Each was built in its own folder by one worker and re-verified by
the lead.

## 2. What changed

Six new fully-implemented projects under `projects/04-medical-imaging/`:

- [`4.16` Functional MRI Analysis](../projects/04-medical-imaging/4.16-functional-mri-analysis)
- [`4.17` Real-Time Intraoperative / Image-Guided Surgery](../projects/04-medical-imaging/4.17-real-time-intraoperative-image-guided-surgery)
- [`4.19` Motion-Compensated 4D-CT Reconstruction](../projects/04-medical-imaging/4.19-motion-compensated-4d-ct-reconstruction)
- [`4.20` Dual-Energy / Spectral CT Reconstruction](../projects/04-medical-imaging/4.20-dual-energy-spectral-ct-reconstruction)
- [`4.21` MR Fingerprinting Reconstruction](../projects/04-medical-imaging/4.21-mr-fingerprinting-reconstruction)
- [`4.22` Quantitative Susceptibility Mapping (QSM)](../projects/04-medical-imaging/4.22-quantitative-susceptibility-mapping-qsm)

`docs/STATUS.md` -> these 6 marked **done** (135/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **4.16 Functional MRI (GLM)** — the task-fMRI **mass-univariate GLM**: an HRF-convolved design
  matrix + per-voxel **OLS t-statistic** (one thread per voxel, shared (X^T X)^-1 in constant
  memory, shared `glm.h` -> CPU==GPU 3.7e-13). Recovers 6/6 planted-active voxels. The SPM/FSL
  activation-map core.
- **4.17 Image-Guided Surgery (ICP)** — GPU **Iterative Closest Point** rigid registration:
  per-point nearest-neighbour gather + integer fixed-point atomic cross-covariance (deterministic)
  + host 3x3 Jacobi-SVD **Kabsch** align. RMS 3.2 mm -> 0.24 mm; transform diff 0. Surface-to-image
  registration for navigation.
- **4.19 Motion-Compensated 4D-CT** — a per-pixel gather (like 4.01) that ramp-filters a
  phase-binned breathing-phantom sinogram and reconstructs naive 4D-FBP vs **motion-compensated**
  (per-phase DVF warp). Shared `mc4dct.h` -> ~2e-6; moving-nodule peak 0.897 -> 1.041 (true 1.0).
- **4.20 Dual-Energy CT** — projection-domain **material decomposition**: one thread per sinogram
  bin solves a 2x2 **nonlinear system by Newton's method** (forward model/Jacobian/step in shared
  `dect.h`, CPU==GPU ~7e-15). Water/bone basis material images.
- **4.21 MR Fingerprinting** — a dictionary-matching pipeline: Bloch-atom simulation + L2-normalize
  + **cuBLAS SGEMM** to form the whole VxD cosine-score matrix + per-voxel argmax. 64/64 voxels
  recover their ground-truth atom. **Design note:** the synthetic dictionary is greedily pruned to
  be well-separated (top1-top2 margin ~4e-3 >> SGEMM float wobble ~1e-7) so the exact-argmax check
  stays valid and stdout deterministic. Second cuBLAS project (cf. 3.11).
- **4.22 QSM** — dipole inversion: TKD + Tikhonov least-squares (closed-form + iterative gradient
  descent) recovering susceptibility chi from a 3-D field map, with **cuFFT 3-D** transforms
  (CPU==GPU RMS ~1e-16; iterative converges to the Wiener minimizer). Deconvolution in k-space.

All six are clearly-labeled **reduced-scope teaching versions** (synthetic phantoms/dictionaries,
labeled synthetic), with production tools (SPM/FSL/AFNI, 3D Slicer/IGSTK, RTK 4D-CT, GE/Siemens
DECT, MRF dictionaries, MEDI/STI Suite) named in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/04-medical-imaging/4.21-mr-fingerprinting-reconstruction   # (or any of the six)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

4.21 links **cuBLAS**, 4.22 links **cuFFT** (both in `.vcxproj` + CMake, BUILD_GUIDE §7b). The
others are pure custom kernels.

## 5. What to study here

Reading path: **4.16** (per-voxel OLS) -> **4.20** (per-bin Newton) -> **4.17** (ICP + Kabsch) ->
**4.19** (motion-compensated FBP) -> **4.22** (cuFFT dipole inversion) -> **4.21** (cuBLAS
dictionary match + the determinism-preserving dictionary-pruning trick). Exercise: in **4.16**,
change the HRF and watch which voxels survive thresholding; in **4.21**, shrink the top1-top2
margin and observe how SGEMM float-order wobble can start flipping argmax (why the pruning matters).

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 6 project folders changed; no shared/root file; no stray artifacts.
- ✅ **Clean rebuild** (`/t:Rebuild`, fat arch list) of all 6 in both `Release|x64` and
  `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (12/12 builds), incl. cuBLAS (4.21) + cuFFT (4.22).
- ✅ All 6 **demos PASS**: GPU==CPU (fMRI 1e-9; DECT 7e-15; MRF argmax exact; QSM 1e-16;
  4D-CT 1e-3; ICP transform matches).
- ✅ `verify_project.py` -> **DONE** for all 6 (comment ratios **0.88–1.04**).
- **Workflow:** 6 agents, ~1.15M agent tokens, 483 tool uses.
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All six are **reduced-scope teaching versions**: small volumes/point-clouds, pruned synthetic
  MRF dictionary, 2-D 4D-CT. Labeled synthetic; production scale described in each THEORY.md.

## 8. Next push preview

The **last 8 domain-4 projects** (`4.23` ASL, `4.24` super-resolution, `4.25` harmonization,
`4.26` vessel segmentation, `4.27` radiomics, `4.29` light-sheet, `4.32` landmark detection,
`4.33` real-time MRI) over the next ~2 batches, completing **domain 4 (33/33)**. Then domain 5
(radiation therapy). Same workflow, lead-verified.
