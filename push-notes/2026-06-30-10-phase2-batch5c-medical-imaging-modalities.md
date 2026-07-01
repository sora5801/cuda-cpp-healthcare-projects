# Push 2026-06-30 #10 -- phase2 batch5c medical-imaging modalities

> Push-note (CLAUDE.md section 7.1). Third domain-4 batch: 6 Intermediate imaging-modality
> projects, each worker-built and independently lead-verified.

## 1. Summary

Six more **domain-4 (medical imaging) Intermediate** projects are complete, taking the
collection to **123 -> 129 / 301 (42.9%)** and domain 4 to **19/33**. This batch is a tour of
**specialized imaging modalities**: single-molecule localization microscopy, digital pathology,
optical coherence tomography, photoacoustic imaging, breast tomosynthesis, and diffusion MRI.
Each modality reduces to a familiar GPU shape — per-pixel gather/backprojection, batched cuFFT,
localize-then-render, or per-voxel fit — showing how one toolbox serves very different scanners.
Each was built in its own folder by one worker and re-verified by the lead.

## 2. What changed

Six new fully-implemented projects under `projects/04-medical-imaging/`:

- [`4.10` Super-Resolution Microscopy (SMLM)](../projects/04-medical-imaging/4.10-super-resolution-microscopy-reconstruction)
- [`4.11` Digital Pathology / Whole-Slide Image Analysis](../projects/04-medical-imaging/4.11-digital-pathology-whole-slide-image-analysis)
- [`4.12` Optical Coherence Tomography Processing](../projects/04-medical-imaging/4.12-optical-coherence-tomography-processing)
- [`4.13` Photoacoustic Image Reconstruction](../projects/04-medical-imaging/4.13-photoacoustic-image-reconstruction)
- [`4.14` Digital Breast Tomosynthesis](../projects/04-medical-imaging/4.14-digital-breast-tomosynthesis)
- [`4.15` Diffusion MRI & Tractography](../projects/04-medical-imaging/4.15-diffusion-mri-tractography)

`docs/STATUS.md` -> these 6 marked **done** (129/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **4.10 SMLM (STORM/PALM)** — a `localize_kernel` (one thread per interior pixel: 3x3 local-max
  detect + 7x7 Gaussian-weighted-centroid sub-pixel fit) then a `render_kernel` (per-localization
  fixed-point atomicAdd into an 8x-upsampled image). Shared `smlm.h` -> bit-identical CPU/GPU (187
  localizations, mean_err=0). The localize-then-render super-resolution pipeline.
- **4.11 Digital Pathology (MIL)** — a gated-attention **Multiple-Instance-Learning** head
  (CLAM/ABMIL) over a bag of tile features: per-tile logit kernel (model in constant memory) +
  host softmax + fixed-point atomicAdd pooling. Deterministic GPU==CPU; attention concentrates on
  the 6 planted tumor tiles (TUMOR call). Weakly-supervised WSI classification.
- **4.12 OCT** — SD-OCT reconstruction: a dispersion-compensation/preprocess kernel + **batched
  cuFFT** (one FFT per A-scan) + magnitude kernel (naive-DFT CPU reference, shared `oct_core.h`).
  Peak depths exact CPU==GPU. Batched-FFT signal processing.
- **4.13 Photoacoustic** — 2-D **delay-and-sum backprojection**: one thread per image pixel
  gathers/interpolates every sensor's pressure trace at its travel-time delay (shared `pa_core.h`,
  ~3e-4). Ring-array data with 3 planted absorbers peaks at the strongest source. A sibling of the
  ultrasound beamformer (4.06) and the CT flagship (4.01).
- **4.14 Tomosynthesis (SART)** — limited-angle **SART** iterative reconstruction (thread-per-ray
  forward projection, thread-per-pixel backproject+update, device-resident loop), shared
  `dbt_geometry.h` -> CPU==GPU ~2e-7 on a 2-lesion breast phantom. Limited-angle recon.
- **4.15 Diffusion MRI & Tractography** — two kernels: per-voxel **DTI tensor fit** (OLS
  pseudo-inverse in constant memory + analytic 3x3 symmetric eigensolve -> FA/MD/v1) and
  deterministic **streamline tractography** (trilinear direction-field interpolation). Exact
  CPU/GPU parity; FA~0.80 recovered on the synthetic bundle.

All six are clearly-labeled **reduced-scope teaching versions** (synthetic data, labeled
synthetic), with production tools (ThunderSTORM/Picasso, CLAM/TransMIL, OCT vendor pipelines,
k-Wave, Hologic/Siemens DBT, FSL/MRtrix3) named in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/04-medical-imaging/4.15-diffusion-mri-tractography   # (or any of the six)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

4.12 links **cuFFT** (batched A-scan FFTs). The others are pure custom kernels (localize/render,
attention MIL, delay-and-sum, SART, tensor fit + tractography).

## 5. What to study here

Reading path: **4.13** (delay-and-sum, closest to the flagships) -> **4.14** (SART iterative) ->
**4.12** (batched cuFFT) -> **4.10** (localize + atomic render) -> **4.11** (attention MIL) ->
**4.15** (per-voxel eigensolve + streamline integration). Exercise: in **4.10**, lower the
detection threshold and watch localization count vs. false positives; in **4.15**, seed
tractography from a different voxel and follow the streamline.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 6 project folders changed; no shared/root file; no stray artifacts.
- ✅ **Clean rebuild** (`/t:Rebuild`, fat arch list) of all 6 in both `Release|x64` and
  `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (12/12 builds), incl. the cuFFT link in 4.12.
- ✅ All 6 **demos PASS**: GPU==CPU (SMLM/pathology/OCT-peaks exact; photoacoustic/tomosynthesis
  1e-3; DTI fit 1e-9).
- ✅ `verify_project.py` -> **DONE** for all 6 (comment ratios **0.82–1.13**).
- **Workflow:** 6 agents, ~1.18M agent tokens, 520 tool uses.
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All six are **reduced-scope teaching versions**: small frames/slides/volumes, pre-extracted
  tile features (no CNN encoder), synthetic phantoms. Labeled synthetic; production scale
  described in each THEORY.md.

## 8. Next push preview

Continue domain-4 Intermediates (`4.16` fMRI, `4.17` intraoperative guidance, `4.19` 4D-CT,
`4.20` spectral CT, `4.21` MR fingerprinting, `4.22` QSM, `4.23` ASL perfusion, `4.24`
super-resolution, `4.25` harmonization, `4.26` vessel segmentation, `4.27` radiomics, `4.29`
light-sheet, `4.32` landmark detection, `4.33` real-time MRI) in ~6-project batches to complete
**domain 4 (33/33)**. Then domain 5 (radiation therapy). Same workflow, lead-verified.
