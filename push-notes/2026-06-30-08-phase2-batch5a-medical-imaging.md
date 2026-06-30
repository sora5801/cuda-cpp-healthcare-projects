# Push 2026-06-30 #08 -- phase2 batch5a medical-imaging

> Push-note (CLAUDE.md section 7.1). First Phase-2 batch in **Domain 4 (Medical Imaging &
> Image Reconstruction)**: 6 Beginner projects, each worker-built and lead-verified.

## 1. Summary

The build-out crosses into its **fourth domain**. Six **domain-4 (medical imaging)** Beginner
projects are complete, taking the collection to **111 -> 117 / 301 (38.9%)** and domain 4 to
**7/33** (the flagship `4.01` CT FDK backprojection was already done). This batch is a tour of
**image formation / rendering on the GPU**: ultrasound beamforming, deep-learning segmentation,
marching-cubes surface extraction, DRR generation, deconvolution microscopy, and a virtual-
colonoscopy ray-caster. The dominant pattern is the **per-pixel / per-voxel gather** (the same
shape as the CT-backprojection flagship). Each was built in its own folder by one worker and
re-verified by the lead.

## 2. What changed

Six new fully-implemented projects under `projects/04-medical-imaging/`:

- [`4.06` Ultrasound Beamforming](../projects/04-medical-imaging/4.06-ultrasound-beamforming)
- [`4.07` Medical Image Segmentation (Deep Learning)](../projects/04-medical-imaging/4.07-medical-image-segmentation-deep-learning)
- [`4.18` Image-Based 3D Printing / Model Generation](../projects/04-medical-imaging/4.18-image-based-3d-printing-model-generation-for-surgery)
- [`4.28` GPU-Accelerated DRR Generation for 2D/3D Registration](../projects/04-medical-imaging/4.28-gpu-accelerated-drr-generation-for-2d-3d-registration)
- [`4.30` Deconvolution Microscopy](../projects/04-medical-imaging/4.30-deconvolution-microscopy)
- [`4.31` Virtual Colonoscopy & CT Colonography](../projects/04-medical-imaging/4.31-virtual-colonoscopy-ct-colonography)

`docs/STATUS.md` -> these 6 marked **done** (117/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **4.06 Ultrasound Beamforming** — **Delay-and-Sum**: one thread per image pixel loops over
  all 64 transducer elements, computes the round-trip focal delay, linearly interpolates each
  element's RF trace, and sums (shared `beamform.h` -> CPU==GPU ~1.5e-4). Recovers the embedded
  point scatterer at (3.9, 20.1) mm vs truth (4.0, 20.0). The gather pattern, applied to US.
- **4.07 DL Segmentation** — a fixed-weight 2-layer 3x3x3 conv head (Gaussian denoise -> local
  -mean threshold + ReLU + per-voxel argmax), one thread per voxel (3-D stencil/gather), weights
  in constant memory. Label maps match CPU exactly; **Dice = 0.9609** vs a synthetic lesion sphere.
- **4.18 Marching Cubes** — isosurface extraction for surgical 3-D-print models via the
  **count -> prefix-sum -> scatter compaction** idiom (hand-rolled deterministic scan), shared
  core -> exact CPU==GPU meshes on a synthetic sphere (cross-checked against analytic 4*pi*r^2).
  (The worker hand-rolled the scan rather than pull in Thrust, to avoid changing shared build
  flags — a documented exercise; see BUILD_GUIDE §7c for the Thrust route.)
- **4.28 DRR Generation** — a cone-beam **Digitally Reconstructed Radiograph** renderer: one
  thread per detector pixel, each ray-marching through a CT volume with tri-linear interpolation
  (shared `drr_core.h` -> CPU==GPU 8.3e-7). The forward model behind 2-D/3-D registration.
- **4.30 Deconvolution Microscopy** — **Richardson-Lucy** deconvolution via **cuFFT**
  FFT-convolution (PSF transfer function once; forward `H` and adjoint `conj(H)` per iteration;
  1/N folded into a custom complex-multiply kernel), checked vs a direct-convolution CPU
  reference (worst error 1.3e-13). A second, iterative use of cuFFT.
- **4.31 Virtual Colonoscopy** — a GPU **volume ray-caster** rendering one fly-through frame:
  each thread marches a ray to the air->wall iso-surface, computes a central-difference gradient
  normal, and does Blinn-Phong headlamp shading. A planted polyp is the known answer (~0.73 vs
  ~0.36 brightness). CPU==GPU to FP32 (~4.8e-7).

All six are clearly-labeled **reduced-scope teaching versions** (synthetic phantoms/volumes,
labeled synthetic), with production tools (PixelFlow/Verasonics, nnU-Net/MONAI, VTK marching
cubes, Plastimatch DRR, DeconvolutionLab/Huygens, VMTK/3D Slicer) named in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/04-medical-imaging/4.30-deconvolution-microscopy   # (or any of the six)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

4.30 links **cuFFT** (`.lib` in both `<Link>` sections + `CMakeLists.txt`). The others are pure
custom kernels (gather/ray-march, 3-D stencil, prefix-sum compaction).

## 5. What to study here

Reading path: **4.06** (delay-and-sum gather) -> **4.28** (ray-march DRR) -> **4.31** (ray-cast
+ shading) -> **4.07** (3-D conv stencil + Dice) -> **4.18** (marching cubes via prefix-sum
compaction) -> **4.30** (iterative FFT deconvolution). Read alongside flagship **4.01** to see
the gather/backprojection pattern reused across modalities. Exercise: in **4.06**, move the
scatterer and confirm the focal spot tracks; in **4.30**, increase the RL iteration count and
watch the restored image sharpen.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 6 project folders changed; no shared/root file; **no stray image
  artifacts** (the `.gitignore` netpbm rule + running demos from inside each project folder kept
  rendered outputs out of the tree — important for an imaging domain).
- ✅ **Clean rebuild** (`/t:Rebuild`, fat arch list) of all 6 in both `Release|x64` and
  `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (12/12 builds), incl. the cuFFT link in 4.30.
- ✅ All 6 **demos PASS**: GPU==CPU (segmentation/marching-cubes exact-or-1e-3; beamform 1.5e-4;
  DRR 8.3e-7; deconvolution 1.3e-13; colonoscopy 4.8e-7).
- ✅ `verify_project.py` -> **DONE** for all 6 (comment ratios **0.62–1.15**).
- **Workflow:** 6 agents, ~1.08M agent tokens, 459 tool uses.
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All six are **reduced-scope teaching versions**: synthetic phantoms, fixed (untrained) conv
  weights, small volumes. Labeled synthetic; production scale described in each THEORY.md.

## 8. Next push preview

Continue domain-4 Intermediates — the reconstruction heavyweights (`4.2` iterative CT, `4.3`
compressed-sensing MRI, `4.5` PET, `4.8` deformable registration, `4.9` denoising, `4.13`
photoacoustic, …) in ~6-project batches. Several will use cuFFT/cuBLAS. Same workflow,
lead-verified, one push-note per batch.
