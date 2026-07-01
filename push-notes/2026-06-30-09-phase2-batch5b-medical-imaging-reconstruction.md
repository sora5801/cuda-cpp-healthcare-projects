# Push 2026-06-30 #09 -- phase2 batch5b medical-imaging reconstruction

> Push-note (CLAUDE.md section 7.1). Second domain-4 batch: 6 Intermediate image-reconstruction
> projects, each worker-built and independently lead-verified.

## 1. Summary

Six **domain-4 (medical imaging) Intermediate** projects are complete, taking the collection
to **117 -> 123 / 301 (40.9%) — past 40%** — and domain 4 to **13/33**. This is the
**reconstruction-heavyweights** batch: iterative CT, compressed-sensing MRI, deep-learning
MRI/CT reconstruction, PET, deformable registration, and image denoising. The unifying idea is
**iterative reconstruction** — forward-project / compare / back-project loops (SIRT, FISTA,
MLEM, unrolled cascades, Demons) — several of them cuFFT-accelerated. Each was built in its own
folder by one worker and re-verified by the lead.

## 2. What changed

Six new fully-implemented projects under `projects/04-medical-imaging/`:

- [`4.02` Iterative / Model-Based CT Reconstruction](../projects/04-medical-imaging/4.02-iterative-model-based-ct-reconstruction)
- [`4.03` MRI Reconstruction with Compressed Sensing](../projects/04-medical-imaging/4.03-mri-reconstruction-with-compressed-sensing)
- [`4.04` Deep-Learning MRI/CT Reconstruction](../projects/04-medical-imaging/4.04-deep-learning-mri-ct-reconstruction)
- [`4.05` PET Image Reconstruction](../projects/04-medical-imaging/4.05-pet-image-reconstruction)
- [`4.08` Deformable Image Registration](../projects/04-medical-imaging/4.08-deformable-image-registration)
- [`4.09` Image Denoising & Restoration](../projects/04-medical-imaging/4.09-image-denoising-restoration)

`docs/STATUS.md` -> these 6 marked **done** (123/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **4.02 Iterative CT (SIRT + TV)** — **SIRT** with a total-variation prior: four kernels
  (ray-per-thread forward projection, residual, pixel-per-thread fused backprojection+update,
  ping-pong TV step), shared geometry header -> CPU==GPU 6.8e-4. RMSE-vs-truth 0.104 on a noisy
  disc phantom. The model-based-recon cousin of the FDK flagship 4.01.
- **4.03 Compressed-Sensing MRI (FISTA + cuFFT)** — single-slice single-coil Cartesian CS-MRI:
  **FISTA** with **cuFFT** for the two per-iteration 2-D FFTs + custom mask/prox/momentum
  kernels (checked vs a hand radix-2 FFT CPU reference, RMS 2.8e-8). Beats zero-filling 8.84x.
  Iterative FFT reconstruction — the canonical CS pipeline.
- **4.04 Deep-Learning Reconstruction (unrolled)** — an **unrolled cascade** of 12 stages, each
  a fixed 3x3 denoiser stencil (stand-in for a trained CNN) + a k-space **data-consistency**
  step (direct DFT stand-in for cuFFT). Shared `recon_core.h`/`dft_core.h` -> CPU==GPU ~8e-6;
  11% RMS improvement over zero-filled. The "learn to reconstruct" architecture, transparently.
- **4.05 PET Reconstruction (MLEM)** — full **MLEM**: a shared parallel-beam geometry drives
  byte-parity between the CPU reference and three GPU gather kernels (forward-project per-LOR,
  ratio, per-pixel back-project+update, no atomics -> deterministic). CPU==GPU 6.5e-5. The
  workhorse of emission tomography.
- **4.08 Deformable Registration (Demons)** — 2-D **Thirion's Demons**: GPU
  warp+force+separable-Gaussian kernels (one thread per pixel, ping-pong), shared core -> CPU==GPU
  ~5e-15 px; SSD drops 99.9%. **Bug found & fixed:** a flipped force sign (F-Mw) that had made
  the SSD diverge — caught by the CPU/GPU comparison.
- **4.09 Image Denoising (Non-Local Means)** — the catalog's named "custom CUDA block matching":
  **NLM** with one thread per output pixel (shared `nlm_core.h` -> CPU==GPU 2.4e-7). PSNR
  22.09 -> 29.99 dB on a synthetic phantom.

All six are clearly-labeled **reduced-scope teaching versions** (synthetic phantoms, fixed/
untrained denoisers), with production tools (ASTRA/RTK, BART/SigPy, MoDL/end-to-end VarNet,
STIR/CASToR, Elastix/ANTs/Plastimatch, BM3D/NLM) named in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/04-medical-imaging/4.03-mri-reconstruction-with-compressed-sensing   # (or any)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

4.03 links **cuFFT**. The others are pure custom kernels (forward/back-projection, DFT, Demons,
NLM block matching).

## 5. What to study here

Reading path: **4.09** (per-pixel NLM) -> **4.08** (Demons registration) -> the recon loops:
**4.05** (MLEM) -> **4.02** (SIRT+TV) -> **4.03** (FISTA+cuFFT) -> **4.04** (unrolled DL recon).
Read alongside flagship 4.01: filtered backprojection is one-shot, these are *iterative*.
Exercise: in **4.02**, vary the TV weight and watch the noise/edge trade-off; in **4.03**,
increase the k-space undersampling and confirm CS still beats zero-filling.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 6 project folders changed; no shared/root file; no stray image
  artifacts.
- ✅ **Clean rebuild** (`/t:Rebuild`, fat arch list) of all 6 in both `Release|x64` and
  `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (12/12 builds), incl. the cuFFT link in 4.03.
- ✅ All 6 **demos PASS**: GPU==CPU (registration 5e-15; CS-MRI 2.8e-8; NLM 2.4e-7; iterative
  CT / DL / PET 1e-3..2e-3).
- ✅ `verify_project.py` -> **DONE** for all 6 (comment ratios **0.76–1.18**).
- **Workflow:** 6 agents, ~1.08M agent tokens, 461 tool uses (relaunched after a window reset;
  first attempt was killed mid-run by the usage limit).
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All six are **reduced-scope teaching versions**: small phantoms, fixed (untrained) denoisers
  standing in for CNNs, single-coil/single-slice MRI, 2-D registration. Labeled synthetic;
  production scale described in each THEORY.md.

## 8. Next push preview

Continue domain-4 Intermediates (`4.10` super-resolution microscopy, `4.11` digital pathology,
`4.12` OCT, `4.13` photoacoustic, `4.14` tomosynthesis, `4.15` diffusion MRI/tractography, …)
in ~6-project batches through to `4.33`. Same workflow, lead-verified, one push-note per batch.
