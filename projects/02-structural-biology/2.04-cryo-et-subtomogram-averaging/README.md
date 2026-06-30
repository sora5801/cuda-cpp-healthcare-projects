# 2.4 — Cryo-ET Subtomogram Averaging

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.4`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

In cryo-electron tomography the same molecular machine appears thousands of times
inside one extremely noisy 3-D tomogram, each copy at a random orientation.
**Subtomogram averaging (STA)** recovers a clean structure by *aligning* every
noisy copy to a common reference and *averaging* them — noise cancels as `1/√N`,
signal reinforces. This project implements the **align + average** inner loop as
a **reduced-scope teaching version**: each candidate cube is matched to the
reference by an in-plane rotation search, and for each trial rotation the GPU
computes the **cross-correlation over all 3-D shifts in Fourier space** using
**cuFFT** (the cross-correlation theorem). A plain-C++ direct correlation is the
trusted baseline; the synthetic data has a known answer so you can *see* the
right poses recovered. The lesson is twofold: how an FFT library turns an
`O(V²)` shift search into `O(V log V)`, and how to use that library without it
being a black box.

## What this computes & why the GPU helps

Cryo-electron tomography (cryo-ET) images entire cells or organelles, and subtomogram averaging (STA) extracts repeating structural units from noisy 3D tomograms by aligning and averaging thousands of subtomograms. GPU acceleration applies to: (1) tomogram reconstruction from tilt series (weighted back-projection or SART), (2) template matching for particle picking, and (3) subtomogram alignment (cross-correlation in Fourier space). RELION-4 extended STA; the IsoNet neural network corrects missing wedge artifacts with GPU inference.

**The parallel bottleneck:** the **alignment search** — for every (candidate,
trial-angle) pair, finding the translation that maximizes cross-correlation
against the reference. Done directly this is `O(V²)` per job (V = d³ voxels, V
shifts × O(V) per shift). The GPU instead computes the correlation at **all
shifts at once** via the cross-correlation theorem: `corr = IFFT(conj(FFT(ref)) ·
FFT(cand))`, an `O(V log V)` operation that **cuFFT batches** across all
candidates and angles in a single launch. Rotation, the per-frequency complex
multiply, and the peak reduction are tiny custom kernels around that library call.

## The algorithm in brief

- **Zero-mean** the reference and every candidate (so correlation peaks are meaningful).
- **Rotate** each candidate by each trial angle about z (bilinear interpolation).
- **cuFFT R2C** (batched) every rotated cube and the reference.
- **conj(ref) · cand** per frequency — the cross-correlation theorem.
- **cuFFT C2R** (batched) back to a correlation field; the **peak** is the best
  shift, the value at `(0,0,0)` is the zero-shift score; normalize to **NCC**.
- **argmax over angles** → each candidate's pose; **average** the aligned cubes
  → the refined reference.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/cryo-et-subtomogram-averaging.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/cryo-et-subtomogram-averaging.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\cryo-et-subtomogram-averaging.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the result, shows the
GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/` — a tiny, offline input so the demo runs
  with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented, idempotent).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: EMDB STA deposits (https://www.ebi.ac.uk/emdb/); EMPIAR-10064 and related datasets (https://www.ebi.ac.uk/empiar/); SHREC subtomogram challenge datasets (verify URL); CryoDRGN-ET benchmark (https://github.com/ml-struct-bio/cryodrgn).

## Expected output

Success looks like `demo/expected_output.txt`. The program computes the result on
both the **GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`)
and asserts they agree within the documented tolerance — that agreement is the
correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
2. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
3. [`src/kernels.cu`](src/kernels.cu) — the kernel(s) and host wrapper.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline.
5. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

RELION-4 STA (https://github.com/3dem/relion) — Bayesian subtomogram averaging with CUDA; IsoNet (https://github.com/IsoNet-cryoET/IsoNet) — GPU deep learning missing wedge correction; dynamo (https://wiki.dynamo.biozentrum.unibas.ch) — subtomogram averaging with GPU; IMOD (https://bio3d.colorado.edu/imod/) — tomogram reconstruction toolkit.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Use a CUDA library (cuFFT) + batched independent jobs** (`docs/PATTERNS.md` §1
and §5; exemplar flagship `8.03`). Each (candidate, angle) is an independent job;
`cufftPlanMany` batches one 3-D R2C/C2R FFT per job, and tiny custom kernels do
the rotation, the per-frequency `conj(ref)·cand` multiply, and a deterministic
shared-memory reduction for the peak and zero-shift scores. The full catalog
pattern (`cuFFT for 3D Fourier-space cross-correlation; custom back-projection
kernels; template matching; PyTorch CUDA for IsoNet`) spans reconstruction and
deep learning too; this teaching version implements the cross-correlation
alignment core.

## Exercises

1. **Break the conjugate.** In `xcorr_mul_kernel`, change `conj(ref)·cand` to
   `ref·cand` (drop the conjugate). Re-run: the recovered peak shifts/mirrors.
   Explain why — this is the difference between correlation and convolution
   (THEORY §2, §5).
2. **Forget the `1/V`.** Remove the `invV` scaling in `reduce_kernel`. The NCC
   blows up by a factor of `V`. This is the cuFFT-unnormalized-inverse gotcha
   (THEORY §5) — feel it, then put it back.
3. **Bigger cubes.** Regenerate with `python scripts/make_synthetic.py --d 32`
   (rebuild, re-capture `expected_output.txt`). Watch the GPU's lead over the CPU
   grow — the FFT advantage scales with `V`.
4. **Finer angle grid.** Bump `--angles 36` (10° steps). Do the recovered poses
   refine? What happens to the peak NCC? (Hint: interpolation blur sets a floor.)
5. **Use the peak shift.** The demo verifies the *zero-shift* score; extend
   `reduce_kernel` to also return the **argmax voxel** (the best translation) and
   apply it when building the average. This is true translational alignment.
6. **Constant-memory reference spectrum.** At large batch sizes, put the
   reference spectrum in `__constant__`/texture memory and measure the change
   (THEORY §4, "Memory hierarchy").

## Limitations & honesty

- **Reduced scope (CLAUDE.md §13).** This is the *teaching* core of STA, not a
  research tool. We search **one in-plane rotation about z**, not the full 3-D
  orientation space (3 Euler angles) that RELION-4/Dynamo search. THEORY §7 maps
  every simplification to the real algorithm.
- **Synthetic data, labeled as such.** The sample is generated by
  `scripts/make_synthetic.py` (Gaussian-blob motif + rotation + noise) with a
  **planted, known answer**. It is not real cryo-ET data and implies no
  biological or clinical result.
- **No missing-wedge model.** Our motif is fully sampled; real subtomograms have
  the characteristic missing wedge that requires masking / constrained
  correlation / IsoNet-style inpainting (THEORY §1, §7).
- **No CTF, no dose weighting, no iteration, no classification.** Production STA
  iterates align→average→re-align with Bayesian weighting and gold-standard FSC;
  we do a single pass.
- **Tiny problem.** 6 cubes at `16³` is sized for a fast, deterministic demo;
  the timing line is a *teaching artifact*, never a benchmark claim (CLAUDE.md §12).
