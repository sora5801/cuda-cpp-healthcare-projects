# Demo — 4.15 Diffusion MRI & Tractography

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project (Release|x64) if the executable is missing.
2. **Run** it on the committed `data/sample/dwi_sample.txt` — a synthetic
   16×16×4 DWI volume with a curved fiber bundle (see `data/README.md`).
3. **Fit** the diffusion tensor per voxel on both the CPU and the GPU and
   **verify** they agree (fit tolerance 1e-9).
4. **Trace** deterministic streamlines through the fitted direction field on both
   the CPU and the GPU and **verify** they agree (tract tolerance 1e-3).
5. **Report** FA/MD statistics, the highest-FA seed voxels' tensor fits, and the
   streamline lengths; print `PASS`/`FAIL`.
6. **Time** each kernel (CUDA events) vs. the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric verification error (which vary
  run to run), so it is shown but never diffed.

## What you are looking at

- **mean/max FA ≈ 0.80** on the bundle: the fit recovered the strong anisotropy
  of the ground-truth cigar tensors (background voxels have FA ≈ 0, pulling the
  reported mean down slightly).
- The **seed voxels** are the highest-FA tissue voxels; their principal
  eigenvector `v1` points **along the local arc tangent** — e.g. at `(12,12,·)`,
  `v1 = (0.24, 0.97, 0)`, which is the fiber direction the tractography follows.
- The **streamlines** are short polylines that walk the curved bundle; their
  point counts are deterministic for this Release build.

## Canonical output

See [`expected_output.txt`](expected_output.txt) for the exact stdout the demo
asserts. `PASS` means: (a) the GPU tensor fit matched the CPU reference to
< 1e-9, and (b) the GPU streamlines matched the CPU streamlines exactly (both
trace through the one verified fit field — see `THEORY.md` §"Numerical
considerations" for why that separation matters).

> All values reflect the **synthetic** sample (a known ground-truth phantom); they
> carry no clinical meaning.
