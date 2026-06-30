# Demo — 2.14 Protein-Ligand Co-Folding

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed synthetic complex (`data/sample/complex_sample.txt`).
3. **Co-fold**: run a reverse-diffusion loop whose every step is a self-attention
   pass over the joint protein+ligand token sequence, on both the **CPU
   reference** and the **GPU**, starting from the same noised positions.
4. **Verify** two things and print a clear `PASS`/`FAIL`:
   - the GPU final positions match the CPU's within `1e-3` (numerical agreement),
   - the recovered **RMSD-to-native** drops below `0.5` Angstrom (the science
     check: the noise cloud actually folded back into the planted complex).
5. **Time** the denoising loop (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately (PATTERNS.md §3):

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
2.14 -- Protein-Ligand Co-Folding
[reduced-scope teaching model: analytic-score diffusion, not a learned network]
complex: 17 tokens (12 protein + 5 ligand), 160 denoising steps
schedule: temp=0.500 step_frac=0.100 type_bias=1.000 noise_scale=1.000
RMSD to native: start=1.5503  ->  final=0.0120 (Angstrom)
ligand pose (final x y z):
  atom 0: -3.6116 0.4861 0.9701
  atom 1: -1.8000 -0.4999 1.8000
  atom 2: 0.0000 0.4999 2.6000
  atom 3: 1.8000 -0.4999 3.4000
  atom 4: 3.5999 0.4999 4.2000
RESULT: PASS (GPU==CPU within 1e-03; pose folded RMSD<0.5)
```

### How to read it

- **RMSD start -> final**: the noised cloud begins ~1.55 A from the native
  complex; after 160 denoising steps the reverse diffusion drives it to ~0.012 A
  — i.e. it recovered the planted bound pose almost exactly.
- **ligand pose**: the final predicted coordinates of the 5 ligand atoms. Compare
  them to the natives in `data/sample/complex_sample.txt` (the rows with type `1`,
  at x = -3.6, -1.8, 0.0, 1.8, 3.6) — each atom lands on its own native position,
  not the chain centroid. That separation is the whole point: geometric attention
  resolved every atom individually.
- **stderr timing** (shown, not diffed): at this toy token count the per-step
  attention is launch-bound, so the GPU loop can be *slower* than the CPU — the
  honest-timing rule (PATTERNS.md §7). The GPU's advantage grows with sequence
  length, where the O(N²) attention dominates.

> This is the **reduced-scope teaching** model: the "score network" is a fixed
> analytic function of geometry (no learned weights), so the math is fully
> transparent and CPU/GPU agree to ~machine precision. A real co-folding model
> (Boltz-1, AlphaFold3) replaces that analytic score with a trained transformer —
> see [`../THEORY.md`](../THEORY.md) "Where this sits in the real world".
