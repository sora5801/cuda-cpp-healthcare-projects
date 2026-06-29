# Demo — 1.21 Polarizable / AMOEBA Force Field MD

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/amoeba_ensemble.txt` (8 synthetic
   polarization systems).
3. **Solve** each system's self-consistent **induced dipoles** with a matrix-free
   **conjugate-gradient** solver — on the **CPU** (`reference_cpu.cpp`) and on the
   **GPU** (`kernels.cu`, one thread per system).
4. **Verify** the GPU result against the CPU reference and print a clear
   `PASS`/`FAIL`.
5. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## How to read the output

The per-member table prints, for each system: its index, atom count, the swept
**half-separation** between the two partner atoms, the **CG iteration count**, the
**polarization energy** `Upol = -½ Σ mu_i·E_i^perm`, the net induced dipole along
x, and the **peak induced-dipole magnitude**. The trend is the lesson: as the
atoms approach (later members), the coupling strengthens, so `Upol` becomes more
negative and the dipoles grow.

`RESULT: PASS` means the GPU induced dipoles agree with the CPU reference to
within `1.0e-9` (they run the identical double-precision CG loop, so they match to
machine precision — the actual worst diff, printed on stderr, is ~`1e-17`).

## Expected result

```
1.21 -- Polarizable / AMOEBA Force Field MD
AMOEBA induced-dipole SCF via matrix-free conjugate gradient
ensemble: 8 members, CG tol=1.0e-08, max_iter=64
per-member (idx, atoms, half-sep, CG iters, polarization energy, net dipole_x, peak |mu|):
  m0    n=3 sep=4.000  iters= 2  Upol=-0.004768  mu_x=+0.190717  max|mu|=0.075060
  m1    n=3 sep=3.714  iters= 2  Upol=-0.004840  mu_x=+0.193593  max|mu|=0.076404
  m2    n=3 sep=3.429  iters= 2  Upol=-0.004941  mu_x=+0.197654  max|mu|=0.078292
  m3    n=3 sep=3.143  iters= 2  Upol=-0.005091  mu_x=+0.203630  max|mu|=0.081056
  m4    n=3 sep=2.857  iters= 2  Upol=-0.005323  mu_x=+0.212918  max|mu|=0.085318
  m5    n=3 sep=2.571  iters= 2  Upol=-0.005713  mu_x=+0.228532  max|mu|=0.092415
  m6    n=3 sep=2.286  iters= 2  Upol=-0.006456  mu_x=+0.258228  max|mu|=0.105751
  m7    n=3 sep=2.000  iters= 2  Upol=-0.008233  mu_x=+0.329305  max|mu|=0.137227
summary: strongest Upol=-0.008233  largest |mu|=0.137227  total CG iters=16
RESULT: PASS (GPU ensemble matches CPU within tol=1.0e-09)
```

(Timing lines on **stderr** will differ on your machine — that is expected.)
