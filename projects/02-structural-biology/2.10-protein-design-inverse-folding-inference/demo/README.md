# Demo — 2.10 Protein Design / Inverse Folding Inference

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/backbone_sample.txt` (a synthetic
   60-residue backbone).
3. **Verify** the GPU design against the CPU reference (`reference_cpu.cpp`) and
   print a clear `PASS`/`FAIL`. Because every quantity is an integer computed by
   the *same* shared scoring core, agreement is **exact** (no tolerance).
4. **Time** the two kernels (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the verify flag (which vary run to run), so it
  is shown but never diffed.

## How to read the result

```
residues L = 60   buried (>=16 contacts) = 28   exposed = 32
native   : TFFFFFFFFFFFFFNDFFFFHFFFFFFFWPPGSSKGPPNRTRDDDRRRERRRRREEEEEG
designed : FFFFFFFFFFFFFFFFFFFFFFFFFFFFNPPGSSSGPPNRRRDDDRRRERRRRREEEEEE
native sequence recovery: 87%
```

- **`buried` vs `exposed`** — how many residues sit in the packed core (≥ 16 Cα
  neighbours within 10 Å) versus the surface. This is the geometric signal the
  design uses.
- **`native`** — the ground-truth amino-acid sequence stored with the backbone.
- **`designed`** — the sequence the GPU designed for this backbone. Notice the
  **buried core is filled with hydrophobic Phe (`F`)** and the **surface with
  charged Arg/Asp/Glu (`R`/`D`/`E`)** — the hydrophobic-core rule made visible.
  The middle of the chain (`PPGSSG…`) shows the *graded* choices at intermediate
  burial.
- **`native sequence recovery`** — the percentage of positions where the design
  matches the native. This is the headline metric real inverse-folding tools
  report (ProteinMPNN ~50% on real proteins). Here it is 87% **by construction**
  of the synthetic sample (see `data/README.md`) — it teaches the *metric*, not a
  design capability.

## Expected result

The full deterministic stdout is in [`expected_output.txt`](expected_output.txt),
captured from a real `Release|x64` run on an NVIDIA RTX 2080 (`sm_75`), CUDA 13.3.
The `RESULT: PASS` line confirms the GPU design matched the CPU reference exactly.
