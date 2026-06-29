# Demo — 10.02 Real-Time Soft-Tissue Deformation (PBD)

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/cloth_params.txt` (a 24×24 pinned sheet, 300 steps).
3. **Verify** that the GPU PBD solver and the CPU reference reach the same final
   mesh (within a physically-negligible tolerance).
4. **Report** sampled particle positions and the **drape depth** (how far the
   sheet sagged under gravity).

stdout (final positions) is deterministic and diffed against
[`expected_output.txt`](expected_output.txt); the timing line is on stderr only.

## Canonical output

See [`expected_output.txt`](expected_output.txt). The pinned top edge holds while
the rest drapes ~12.6 units under gravity, symmetric about the centre. `RESULT:
PASS` means the GPU and CPU meshes agree to within `1e-3` on positions of
magnitude ~10.

> **Numerical note:** over thousands of constraint iterations the GPU and CPU
> drift at the ~`1e-5` level because their floating-point fused-multiply-add
> behaviour differs — a real lesson in GPU reproducibility (see THEORY). The
> agreement to ~6 significant figures confirms correctness.

> The mesh is a **synthetic grid sheet**, not a patient organ — a demonstration of
> the PBD/GPU pattern, not a validated biomechanical model.
