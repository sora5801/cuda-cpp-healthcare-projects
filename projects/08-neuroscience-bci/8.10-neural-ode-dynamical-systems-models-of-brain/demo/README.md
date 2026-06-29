# Demo — 8.10 Neural ODE / Dynamical Systems Models of Brain

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/` input.
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`) and
   print a clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
8.10 -- Neural ODE / Dynamical Systems Models of Brain
[template placeholder kernel: SAXPY  out = a*x + y]
n = 8  a = 2
out[0:8] = 0.000000 12.000000 24.000000 36.000000 48.000000 60.000000 72.000000 84.000000
RESULT: PASS (GPU matches CPU within tol=1.0e-05)
```

> **Template note:** this is the SAXPY placeholder (`out = a*x + y`). TODO(impl):
> once the real kernel is in place, update `expected_output.txt` and this file so
> the demo demonstrates *this project's* computation.
