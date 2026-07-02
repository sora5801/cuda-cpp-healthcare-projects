# Demo — 6.25 Liver & Kidney Perfusion Modeling

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed synthetic lobule (`data/sample/lobule.txt`).
3. **Verify** two things:
   - the GPU per-sinusoid results match the CPU reference (`reference_cpu.cpp`) to
     round-off (they run the *same* RK4 from `perfusion.h`), and
   - the mean extraction ratio matches the **analytic first-order limit** — a
     check on the physics, not just CPU==GPU agreement.
   It then prints a clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Reading the result

Each row is one **sinusoid** (a capillary in the lobule) at a different inlet
blood velocity `v`. Slower blood spends more residence time next to the
metabolizing hepatocytes, so it is cleared more — note the extraction ratio
**falls** monotonically as `v` rises (21.8% at 0.2 mm/s down to 4.8% at 1.0 mm/s).
The `lobule extraction ratio: mean` is the whole-organ-unit clearance proxy.

## Expected result

```
6.25 -- Liver & Kidney Perfusion Modeling
SYNTHETIC liver lobule: 4096 parallel sinusoids, L=0.500 mm, C_in=1.000 uM, Km=50.000 uM
zonation Vmax: periportal=8.000 -> centrilobular=2.000 uM/s; velocity sweep 0.2000..1.0000 mm/s over 200 RK4 steps
sample sinusoids (v[mm/s] -> C_out[uM] extraction[%]):
  s0    :   0.2000 ->   0.7822  21.780
  s1024 :   0.4000 ->   0.8846  11.545
  s2048 :   0.6001 ->   0.9215   7.850
  s3072 :   0.8001 ->   0.9405   5.946
  s4095 :   1.0000 ->   0.9521   4.786
lobule extraction ratio: mean=0.0930  min=0.0479  max=0.2178
analytic first-order limit: mean extraction=0.0946
RESULT: PASS (GPU==CPU within 1e-09; mean extraction within 1e-02 of analytic)
```

Timing (on `stderr`) varies per machine; on the reference RTX 2080 the GPU kernel
runs in ~1 ms while the serial CPU loop over 4096 sinusoids takes ~35 ms. This is
a launch-bound toy size — the GPU's advantage grows toward the millions of
segments a real organ model needs.
