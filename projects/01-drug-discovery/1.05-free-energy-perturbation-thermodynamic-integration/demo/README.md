# Demo — 1.5 Free Energy Perturbation / Thermodynamic Integration

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/alchemy_sample.txt` input.
3. **Verify** twice: the GPU per-window averages against the CPU reference
   (`reference_cpu.cpp`), *and* the integrated free energy against the closed-form
   answer — printing a clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic (counter-based RNG; fixed precision) and
  is diffed against [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing, the MC acceptance rate, and the numeric errors
  (which vary run to run), so it is shown but never diffed.

## What you are looking at

The synthetic sample morphs harmonic **state A** (`k=1`) into **state B** (`k=4`) at
`kT=1` over 11 λ-windows. The printed **TI curve** `⟨∂U/∂λ⟩` falls monotonically from
`+3.64` (λ=0) toward `−0.12` (λ=1) — the shape of the alchemical force along the path.
Its integral is the free-energy difference:

- `DeltaG_TI = +0.71099` — the Monte-Carlo + trapezoid estimate.
- `DeltaG_analytic = +0.69315` — the exact answer `½·kT·ln(kB/kA) = ½·ln 4 = ln 2`.

The ~0.018 gap is the **trapezoid discretisation bias** of an 11-window grid (the
integrand is steeply curved near λ=0). Run `python scripts/make_synthetic.py
--windows 41` and re-run to watch the estimate converge to `0.693147`.

## Expected result

```
1.5 -- Free Energy Perturbation / Thermodynamic Integration
alchemical TI: stateA(k=1.000) -> stateB(k=4.000) at kT=1.000, 11 windows
MC sampling: 2000 equil + 20000 samples per window, step=0.600
TI curve <dU/dlambda> per window (lambda -> mean):
  w0  lambda=0.00  <dU/dlambda>=  +3.63892
  w1  lambda=0.10  <dU/dlambda>=  +2.17333
  w2  lambda=0.20  <dU/dlambda>=  +1.20911
  w3  lambda=0.30  <dU/dlambda>=  +0.86226
  w4  lambda=0.40  <dU/dlambda>=  +0.56883
  w5  lambda=0.50  <dU/dlambda>=  +0.32128
  w6  lambda=0.60  <dU/dlambda>=  +0.18905
  w7  lambda=0.70  <dU/dlambda>=  +0.10293
  w8  lambda=0.80  <dU/dlambda>=  -0.00265
  w9  lambda=0.90  <dU/dlambda>=  -0.07411
  w10 lambda=1.00  <dU/dlambda>=  -0.11909
DeltaG_TI       = +0.71099  (trapezoid over lambda)
DeltaG_analytic = +0.69315  (= 1/2 kT ln(kB/kA))
RESULT: PASS (GPU==CPU within 1e-09; TI within 5e-02 of analytic)
```

A representative **stderr** (numbers vary per machine/run):

```
[data]   source: data/sample/alchemy_sample.txt  (11 windows)
[timing] CPU: 5.7 ms   GPU kernel: 33.3 ms
[mc]     overall MC acceptance = 81.7% (tune `step` to trade acceptance vs exploration)
[verify] worst |CPU-GPU| per window = 8.216e-14 (tol 1.0e-09)
[verify] |TI - analytic|            = 1.785e-02 (tol 5.0e-02)
```
