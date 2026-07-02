# Demo — 6.16 Cardiac Mechanics & Electromechanical Coupling

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/heart_ensemble.txt`.
3. **Verify** the GPU ensemble against the CPU reference (`reference_cpu.cpp`) and
   print a clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

Each **virtual heart** is a 0-D electromechanics model: an electrical activation
drives a calcium transient, calcium recruits cross-bridges, the recruited state
raises the chamber's *elastance*, and the resulting pressure ejects blood into a
Windkessel arterial load — producing a pressure–volume (PV) loop. The demo sweeps
**contractility × afterload** (`6 × 6 = 36` hearts) and reports each heart's
end-diastolic/end-systolic volume, stroke volume, **ejection fraction (EF)**, and
peak pressure. One GPU thread integrates one heart (batch-ODE pattern).

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## What to look for

- **Contractility raises EF/SV.** As `Tref` climbs from 1.5 → 3.5 mmHg/mL, EF
  rises from ~36% (a weak / failing ventricle) to ~65% (healthy). This is the
  headline coupling result.
- **Afterload raises peak pressure.** Higher `R_sys` makes the ventricle pressurise
  more to eject the same stroke.
- The `best EF` / `worst EF` lines pick out the sweep corners deterministically.

## Expected result

```
6.16 -- Cardiac Mechanics & Electromechanical Coupling
[reduced-scope teaching model: 0-D electromechanics + Windkessel; NOT FOR CLINICAL USE]
ensemble: 36 hearts (6 contractility x 6 afterload), 10 beats @ dt=0.10 ms, 8000 steps/beat
sample hearts (Tref[mmHg/mL] Rsys[mmHg.ms/mL] -> EDV ESV SV[mL] EF% Ppeak[mmHg]):
  h0    :   1.50   0.7000 -> 138.02  88.44  49.58  35.92    76.19
  h9    :   1.90   1.6000 -> 137.50  73.99  63.52  46.19    78.09
  h18   :   2.70   0.7000 -> 136.48  57.27  79.21  58.04    77.33
  h27   :   3.10   1.6000 -> 135.97  52.04  83.93  61.73    79.81
  h35   :   3.50   2.2000 -> 135.47  47.97  87.51  64.59    81.47
best EF : h30    (Tref=3.50 R=0.7000) EF= 64.61% SV= 87.53 mL
worst EF: h5     (Tref=1.50 R=2.2000) EF= 35.83% SV= 49.46 mL
ensemble: mean EF = 53.25%
RESULT: PASS (GPU ensemble matches CPU within tol=1.0e-01)
```

The `RESULT: PASS` line confirms the GPU per-heart PV-loop summaries match the CPU
reference within the documented tolerance (`0.1` mL / mmHg / percentage-point — a
*physical* tolerance for this ~80,000-step-per-heart solver; see THEORY.md
"Numerical considerations" for why bit-identical is not achievable and why this
tolerance is honest). On stderr you will also see a timing line: on this tiny
36-heart ensemble the many long, branch-heavy per-thread integrations are
launch/latency-bound, so the GPU can be *slower* than the CPU — the GPU's edge
appears only as the ensemble grows into the thousands (PATTERNS.md §7).
