# Demo — 6.2 Whole-Heart Digital Twin

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/heart_ensemble.txt`.
3. **Verify** the GPU ensemble against the CPU reference (`reference_cpu.cpp`)
   and print a clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program computes, for each of the 12 virtual hearts in a **contractility
sweep**, a full closed-loop cardiac cycle (FitzHugh-Nagumo electrophysiology →
time-varying-elastance contraction → 3-element Windkessel circulation) and
reports the **pressure–volume summary**: end-diastolic/end-systolic volume,
**stroke volume (SV)**, **ejection fraction (EF)**, and peak ventricular/arterial
pressures. It then performs the **twin-fit** step: scanning the ensemble for the
contractility whose stroke volume best matches a synthetic clinical target.

The output is split deliberately:

- **stdout** is byte-for-byte deterministic (all double-precision RK4 over a
  fixed number of steps) and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries timing and the numeric verification error (which vary run to
  run), so it is shown but never diffed.

## Expected result

```
6.2 -- Whole-Heart Digital Twin (reduced-scope teaching model)
closed-loop 0-D twin: FitzHugh-Nagumo EP + elastance mechanics + 3-element Windkessel
ensemble: 12 virtual hearts, contractility E_max 1.20..3.40 mmHg/mL, 6 beats @ dt=0.10 ms
SYNTHETIC parameters -- not a real patient. Not for clinical use.
member  Emax(mmHg/mL)  EDV(mL)  ESV(mL)   SV(mL)   EF(%)  Ppk_lv  Ppk_ao
  m0           1.200   143.26    87.97   55.287   38.6  150.22   87.23
  ...
  m11          3.400   143.19    52.79   90.399   63.1  355.35  141.70
twin-fit: target SV = 70.000 mL -> best member m3 (Emax=1.800 mmHg/mL, SV=69.264 mL, EF=48.4%)
RESULT: PASS (GPU ensemble matches CPU within tol=1.0e-09)
```

**How to read it.** As contractility `E_max` rises across the sweep, the
ventricle empties more (ESV falls), so **stroke volume and ejection fraction
increase** and **peak pressures rise** — exactly the Frank-Starling / elastance
behaviour a cardiologist expects. The twin-fit line picks member `m3`
(`E_max = 1.8 mmHg/mL`, `SV = 69.3 mL`) as the closest match to the synthetic
target `SV = 70 mL`, illustrating one step of parameter estimation.

`RESULT: PASS` means the GPU (one thread per virtual heart) reproduced the serial
CPU reference to within `1e-9` — far below any physiological significance,
because both run the **same shared double-precision physics** (`src/heart.h`).

> The numbers come from a simplified 0-D model on **synthetic** parameters. They
> are a teaching artifact, **not** a clinical measurement.
