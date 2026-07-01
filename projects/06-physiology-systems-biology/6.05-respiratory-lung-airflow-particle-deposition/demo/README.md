# Demo — 6.5 Respiratory / Lung Airflow & Particle Deposition

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/lung_params.txt` input — a synthetic
   5-µm aerosol inhaled at 30 L/min through 16 idealized airway generations,
   tracking 200 000 Monte-Carlo particle histories.
3. **Verify** the GPU per-generation deposition tally against the CPU reference
   (`reference_cpu.cpp`) and print a clear `PASS`/`FAIL`. Because the tally is
   **integer** and both sides run the **identical** histories (shared
   `lung_physics.h`), the two must agree **exactly** (0 mismatches).
4. **Time** the GPU kernel (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing (which varies run to run), so it is shown but
  never diffed.

## Expected result

```
6.5 -- Respiratory / Lung Airflow & Particle Deposition
aerosol: d_p = 5.0 um, rho_p = 1000 kg/m^3
airway : 16 generations, flow = 30.0 L/min
particles = 200000, seed = 12345
deposited = 43726 of 200000 (21.9%), exhaled = 156274
peak deposition generation = 0
deposition per generation (counts):
  7353 5910 4739 3922 3410 2725 2376 2048 1817 1646 1509 1395 1288 1232 1193 1163
RESULT: PASS (GPU deposition tally matches CPU exactly)
```

## How to read it

- **`deposited = 43726 of 200000 (21.9%)`** — about a fifth of the 5-µm particles
  stick to an airway wall before reaching the alveoli; the rest are exhaled. A
  21.9 % total deposition fraction for a 5-µm particle at resting flow is in the
  right ballpark for the conducting airways.
- **`peak deposition generation = 0`** — deposition is highest in the **trachea**
  (generation 0). At 5 µm the dominant mechanism is **inertial impaction**, which
  scales with the Stokes number ∝ velocity; the trachea has the highest air
  velocity, so it catches the most particles. Deposition then falls monotonically
  with depth as the airways branch and the air slows.
- Change the particle diameter to see the physics flip: sub-micron particles
  (e.g. `--d_p 0.01`) are dominated by **Brownian diffusion** and deposit deeper.
