# Demo — 5.10 Secondary Cancer Risk & Stray-Dose Monte Carlo

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed synthetic phantom `data/sample/phantom.txt`.
3. **Verify** the GPU stray-dose tally against the CPU reference
   (`reference_cpu.cpp`) — they must be **bit-identical** (fixed-point integers),
   printed as `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic (per-organ fixed-point dose printed
  exactly; risk derived from those exact integers) and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the wall-clock timing and a safety note (which vary or are
  advisory), so it is shown but never diffed.

## What you are looking at

Each row of the table is an organ. `dose_fixed` is the accumulated stray dose in
fixed-point units (integer, so CPU and GPU agree exactly). `stray/target` is that
organ's dose relative to the in-field **Target** dose — note it is 3–5 orders of
magnitude smaller, exactly the out-of-field falloff the physics predicts.
`LAR(per1e4)` is the illustrative BEIR-VII-style lifetime cancer-risk contribution.
The final line sums the out-of-field organs into a total secondary-cancer risk.

The interesting thing to watch: distant organs still get a floor of dose from the
**leakage** channel (roughly uniform), while nearer organs get extra **scatter**
dose via forced detection — so the stray-dose profile is highest just past the
field edge and flattens toward the far end of the body.

## Expected result

```
5.10 -- Secondary Cancer Risk & Stray-Dose Monte Carlo
phantom: 9 organs, field ends at organ 1, mu=0.070 /cm, organ=8.0 cm
histories = 200000, VR = survival-biasing + forced-detection + roulette
scatter_frac=0.90 sidescatter=0.0008 leakage=6.00e-05 neutron=2.00e-05
organ                 dose_fixed   stray/target   LAR(per1e4)
Target              8598699914013    1.000e+00   0.0000e+00
RedMarrow             50806673490    5.909e-03   6.0968e+01
Colon                 35163352027    4.089e-03   3.5163e+01
Lung                  26088909543    3.034e-03   2.2176e+01
Stomach               20786103550    2.417e-03   1.4550e+01
Bladder               17654168821    2.053e-03   9.7098e+00
Breast                15776648883    1.835e-03   1.2621e+01
Thyroid               14627924295    1.701e-03   5.8512e+00
Skin                  13906253920    1.617e-03   1.3906e+00
total out-of-field secondary-cancer LAR = 1.6243e+02 per 10^4 persons
RESULT: PASS (GPU dose tally matches CPU exactly)
```

The exact numbers live in [`expected_output.txt`](expected_output.txt), captured
from a real run on the committed sample. Note the `stray/target` column: every
out-of-field organ is ~1.6e-3 to 5.9e-3 of the target dose (sub-percent), and it
falls off monotonically with distance from the field edge — the out-of-field
falloff the science predicts. `RESULT: PASS` means the GPU and CPU stray-dose
tallies matched to the last bit.

> All data here is **synthetic** and the risk coefficients are illustrative. This
> demo is educational and must not be used for any clinical decision.
