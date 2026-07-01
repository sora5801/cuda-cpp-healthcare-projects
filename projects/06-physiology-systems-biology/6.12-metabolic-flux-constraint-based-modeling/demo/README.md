# Demo — 6.12 Metabolic Flux / Constraint-Based Modeling

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed synthetic toy model (`data/sample/toy_core_model.txt`).
3. **Verify** the GPU knockout screen against the CPU reference (`reference_cpu.cpp`)
   and print a clear `PASS`/`FAIL`.
4. **Time** the GPU screen (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## What you are looking at

Each line of the screen is a **single-reaction gene knockout**: delete that
reaction (clamp its flux to zero), re-solve the Flux Balance Analysis linear
program, and report the mutant's optimal biomass flux as a percentage of the wild
type. The label classifies the result:

- **ESSENTIAL** — growth collapses to ~0 (the reaction has no alternative route).
  In real biology these are candidate drug targets.
- **reduced** — growth drops but survives (an alternative, lower-capacity route
  exists). Here `B->C` falls to 30% because carbon is forced through the
  capacity-3 bypass.
- **neutral** — no growth change (a redundant isozyme or unused overflow reaction).

The wild-type biomass is **10** (capped by the substrate uptake bound). The
screen finds **3 essential, 1 growth-reducing, 4 neutral** reactions — a result
you can verify by hand from the network in `data/README.md`.

## Expected result

```
6.12 -- Metabolic Flux / Constraint-Based Modeling
model: 4 metabolites x 8 reactions   (SYNTHETIC toy network)
wild-type max biomass flux = 10.0000
single-reaction knockout screen (growth as % of wild type):
  KO uptake_A     biomass=0.0000  (  0.00% WT)  ESSENTIAL
  KO A->B_1       biomass=10.0000  (100.00% WT)  neutral
  KO A->B_2iso    biomass=10.0000  (100.00% WT)  neutral
  KO B->C         biomass=3.0000  ( 30.00% WT)  reduced
  KO A->C_byp     biomass=10.0000  (100.00% WT)  neutral
  KO C->D         biomass=0.0000  (  0.00% WT)  ESSENTIAL
  KO D->biomass   biomass=0.0000  (  0.00% WT)  ESSENTIAL
  KO A->waste     biomass=10.0000  (100.00% WT)  neutral
summary: 3 essential, 1 growth-reducing, 4 neutral reactions
RESULT: PASS (GPU screen matches CPU within tol=1.0e-09)
```

Because the CPU and GPU run the **identical** deterministic simplex (shared
`__host__ __device__` code in `src/fba.h`), the two objective arrays agree
bit-for-bit — the reported `worst |CPU-GPU| objective diff` on stderr is `0`.
