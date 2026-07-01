# Demo — 5.12 FLASH Radiotherapy GPU Modeling

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/flash_ensemble.txt` input.
3. **Verify** the GPU ensemble against the CPU reference (`reference_cpu.cpp`)
   per member and print a clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## How to read the result

Each row is one oxygen level (pO₂), reporting the **oxygen-fixed damage** under
**conventional** vs **FLASH** delivery of the *same* 10 Gy dose, the minimum O₂
reached during the FLASH burst (the depletion depth), and the **sparing factor**
= `conv_damage / flash_damage`. A sparing factor **> 1** means FLASH did *less*
damage — the modelled normal-tissue-sparing FLASH effect.

The teaching payoff is the **shape** of the sparing column: FLASH spares tissue
at every oxygen level, with the **largest** relative sparing at low oxygenation
(where a small O₂ depletion moves furthest down the steep part of the oxygen-
enhancement-ratio curve), shrinking toward ~1.05× as the tissue becomes well
oxygenated. This entire pattern is *emergent* from the shared ODE — nothing about
"FLASH" is hard-coded; the only difference between the two modes is the timing of
the pulses.

## Expected result

```
5.12 -- FLASH Radiotherapy GPU Modeling
[educational reduced-scope model -- not for clinical use]
ensemble: 8 pO2 levels x 2 delivery modes = 16 members
dose = 10.0 Gy in 10 pulses; conv gap = 0.04000 s, FLASH gap = 0.00001 s
pO2[mmHg]  conv_damage  flash_damage  flash_minO2[uM]  sparing
     2.00     16.05474      13.96166          0.67558   1.1499
     7.43     23.02866      19.76140          2.55334   1.1653
    12.86     25.54617      22.61183          4.49699   1.1298
    18.29     26.74486      24.27952          6.50841   1.1015
    23.71     27.43473      25.35509          8.58945   1.0820
    29.14     27.88303      26.09567         10.74185   1.0685
    34.57     28.19849      26.63278         12.96728   1.0588
    40.00     28.43297      27.04015         15.26729   1.0515
mean FLASH sparing factor = 1.1009 (conv damage / FLASH damage)
RESULT: PASS (GPU ensemble matches CPU within tol=1.0e-09)
```

Because both the CPU and GPU run the *identical* double-precision RK4 integrator
from `src/flash.h`, they agree to ~1e-14 (far below the 1e-9 tolerance). The
`stderr` timing will differ on your machine — that is expected and not diffed.
