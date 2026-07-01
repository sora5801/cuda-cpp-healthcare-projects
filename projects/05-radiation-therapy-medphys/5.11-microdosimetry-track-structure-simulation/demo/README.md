# Demo — 5.11 Microdosimetry & Track-Structure Simulation

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed synthetic sample `data/sample/track_params.txt`.
3. **Verify** the GPU tallies against the serial CPU reference
   (`reference_cpu.cpp`) and print a clear `PASS`/`FAIL`.
4. **Time** the GPU Monte Carlo (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and per-tally match flags (which vary run to run),
  so it is shown but never diffed.

## Reading the result

The demo simulates 4000 charged-particle tracks through a 100 nm water box and
reports:

- **energy imparted** — total energy deposited (integer quanta × quantum size),
  the exactly-tallied quantity that makes GPU == CPU verification trivial.
- **DNA damage** — total single-strand breaks (SSB) and double-strand breaks
  (DSB), plus the **SSB/DSB ratio** (a low ratio ⇒ dense, high-LET tracks that
  make clustered, lethal DSBs) and **DSB per track**.
- **dose-mean lineal energy yD** (keV/µm) — the headline microdosimetric summary,
  dominated by the high-LET tail of the spectrum.
- **lineal-energy spectrum f(y)** — the 12-bin histogram of per-track lineal
  energy; a broad, right-skewed distribution is the microdosimetric fingerprint
  of a mixed field.

The whole point: the **GPU tallies match the CPU exactly** (tolerance = 0),
because every scored quantity is an integer and integer atomics commute.

## Expected result

```
5.11 -- Microdosimetry & Track-Structure Simulation
water box 100.0 nm, sigma_ion=1.000 /nm (LET spread 0.50), dna_radius=3.00 nm, 25 DNA segments
tracks = 4000, quantum = 30.0 eV
energy imparted = 1599001 quanta (47970.030 keV total)
DNA damage: SSB = 1032, DSB = 457  (SSB/DSB = 2.258)
DSB per track = 0.1143
dose-mean lineal energy yD = 225.677 keV/um
lineal-energy spectrum f(y) (counts per bin):
  15 373 808 861 630 465 302 201 136 80 40 89
RESULT: PASS (GPU tallies match CPU exactly)
```

This is captured in [`expected_output.txt`](expected_output.txt) from a real run
on an RTX 2080 (sm_75). The stdout is deterministic, so it is identical on any
CUDA-capable GPU; only the stderr timing differs.
