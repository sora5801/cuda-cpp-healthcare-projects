# Demo — 5.13 BNCT Dose Calculation & Optimization

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed synthetic input `data/sample/bnct_params.txt`.
3. **Verify** the GPU per-component dose tally against the CPU reference
   (`reference_cpu.cpp`) **exactly** — because energy is deposited as integer
   keV quanta, the GPU's `atomicAdd` order does not change the sum, so the
   tolerance is **zero mismatches** (docs/PATTERNS.md §4).
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the boron-fraction note (which vary run to
  run in wall-time), so it is shown but never diffed.

## What you are looking at

- **component totals** — the four BNCT physical dose components (boron,
  nitrogen, hydrogen-capture gamma, fast-neutron recoil), each with its share of
  the total deposited energy. In this synthetic case boron capture accounts for
  ~35% and the fast-neutron background ~54%.
- **CBE/RBE-weighted biological dose** — the clinically relevant quantity
  `D_bio = Σ_c w_c · D_c` in Gy-Eq, using representative CBE (boron ≈ 3.8) and
  RBE weights. It is larger than the raw physical dose because the high-LET
  components are up-weighted.
- **boron depth-dose** — the headline curve: keV deposited by ¹⁰B(n,α)⁷Li vs.
  depth. Note it **rises to a peak a couple cm deep, then falls** — fast neutrons
  must first slow to thermal energies before boron can capture them, so the
  thermal-neutron flux (and thus the boron dose) builds up below the surface.
  This "thermal-neutron build-up" shape is the real, teachable BNCT signature.

## Expected result (captured from a real run on an RTX 2080, sm_75)

```
5.13 -- BNCT Dose Calculation & Optimization
REDUCED-SCOPE TEACHING MODEL (synthetic 1-D two-group MC; not clinical)
slab L=10.0 cm, 20 depth bins, histories=200000
thermal Sigma_a (1/cm): B=0.0700 N=0.0050 H=0.0210  Sig_s_th=1.500
fast: Sig_s=0.900 /cm  p_thermalize=0.25
component totals (keV quanta and % of physical dose):
  boron  (10B(n,a)7Li) :    173236140  ( 35.0%)
  nitro  (14N(n,p)14C) :      3398554  (  0.7%)
  gamma  (1H(n,g)2H)   :     49764224  ( 10.1%)
  fast   (recoil p)    :    267952000  ( 54.2%)
physical dose total = 0.0005 Gy (scale 1.000e-12 Gy/keV)
CBE/RBE-weighted biological dose = 0.0016 Gy-Eq
boron depth-dose (keV per bin):
  6290130 9240000 10937850 11873400 12469380 12654180 12358500 12132120 11679360 11106480 10030020 9293130 8392230 7849380 6724410 5934390 5112030 4026330 3042270 2090550
RESULT: PASS (GPU per-component dose tally matches CPU exactly)
```

The `keV`-per-bin integers are **byte-identical** every run and on any CUDA GPU
(the RNG and transport are shared host/device integer math). Only the stderr
timing changes.
