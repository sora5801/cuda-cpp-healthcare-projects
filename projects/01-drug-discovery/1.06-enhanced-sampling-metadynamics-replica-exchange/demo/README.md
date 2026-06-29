# Demo — 1.6 Enhanced Sampling — Metadynamics & Replica Exchange

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/metad_config.txt` (a synthetic 1-D
   double-well metadynamics run).
3. **Verify** two things and print a clear `PASS`/`FAIL`:
   - the CPU reference and the GPU kernel recover the **same** free-energy surface
     (within 0.25 kT over the well-sampled core), and
   - the recovered **barrier height** matches the *known analytic* value (within
     0.35 kT) — the science check.
4. **Report** the recovered FES next to the true landscape, plus timing and
   chaotic per-walker diagnostics on stderr.

The program splits its output deliberately (PATTERNS.md §3):

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt). It prints only the **robust
  statistical observable** — the recovered FES (rounded to 0.1 kT) and the verdict.
- **stderr** carries timing and the *chaotic* per-walker numbers (crossing counts,
  individual final positions), which are not reproducible across platforms; it is
  shown but never diffed.

## Expected result

```
1.6 -- Enhanced Sampling -- Metadynamics & Replica Exchange
well-tempered metadynamics on a 1-D double well (SYNTHETIC model)
ensemble: 64 walkers x 20000 steps; barrier A=5.00 kT, gamma=10.0, pace=50, sigma=0.10
grid: 121 bins over s in [-2.00, 2.00]; hills/walker=400
recovered FES F(s) [kT] at s = -1.0 -0.5  0.0 +0.5 +1.0:
  est :  0.0  3.0  5.0  3.0  0.4
  true:  0.0  2.8  5.0  2.8  0.0
barrier height: recovered 5.0 kT vs true 5.0 kT
RESULT: PASS (CPU & GPU recover the same FES within 0.25 kT; barrier matches analytic within 0.35 kT)
```

## How to read it

- **`est` vs `true`** — the metadynamics-recovered free energy next to the *known*
  double well. They agree at the minima (`s=±1`, F≈0), at the half-way points
  (`s=±0.5`, F≈3 kT), and at the **barrier** (`s=0`, F=5 kT). That the recovered
  curve has a barrier *at all* means the walkers crossed it — plain MD would have
  stayed in one well and recovered nothing past it.
- **`barrier height`** — the single most important number: metadynamics recovered
  the **5.0 kT** barrier it was never told about. This is the "did it work?" check.
- **stderr `ensemble barrier crossings`** — hundreds-to-thousands of crossings
  across the ensemble (plain MD ≈ 0). It also shows the CPU and GPU counts
  *differ* slightly: the trajectories are chaotic, so individual paths diverge
  across platforms even though the *FES* they produce agrees. That is expected and
  explained in [THEORY.md](../THEORY.md) "Numerical considerations".

A small asymmetry (`est` ≈ 0.4 kT at `s=+1.0` instead of 0.0) is an honest
finite-sampling artifact: the right well happened to be slightly less visited in
this short demo run. Increase `--steps` or `--n-walkers` (via
`scripts/make_synthetic.py`) and it shrinks.
