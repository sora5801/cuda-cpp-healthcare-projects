# Demo — 2.19 Membrane Protein Simulation

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/membrane_sample.txt` input — a tiny
   synthetic coarse-grained membrane patch (18 lipids + a 5-bead protein column).
3. **Verify** the GPU trajectory against the CPU reference (`reference_cpu.cpp`)
   bead-for-bead and print a clear `PASS`/`FAIL`.
4. **Time** the GPU MD loop (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries timing and the GPU-vs-CPU numeric error (which vary run to
  run), so it is shown but never diffed.

## What you are looking at

The report shows the membrane **before** and **after** a short equilibration:

- `bilayer_thickness` — the head-to-head separation of the two leaflets. It
  starts at the build value and settles as the membrane relaxes; staying near
  its initial value means the **bilayer is intact** (it did not fall apart).
- `potential_energy` — total LJ + bond energy. It **drops** (here from ≈ +86 to
  ≈ −41) as beads slide out of their initial mild overlaps into their attractive
  wells: the membrane finding a lower-energy, more physical configuration.
- Three sampled bead positions (a head, a tail, a protein bead) — fixed indices
  so the printed lines are reproducible.

## Expected result

```
2.19 -- Membrane Protein Simulation (coarse-grained, reduced scope)
system: 18 lipids (3 beads each) + 5 protein beads = 59 beads, 200 steps
box: 6.00 x 6.00 (x,y periodic; z free slab)   dt=0.0050  kT=0.600  gamma=1.000
initial : bilayer_thickness = 5.000000   potential_energy = 85.965754
final   : bilayer_thickness = 5.378494   potential_energy = -41.339806
head    bead[0] pos = (-0.121311, -0.436603, 2.604485)
tail    bead[2] pos = (-0.876389, -0.478177, 0.819187)
protein bead[54] pos = (2.362616, 2.504353, -1.892562)
RESULT: PASS (GPU matches CPU within tol=1.0e-04)
```

The `stderr` lines (shown by the demo, not diffed) report the per-run timings and
the actual GPU-vs-CPU agreement — typically `~1e-14` here (round-off), far inside
the `1e-4` tolerance, because the CPU and GPU run the **same** double-precision
math in the **same** order (see `THEORY.md` "Numerical considerations").
