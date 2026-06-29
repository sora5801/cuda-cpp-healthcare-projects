# Demo — 1.24 Umbrella Sampling / WHAM Free Energy Profiles

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/umbrella.txt` (a synthetic double-well landscape and
   a 27-window umbrella scan).
3. **Verify** twice:
   - the GPU per-window histograms equal the CPU reference histograms **exactly**
     (integer counts + identical Langevin physics → bit-for-bit), and
   - the WHAM-reconstructed PMF recovers the **known** double-well to within
     0.30 kT over the interior of the scan.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries timings and the mismatch/PMF-error diagnostics (which can
  vary or are run-detail), so it is shown but never diffed.

## Expected result

See [`expected_output.txt`](expected_output.txt). The headline rows print the
WHAM PMF beside the analytic potential `U(x)` at a sweep of bin centers, so you
can watch the curve dip to ~0 kT in the two wells (near `x = ±1`) and rise to the
~4 kT barrier at `x ≈ 0`:

```
barrier: PMF max 4.003 kT at x=-0.050 (true A=4.000 kT)
histograms: 1620000 total counts; GPU==CPU bins: YES
RESULT: PASS (GPU histograms == CPU exactly; WHAM PMF within 0.30 kT of U)
```

`RESULT: PASS` means both checks held: the GPU and CPU produced identical
histograms, and WHAM recovered the barrier height (4.00 kT) and the well shape.

### Why the GPU is *slower* here (and that is the lesson)

The stderr timing shows the GPU kernel taking longer than the CPU. That is
expected and honest: the kernel launches only **27 threads** (one per window), far
too few to fill a GPU, and each runs a long serial trajectory. This is the
"launch-bound / under-occupied" regime (PATTERNS.md §7). Real umbrella sampling
runs **hundreds of windows**, each a full all-atom MD simulation with thousands of
force evaluations per step — there the per-window work is enormous and the GPU
wins decisively. The exercises in the project README show how to widen the scan
toward that regime.

> The landscape is **synthetic** — a teaching double-well, not a real molecule.
