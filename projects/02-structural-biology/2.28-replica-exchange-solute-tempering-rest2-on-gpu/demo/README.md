# Demo — 2.28 Replica Exchange Solute Tempering (REST2) on GPU

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/rest2_config.txt` input.
3. **Run the whole REST2 simulation twice** — GPU sampler vs CPU reference,
   sharing the same exchange step — and **verify** they agree on robust aggregate
   observables (right-well occupancy, acceptance ratio), printing `PASS`/`FAIL`.
4. **Time** the kernels (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt). It reports the **CPU reference**
  results (the portable, deterministic baseline).
- **stderr** carries the timing and the verification diff (which vary run to run),
  so it is shown but never diffed.

## What to look at

- **`lambda ladder (cold->hot)`** — the REST2 scaling factors. r0 is λ = 1 (the
  physical 300 K replica); they shrink toward the hot end.
- **`per-replica ... beadsRight`** — how many of each replica's 8 beads ended in
  the right well.
- **The headline line** — `cold replica r0: 8/8 beads in the global (right) well
  [start was 0/8, trapped left]`. This is REST2 doing its job: r0 started trapped
  in the wrong (left) basin and, via configuration swaps from hotter replicas,
  reached the global minimum. A plain 300 K run would have stayed at 0/8.
- **`RESULT: PASS`** — the GPU and CPU agree on the robust observables.

## Expected result

```
2.28 -- Replica Exchange Solute Tempering (REST2) on GPU
REST2: 8 replicas, 60 rounds x 200 sweeps, 8 solute beads; barrier h=5.00 tilt=2.00
lambda ladder (cold->hot): 1.0000 0.8548 0.7306 0.6245 0.5338 0.4562 0.3900 0.3333 
per-replica (idx lambda  accept%  beadsRight):
  r0   1.0000   58.90  8/8
  r1   0.8548   61.75  8/8
  r2   0.7306   64.56  8/8
  r3   0.6245   67.24  8/8
  r4   0.5338   70.44  8/8
  r5   0.4562   73.55  8/8
  r6   0.3900   76.91  7/8
  r7   0.3333   79.56  8/8
cold replica r0: 8/8 beads in the global (right) well [start was 0/8, trapped left]
total right-well beads across ladder: 63/64
exchanges accepted over the run: 152
RESULT: PASS (GPU matches CPU on robust observables: well occupancy +/-16 beads, acceptance +/-1%)
```

The stderr (timing/verify) lines vary run to run; a typical run shows
`right-well beads CPU=63 GPU=63 (diff 0 ...)` and `exchanges CPU=152 GPU=152`,
confirming the CPU and GPU REST2 simulations land on the same statistics.
