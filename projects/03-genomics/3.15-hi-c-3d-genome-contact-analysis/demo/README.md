# Demo — 3.15 Hi-C / 3D Genome Contact Analysis

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/hic_sample.txt` (a 12-bin synthetic
   Hi-C matrix with three known TADs and a known per-bin coverage bias).
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`): the
   per-bin ICE **bias** vectors must agree (here, **exactly** — the fixed-point
   atomic reduction makes GPU and CPU sum identical integers). Prints `PASS`/`FAIL`.
4. **Find TADs**: compute the **insulation score** along the diagonal of the
   balanced matrix and call **TAD boundaries** as its local minima.
5. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the bias error (which vary run to run), so it
  is shown but never diffed.

## What to look for in the result

- The **bias** per bin is the visibility correction ICE recovered; multiply two
  raw counts' bins' biases to get the balancing denominator.
- The **insulation score** is high inside a TAD and **dips sharply at a border**.
  In the sample it bottoms out at **bin 4** and **bin 8** — the two domain borders
  baked into the synthetic data.
- **`TAD boundaries ... 2 found: bin 4, bin 8`** is the headline biological
  result: the pipeline rediscovered the domain structure we planted. That is the
  end-to-end check that the GPU balancing + downstream analysis are correct.
- **`RESULT: PASS (GPU bias matches CPU reference within 1e-09)`** is the
  correctness gate.

## Expected result (captured from a real run)

```
3.15 -- Hi-C / 3D Genome Contact Analysis
matrix: 12 bins, 78 stored contacts (upper triangle)
ICE: 30 iterations, balanced bias per bin (occupied bins):
  bin  0: bias = 0.800760
  ...
  bin 11: bias = 0.953422
insulation score (window=3):
  bin  3: 18.363248
  bin  4: 6.492788          <- TAD border (domain 0|1)
  ...
  bin  8: 6.237471          <- TAD border (domain 1|2)
  ...
TAD boundaries (local minima, radius=1): 2 found
  boundary at bin 4
  boundary at bin 8
RESULT: PASS (GPU bias matches CPU reference within 1e-09)
```

The full, exact stdout is in [`expected_output.txt`](expected_output.txt); the
demo passes when the program reproduces it line-for-line.
