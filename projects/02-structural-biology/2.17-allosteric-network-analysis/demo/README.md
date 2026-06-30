# Demo — 2.17 Allosteric Network Analysis

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed synthetic trajectory in `data/sample/trajectory.txt`.
3. **Compute** the N×N Dynamical Cross-Correlation (DCC) matrix on **both** the CPU
   reference (`reference_cpu.cpp`) and the GPU (`kernels.cu`), and **verify** they
   are **bit-for-bit identical** (the shared `dcc_core.h` math guarantees it).
4. **Analyze** the resulting residue network: build the contact graph, run
   Floyd–Warshall on the `-log|C|` communication weights, and print the
   **allosteric communication pathway** from the allosteric site to the active
   site plus its bottleneck (weakest-correlation) hop.
5. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately (PATTERNS.md §3):

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
2.17 -- Allosteric Network Analysis
trajectory: 30 residues, 120 frames (synthetic)
DCC matrix: 30x30  contact graph: 29 edges (cutoff 8.0 A)
allosteric site: residue 2   active site: residue 27
direct correlation C[2][27] = 0.8552
communication path (26 residues, cost 5.9042): 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
bottleneck hop: 9-10  |C| = 0.4558
C diagonal sample: 1.0000 1.0000 1.0000 1.0000
RESULT: PASS (GPU DCC matrix matches CPU exactly)
```

## How to read it

- **`contact graph: 29 edges`** — the synthetic chain folds into a simple path
  graph (each residue touches only its sequence neighbors), so a signal must walk
  the chain residue by residue.
- **`communication path … 2 3 4 … 27`** — the optimal allosteric route threads all
  26 residues between the two functional sites. Its **cost** is the sum of the
  `-log|C|` edge weights along the route: lower means stronger end-to-end coupling.
- **`bottleneck hop: 9-10`** — the weakest link on the path, at the boundary of the
  engineered "hinge" domain. This is the synthetic stand-in for a real **allosteric
  hotspot residue** whose mutation would break communication.
- **`RESULT: PASS`** — the GPU and CPU correlation matrices agree to the last bit
  (worst difference `0.0`), so the speed of the GPU costs us nothing in accuracy.

> The data is **synthetic** and **not for any clinical or research conclusion** — it
> exists to make the GPU pattern and the network analysis legible and verifiable.
