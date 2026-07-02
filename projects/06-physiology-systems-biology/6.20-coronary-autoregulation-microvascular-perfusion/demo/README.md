# Demo — 6.20 Coronary Autoregulation & Microvascular Perfusion

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/coronary_network.txt` — a tiny synthetic coronary network.
3. **Verify** the GPU pressures against the CPU reference (`reference_cpu.cpp`) and print `PASS`/`FAIL`.
4. **Time** the solves (CUDA events + CPU wall clock) — a *teaching artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing, CG iteration counts, and the CPU/GPU numeric error (which vary run to run),
  so it is shown but never diffed.

## Expected result (stdout)

```
6.20 -- Coronary Autoregulation & Microvascular Perfusion
network: 8 nodes, 8 segments, hematocrit=0.45, aortic P=100.0 mmHg
autoregulation steps: 8   CG tol: 1.0e-10
nodal pressures (mmHg):
 P[0]=100.0000 P[1]=57.7066 P[2]=41.1765 P[3]=42.5168 P[4]=29.4118 P[5]=27.5056 P[6]=20.0000 P[7]=20.0000
inlet perfusion (pre-autoreg)  = 8.4317e+10 (um^3/s)
inlet perfusion (regulated)    = 613442857.9655 (um^3/s)
stenosis segment       = 2 (nodes 1->3)
virtual FFR (Pd/Pa)    = 0.2815  [flow-limiting (<0.80)]
RESULT: PASS (GPU pressures match CPU within tol=1.0e-06 mmHg)
```

## How to read it

- **nodal pressures** fall from the 100 mmHg aortic inlet (P[0]) to the 20 mmHg venous outlets (P[6], P[7]),
  as physical flow requires.
- **pre-autoreg vs. regulated perfusion** shows autoregulation at work: the un-regulated network is grossly
  over-perfused (~8.4e10), and the feedback loop constricts the arterioles to drive perfusion down toward the
  metabolic set-point (~6.1e8). Because conductance ∝ r⁴, small radius changes produce this large swing.
- **virtual FFR** across the modeled stenosis (segment 2) is well below the 0.80 clinical cut-point, so it is
  flagged **flow-limiting** — the read-out this model class targets. (Synthetic; not a clinical result.)
- **RESULT: PASS** means the independent CPU and GPU solves agree within `1e-6 mmHg` (they actually agree to
  ~`5e-14`, reported on stderr).

A typical **stderr** block (values vary; shown, not diffed):

```
[solve]  cold-start CG iters -- CPU: 5   GPU: 5   (later autoregulation solves warm-start -> ~0 iters)
[solve]  final CG residual -- CPU: 1.18e-06   GPU: 1.53e-06
[timing] CPU: 0.006 ms   GPU(all solves): 4.884 ms
[verify] max |P_cpu - P_gpu| = 4.974e-14 mmHg  (tolerance 1.0e-06)
```

The GPU is *slower* than the CPU here because this 8-node network is launch-bound; the GPU's advantage
appears at 10⁴–10⁶ segments (see [`../THEORY.md`](../THEORY.md) §GPU mapping).
