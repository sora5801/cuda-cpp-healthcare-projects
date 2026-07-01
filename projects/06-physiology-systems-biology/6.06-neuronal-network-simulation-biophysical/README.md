# 6.6 — Neuronal Network Simulation (Biophysical)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.6`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

This project simulates a small **network of biophysically detailed neurons** on
the GPU. Each neuron is not a single point but a **multi-compartment cable** — a
soma plus a chain of dendritic segments — and every compartment carries a full
**Hodgkin–Huxley** (HH) membrane with sodium, potassium, and leak currents.
Current flows between neighbouring compartments (the *cable equation*), and
neurons excite one another through **exponential chemical synapses**. We wire the
cells into a ring, kick one of them, and watch a **travelling wave of spikes**
sweep around the loop. The GPU runs one neuron per thread; the result is verified
bit-for-bit against a plain-C++ reference.

## What this computes & why the GPU helps

Simulates networks of morphologically detailed (multi-compartment) neurons using
Hodgkin–Huxley-style conductance-based kinetics in each segment. A single layer-5
pyramidal cell may have 1,000+ compartments each with 10–30 gating variables, and
a cortical column contains thousands of such cells — millions of coupled ODEs.
The **Hines solver** (a tridiagonal Thomas algorithm along each dendritic branch)
integrates one cell efficiently, but *parallelising across cells and synapses* is
where the GPU shines.

**The parallel bottleneck:** every cell must be advanced through the *same* long
time loop (here 4,000 steps), and the cells are independent within a step except
for one-step-delayed synaptic coupling. That makes the work **embarrassingly
parallel over neurons**: we give each neuron its own GPU thread that runs its full
per-step update (Rush–Larsen gates → Hines/Thomas voltage solve → spike test).
With `N` cells the CPU does `N` cable solves serially; the GPU does them at once.
On a tiny 16-cell demo the GPU is *slower* (launch-bound — see Limitations), but
the mapping scales to the thousands of cells real cortical-column models need.

## The algorithm in brief

- **Hodgkin–Huxley conductance kinetics** — Na (m³h), K (n⁴), leak per compartment.
- **Rush–Larsen exponential integration** for the m, h, n gates (unconditionally
  stable; gates stay in [0,1] for any `dt`).
- **Hines tridiagonal solver** for the coupled cable equation — on an unbranched
  cable this is exactly the **Thomas algorithm** (one forward + one back sweep,
  O(compartments), no pivoting needed because the system is diagonally dominant).
- **Backward-Euler** treatment of the diffusion + synapse terms (implicit,
  stable); explicit evaluation of the HH ionic term.
- **Event-driven exponential synapses** (AMPA-like): a presynaptic spike adds a
  conductance quantum that decays with time constant `tauSyn`.
- **Step-synchronous double-buffered spike delivery** (one-step synaptic delay).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/neuronal-network-simulation-biophysical.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/neuronal-network-simulation-biophysical.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\neuronal-network-simulation-biophysical.sln /p:Configuration=Release /p:Platform=x64
```

No extra CUDA libraries are linked — only the CUDA runtime (`cudart`). The Hines
solver and HH kinetics are hand-rolled on purpose so nothing is a black box.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/network.txt`, prints the network
summary, shows the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/network.txt` — a tiny, **synthetic** ring
  configuration so the demo runs with zero downloads.
- **Full/real data:** `scripts/download_data.ps1` / `.sh` print pointers to real
  neuroscience resources (they never bypass any registration).
- **Provenance & license:** see [data/README.md](data/README.md).

Real-world sources (optional, not committed): NeuroMorpho.Org (3D reconstructions,
https://neuromorpho.org), ModelDB (https://modeldb.science), Allen Brain Cell
Atlas (https://portal.brain-map.org), DANDI Archive (https://dandiarchive.org).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The key
lines: each of the 8 shown cells has a **first-spike step that increases by a
constant hop** (the wave), all cells are active, and:

```
RESULT: PASS (GPU spike counts match CPU exactly across 16 cells)
```

The program computes the network on both the **GPU** (`src/kernels.cu`) and a
**CPU reference** (`src/reference_cpu.cpp`). Because they call the identical
double-precision physics in `src/neuron.h`, the soma voltages — and hence the
threshold-crossing spike counts — are **bit-identical**, so the verification
tolerance is exactly **zero**.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
2. [`src/neuron.h`](src/neuron.h) — the shared `__host__ __device__` physics: HH
   kinetics, Rush–Larsen gates, the Hines/Thomas solver, the synapse, `step_neuron`.
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`.cpp`](src/reference_cpu.cpp) —
   the ring config, wiring, and the serial baseline (with the spike-buffer idea).
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-cell
   / per-step-launch idea.
5. [`src/kernels.cu`](src/kernels.cu) — the seed + step kernels and host driver.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **NEURON + CoreNEURON GPU** (https://github.com/neuronsimulator/nrn) — the
  canonical compartmental simulator; CoreNEURON is its CUDA backend. Study how it
  batches Hines matrices and vectorises the gate ODEs.
- **NetPyNE** (https://github.com/suny-downstate-medical-center/netpyne) — a
  high-level network builder on top of NEURON; learn how connectivity/populations
  are specified declaratively.
- **MOOSE** (https://github.com/BhallaLab/moose-core) — multiscale simulator
  coupling electrical + biochemical signalling.
- **Blue Brain / Open Brain Institute** (https://github.com/BlueBrain) —
  production cortical-column models; a look at what "detailed" really means.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**One thread per neuron, one kernel launch per timestep, with ping-pong spike
buffers.** Each thread integrates its own neuron's small fixed-size state in
registers/local memory; the per-step kernel boundary is the grid-wide barrier
that makes the (one-step-delayed) synaptic coupling well-defined and
deterministic. In production, CoreNEURON instead uses **one thread-block per cell**
with a warp walking the dendritic tree, cuSPARSE for batched Hines matrices,
cuRAND for stochastic release, and a struct-of-arrays layout for coalesced gate
access — see THEORY §"Where this sits in the real world".

## Exercises

1. **Grow the network.** `python scripts/make_synthetic.py --ncell 512 --steps 8000`,
   rebuild, and watch the GPU/CPU timing gap close and then reverse as the launch
   overhead is amortised over more cells.
2. **Break the ring into a chain** (skip the wrap-around synapse) and confirm the
   wave stops at the last cell instead of circulating.
3. **Add inhibition.** Introduce a second synapse type with a negative reversal
   potential (`eSyn ≈ -75 mV`, GABA-like) on every other connection and observe
   how it gates the wave.
4. **Fuse the time loop into one kernel** using a cooperative-groups grid barrier
   (`cudaLaunchCooperativeKernel`) to remove the per-step launch overhead — then
   re-time.
5. **Drive it from a real morphology.** Parse an SWC file from NeuroMorpho.Org,
   collapse each branch into compartments, order them for the Hines solver, and
   replace the uniform cable with the reconstructed tree.

## Limitations & honesty

- **Reduced-scope teaching version.** Real biophysical simulators (NEURON) use
  *branching* trees (this demo uses an unbranched cable so the Hines solver is the
  clean Thomas algorithm), variable compartment geometry, many ion channel types,
  and adaptive time-stepping (CVODE). We use fixed `dt`, a single AMPA-like
  synapse, and ≤ 8 uniform compartments per cell. THEORY explains how each
  simplification would be lifted.
- **Synthetic data.** The committed network is generated, not measured; it does
  not represent any real brain region, species, or patient.
- **Timing is a teaching artifact, not a benchmark.** With 4,000 tiny per-step
  launches on 16 cells the run is *launch-bound*, so the GPU is slower than the
  CPU here. That is expected and is called out on `stderr`; the GPU mapping wins
  as the cell count grows into the thousands.
- **Not for clinical use.** Educational study material only.
