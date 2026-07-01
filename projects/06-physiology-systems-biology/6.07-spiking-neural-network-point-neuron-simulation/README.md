# 6.7 — Spiking Neural Network (Point-Neuron) Simulation

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.7`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

This project simulates a small network of **spiking neurons** on the GPU. Each
neuron is a **leaky integrate-and-fire (LIF)** "point neuron": a single membrane
voltage that charges up under synaptic input, leaks back toward rest, and fires a
discrete **spike** when it crosses a threshold — then resets and goes briefly
refractory. The neurons are wired into a sparse **Brunel balanced network**
(excitatory + inhibitory), so a spike from one neuron nudges thousands of others,
and the interplay of excitation and inhibition produces self-sustaining,
cortex-like activity. Every timestep the GPU updates all neurons in parallel (one
thread per neuron) and delivers spikes along the network with an **atomic scatter**
— the two operations that a real brain-scale simulator like GeNN is built around.
The result is verified against a serial CPU reference that runs the identical
physics, so the GPU spike counts match the CPU **exactly**.

## What this computes & why the GPU helps

Point-neuron SNN models (leaky integrate-and-fire, Izhikevich, adaptive exponential
IF) sacrifice morphological detail in exchange for simulating networks of millions
to billions of neurons in real time. Each neuron updates a handful of state
variables per time step; spikes generate synaptic current injections to thousands
of target neurons via a connectivity matrix that is typically sparse (~10 000
synapses/neuron). GeNN generates custom CUDA kernels from user model descriptions,
achieving real-time simulation of 10⁶-neuron Izhikevich networks on a single GPU.
NEST GPU and Brian2CUDA follow similar kernel-generation approaches.

**The parallel bottleneck:** the per-step **spike delivery** — scattering each
spiking neuron's weight into all of its postsynaptic targets. In an active network
this is an irregular, data-dependent write to unpredictable memory locations, with
many source neurons hitting the same target. On the GPU each source neuron is a
thread and it `atomicAdd`s into its targets; we accumulate in **integer
fixed-point** so those atomics are deterministic and match the CPU bit-for-bit. The
state update (one thread per neuron) is embarrassingly parallel and scales linearly
with neuron count — which is why a GPU can run networks orders of magnitude larger
than a CPU in the same wall-clock time.

## The algorithm in brief

- **Leaky integrate-and-fire (LIF)** neurons with an **exponential-Euler** update
  (the exact solution of the linear leak over a step — unconditionally stable).
- **Exponentially-decaying synapses** (a `g`-variable per neuron).
- **Random balanced-network (Brunel) connectivity** — sparse, deterministic wiring
  from a hash of `(seed, source, synapse index)` (no stored matrix).
- **One-step synaptic delay** so parallel and serial updates agree exactly.
- **Integer fixed-point atomic accumulation** of synaptic input for determinism.

The catalog also lists Izhikevich / AdEx neurons, STDP plasticity, and delay-line
spike queues — natural extensions discussed in [THEORY.md](THEORY.md), which has the
full science → math → algorithm → GPU-mapping derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/spiking-neural-network-point-neuron-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/spiking-neural-network-point-neuron-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\spiking-neural-network-point-neuron-simulation.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/network.txt`, prints the result,
shows the GPU-vs-CPU agreement check, and prints a timing line on stderr.

## Data

- **Sample (committed):** `data/sample/network.txt` — a tiny, **synthetic** network
  config (sizes, weights, biophysics) so the demo runs with zero downloads. No
  patient data. Regenerate/resize with `scripts/make_synthetic.py`.
- **Full dataset:** none required. `scripts/download_data.ps1` / `.sh` print links to
  real recordings (Allen Brain Observatory, DANDI) and structural connectomes (Human
  Connectome Project) you could validate against — without bypassing any credentials.
- **Provenance, format & field meanings:** see [data/README.md](data/README.md).

Catalog dataset notes: Allen Brain Observatory — visual cortex spiking data from
Neuropixels (https://portal.brain-map.org); DANDI Archive — electrophysiology
datasets NWB format (https://dandiarchive.org); OpenNeuro — EEG/MEG recordings for
network model validation (https://openneuro.org); Human Connectome Project
structural connectivity matrices (https://db.humanconnectome.org).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
6.7 -- Spiking Neural Network (Point-Neuron) Simulation
network: 200 neurons (160 exc + 40 inh), out_degree=16, 500 steps @ dt=0.10 ms (50.0 ms)
weights: w_exc=0.900  w_inh=-2.200  ext_kick=1.800 every 30
total spikes (GPU) = 943   mean rate = 94.300 Hz
population spike raster (step: count):
  [  0:  0] [ 71:  0] [142:  1] [213:  3] [285:  3] [356:  4] [427:  5] [499:  2]
most active neurons (id:spikes:type):
  [84:10:E] [16:9:E] [29:9:E] [59:9:E] [61:9:E]
RESULT: PASS (GPU spike counts match CPU exactly; final V within 1e-09 mV)
```

The program computes the result on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts they agree: the total, per-step,
and per-neuron spike counts must be **identical** (exact integer check), and final
membrane voltages must agree to `< 1e-9` mV. That agreement is the correctness
guarantee. Timing goes to stderr (a teaching artifact, not a benchmark).

## Code tour

Read in this order:

1. [`src/lif.h`](src/lif.h) — **start here.** The shared `__host__ __device__`
   physics: the LIF neuron update (`lif_step`), the fixed-point synapse helpers, and
   the deterministic connectivity/init. Both CPU and GPU include this, which is why
   they match exactly.
2. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the serial simulation, the
   clearest statement of the per-step algorithm.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the three per-step kernels (external drive,
   atomic spike delivery, neuron update) and the on-device time loop.
5. [`src/main.cu`](src/main.cu) — loads the network, runs CPU + GPU, verifies, reports.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, I/O helpers.

## Prior art & further reading

GeNN (https://github.com/genn-team/genn) — GPU-enhanced SNN code generator (CUDA +
HIP), includes Brian2GeNN and ml_genn deep SNN; SpikingJelly
(https://github.com/fangwei123456/spikingjelly) — PyTorch-based SNN framework with
CUDA extensions; Brian2CUDA (https://github.com/brian-team/brian2cuda) — CUDA code
generation backend for Brian2; NEST GPU (https://github.com/nest/nest-simulator) —
multi-GPU NEST backend scaling to 10⁹ neurons.

- **GeNN** — study how a code generator turns a model description into the exact
  kernels we hand-wrote here (state update + sparse delivery).
- **Brian2 / Brian2CUDA** — the readable Python front end; compare its synapse model
  to our `lif.h`.
- **NEST GPU** — how the same idea scales across multiple GPUs to `10^9` neurons.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Per-neuron state update (**one thread per neuron**, like the `9.02` ensemble ODE)
combined with an **atomic scatter** for synaptic delivery (like `5.01` Monte-Carlo
scoring and `11.09` k-means accumulation), made deterministic with **integer
fixed-point** accumulation, and **ping-pong buffers** for the spike set (like the
`6.04`/`14.02` stencils). The catalog's reference pattern — cuSPARSE SpMV for
synaptic summation and cuRAND for Poisson input — is described in
[THEORY.md](THEORY.md) and left as exercises; we hand-roll the scatter so the atomic
lesson is visible rather than hidden in a library (CLAUDE.md §6, no black boxes).

## Exercises

1. **Izhikevich neuron.** Replace `lif_step` in `lif.h` with the two-variable
   Izhikevich model (richer bursting/adapting dynamics). Because it lives in the
   shared header, CPU/GPU parity is automatic.
2. **cuRAND Poisson drive.** Swap the deterministic `external_drive_fixed` for a true
   per-neuron Poisson spike train (cuRAND). Verify the *statistics* (mean rate)
   rather than exact counts, and discuss why the exact check no longer applies.
3. **Sparse-matrix delivery.** Build the connectivity as a CSR matrix and deliver
   spikes with **cuSPARSE** SpMV (`input = Wᵀ·spikes`). Compare runtime against the
   atomic scatter as `N` grows — where does each win?
4. **Scale it up.** Run `make_synthetic.py --n-exc 8000 --n-inh 2000 --steps 2000`
   and watch the CPU/GPU timing gap flip as the network gets large enough to hide
   launch overhead.
5. **Find the balanced regime.** Sweep `w_exc`/`w_inh`/`ext_kick` to push the mean
   rate down to a realistic `~5–10` Hz asynchronous-irregular state, and plot the
   population raster.

## Limitations & honesty

- **Synthetic, tiny, and fast-firing.** The committed network is 200 neurons over
  50 ms of simulated time, with a deliberately brisk drive (~94 Hz mean rate) so the
  short offline demo shows clear activity. Real cortex is far larger and fires
  sparsely (`~1–20` Hz). The data is **synthetic** and labeled so everywhere.
- **Reduced-scope teaching version.** We use LIF (not Izhikevich/AdEx), a
  deterministic background drive (not cuRAND Poisson), a hand-rolled atomic scatter
  (not cuSPARSE SpMV), a single fixed conduction delay (not delay-line queues), and
  no plasticity (no STDP). Each simplification is deliberate and its production
  counterpart is named in [THEORY.md](THEORY.md).
- **The GPU is slower here — on purpose-honest.** With 200 neurons and 3 kernel
  launches per step the run is launch-bound; the GPU's advantage only appears at the
  `10^5`–`10^6`-neuron scale these simulators exist for. The timing is a teaching
  artifact, never a benchmark claim.
- **Not clinical.** Nothing here diagnoses, treats, or models a specific patient.
  Study material only (CLAUDE.md §1, §8).
