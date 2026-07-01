# THEORY — 6.7 Spiking Neural Network (Point-Neuron) Simulation

> Read this after skimming `README.md`. Reading order in code: `src/lif.h`
> (the physics), `src/reference_cpu.cpp` (the serial algorithm), `src/kernels.cu`
> (the parallel version), `src/main.cu` (the driver).

---

## The science

A **spiking neural network (SNN)** models the brain at the level of individual
neurons that communicate with discrete, all-or-nothing electrical events called
**spikes** (action potentials). Unlike the artificial neural networks of deep
learning — which pass continuous activations — a biological neuron integrates
input over time and fires a spike only when its membrane voltage crosses a
threshold. The *timing* of those spikes carries the information.

A **point neuron** deliberately throws away the neuron's spatial structure (its
branching dendrites and axon, modeled elsewhere by cable equations) and keeps just
its membrane potential as a single number. This is a huge simplification, and it
is exactly what makes brain-scale simulation possible: with a handful of state
variables per cell, a single GPU can simulate **millions of neurons in real time**
(the regime GeNN, NEST GPU, and Brian2CUDA target).

We use the **leaky integrate-and-fire (LIF)** neuron, the canonical point model:

- The membrane acts like a leaky capacitor. Input current charges it up; a
  "leak" conductance constantly pulls it back toward a resting potential.
- When the voltage reaches a **threshold**, the neuron emits a spike, its voltage
  is **reset**, and it enters a brief **refractory period** during which it
  ignores input (modeling the sodium-channel recovery time).
- Each spike travels along the neuron's **synapses** to downstream targets,
  injecting a small **synaptic current** that decays exponentially.

We wire the neurons as a **Brunel random balanced network**: a population of
excitatory neurons (which push targets *toward* firing) and a smaller population
of inhibitory neurons (which push them *away*), connected sparsely at random. The
interplay of excitation and inhibition produces the irregular, self-sustaining
activity seen in real cortex — and is the single most important dynamical lesson
in this project.

---

## The math

### One LIF neuron

The sub-threshold membrane potential `V(t)` (millivolts) of a single neuron obeys
a first-order linear ODE:

```
tau_m * dV/dt = (V_rest - V) + R_m * g(t)
```

- `tau_m` — membrane time constant (ms); larger = slower to charge/leak.
- `V_rest` — resting / leak-reversal potential (mV).
- `R_m` — a membrane-resistance factor converting the synaptic conductance `g`
  into a voltage drive.
- `g(t)` — the synaptic input variable (a conductance/current proxy).

The synaptic input itself decays exponentially between arriving spikes:

```
tau_syn * dg/dt = -g       (plus an instantaneous jump +w at each incoming spike)
```

**Spike / reset / refractory rule** (the nonlinearity that makes it *spiking*):

```
if V(t) >= V_thresh:
    emit a spike at time t
    V <- V_reset
    hold V at V_reset for refractory_ms   (neuron ignores input)
```

### The network

Neuron `i`'s total synaptic input at step `t` is the sum over all presynaptic
neurons `j` that spiked (with a one-step conduction delay) of their weight, plus a
background external drive:

```
g_i(t) receives   sum_{j -> i, j spiked at t-1} w_j   +   ext_i(t)
```

with `w_j = w_exc > 0` if `j` is excitatory and `w_j = w_inh < 0` if inhibitory.
Connectivity is sparse: each neuron sends exactly `out_degree` synapses to random
targets.

### Exponential-Euler integration (the exact linear update)

Because the sub-threshold equation is *linear* in `V` when `g` is held constant
over a step of length `dt`, it has a closed-form update — no approximation error
from discretizing the leak:

```
V_inf   = V_rest + R_m * g                     (the steady state V relaxes toward)
V(t+dt) = V_inf + (V(t) - V_inf) * exp(-dt/tau_m)
g(t+dt) = g(t) * exp(-dt/tau_syn)
```

This **exponential-Euler** scheme is unconditionally stable (V never overshoots
for any `dt`) and, crucially, is a *fixed sequence of arithmetic operations* we can
reproduce bit-for-bit on both CPU and GPU. Both decay factors are pure functions of
the shared parameters, so host and device compute identical bits.

---

## The algorithm

Per timestep `t`, the simulation is two nested loops (see
`reference_cpu.cpp::simulate_cpu`, the clearest statement of it):

```
for each timestep t:
    1. zero the per-neuron synaptic-input accumulator
    2. add each neuron's deterministic external drive
    3. DELIVER: for every neuron s that spiked at t-1,
         for each of its out_degree synapses -> add w_s to the target's accumulator
    4. UPDATE: for every neuron i,
         drive_i = accumulator[i];  fired = lif_step(params, state[i], drive_i)
         record fired
    5. tally spikes; this step's spikes become next step's "delivered" set
```

The **one-step synaptic delay** (spikes from step `t` are delivered at `t+1`) is
both biologically motivated (a minimal conduction delay of one `dt`) and
algorithmically essential: step 3 reads *last* step's spikes and step 4 writes
*this* step's, so there is no read-after-write hazard — which is what lets the GPU
update all neurons simultaneously and still get the serial answer.

### Complexity

Let `N` = number of neurons, `K` = `out_degree`, `T` = steps.

| Work | Serial (CPU) | Parallel (GPU) |
|---|---|---|
| State update | `O(N)` per step | `O(1)` depth, `N` threads |
| Spike delivery | `O(N_spiked * K)` per step | `O(K)` depth per spiking thread, atomic scatter |
| Total | `O(T * N * K)` worst case | same work, spread over the device |

The delivery step dominates in an active network and is the classic SNN
bottleneck: it is an irregular, data-dependent **scatter** (only spiking neurons do
work, and they write to unpredictable targets).

---

## The GPU mapping

This project is the **per-element state update + atomic scatter** pattern
(`docs/PATTERNS.md` §1), the same shape as `5.01` (Monte-Carlo scoring) and `11.09`
(k-means accumulation). Three kernels run per step (see `kernels.cu`):

1. **`external_kernel`** — one thread per neuron; adds the deterministic background
   drive into that neuron's *private* accumulator slot (no atomics needed — one
   writer per slot).

2. **`deliver_kernel`** — one thread per *source* neuron. If it spiked last step, it
   walks its `out_degree` synapses and `atomicAdd`s its weight into each target's
   accumulator. Many source threads can hit the **same** target in the same step, so
   this must be atomic. This is the runtime-dominating kernel and the reason SNNs are
   "scatter-bound."

3. **`update_kernel`** — one thread per neuron; reads its accumulated input, calls
   the shared `lif_step()` (identical math to the CPU), records whether it spiked,
   and atomically bumps an *integer* per-step population counter.

**Ping-pong spike buffers.** The "did neuron `i` spike?" flags are double-buffered
(`d_spiked_a` / `d_spiked_b`): the update kernel writes *this* step's flags into one
buffer while the deliver kernel reads *last* step's from the other, then we swap the
pointers. This is the same double-buffering idea as the stencil flagships (`6.04`,
`14.02`), applied to the spike set.

**Memory hierarchy.** State (`v`, `g`, refractory) lives in **global memory**, one
small struct per neuron, read/written once per step (coalesced by neuron id). The
network parameters travel by value in the kernel argument (a few dozen bytes,
effectively constant/param memory). The whole time loop stays **on the device** — no
per-step host synchronization — so the timed region reflects real kernel work.

**Why not cuSPARSE?** The catalog notes cuSPARSE for synaptic summation via a sparse
matrix-vector product (SpMV): the delivery step *is* `input = W^T · spikes` with `W`
the sparse weight matrix and `spikes` a 0/1 vector. A production code often does
exactly that. We hand-roll the scatter here so the atomic-accumulation lesson is
visible rather than hidden inside a library call (CLAUDE.md §6 — no black boxes).
Swapping in cuSPARSE is an exercise; the trade-off is discussed below.

---

## Numerical considerations

### Determinism via integer fixed-point (the key trick)

Floating-point `atomicAdd` is **not associative**: `(a+b)+c != a+(b+c)` in the last
bits, and the order in which warps reach the atomic is nondeterministic. If we
summed synaptic weights as floats, the GPU total would drift from the CPU and even
from run to run — and a single last-bit difference in `g` can flip a
threshold-crossing, changing which neurons spike and cascading into a totally
different network trajectory.

The fix (`docs/PATTERNS.md` §3): accumulate synaptic input as **scaled integers**
(`SYN_FIXED_SCALE = 2^20`). Integer addition commutes, so any order gives the same
total, exactly matching the CPU's serial sum. We convert weight → `int64`, sum in
`int64`, and convert back to `double` once per neuron per step. A synapse count of
`~10^4` times a weight of `~1` stays far inside `int64`'s `~9.2e18` range, and
`2^20` gives ~6 decimal digits of weight resolution — finer than any biological
precision. `atomic_add_fixed` reinterprets the signed increment as `unsigned long
long` (two's-complement addition is bit-identical for signed/unsigned) to use CUDA's
64-bit atomic.

### Precision

All state is **double precision** and both sides run the identical operation
sequence (same `exp` calls, same order), so the *voltages* also match to ~machine
round-off (`~1e-13` mV observed). The *spike counts* — the headline result — match
**exactly** because they are integers gated on identical fixed-point arithmetic.

### Firing regime

The demo uses a deliberately brisk drive so 50 ms of simulation shows clear activity
(~94 Hz mean rate). Real cortical neurons fire far more sparsely (`~1–20` Hz) in the
asynchronous-irregular regime; reaching that regime is a matter of weaker drive,
larger networks, and true Poisson background input — left as an exercise so the tiny
offline demo stays fast and legible.

---

## How we verify correctness

Two independent checks, run every time (`main.cu`):

1. **Exact spike-count agreement (tolerance == 0).** The total spike count, the
   per-step population counts, and the per-neuron spike counts must be *identical*
   between CPU and GPU. This is only possible because of the shared `lif_step()`
   physics and the integer fixed-point accumulation — it is the strongest check we
   can make and it is exact by construction (`docs/PATTERNS.md` §4, the "integer /
   fixed-point" case, same as `5.01` / `11.09`).

2. **Final-voltage agreement (`< 1e-9` mV).** Every neuron's final membrane
   potential is compared; the observed worst difference is `~1e-13` mV, well inside
   the documented slack. This guards against a bug that happens to preserve spike
   counts but corrupts sub-threshold voltages.

A **structural** sanity check (reproducible via `make_synthetic.py` flags): with
`out_degree=0` the network fires ~581 spikes (drive only); with the full sparse
wiring it fires ~943 (recurrent excitation recruits ~62% more); with inhibition
removed (`w_inh=0`) it fires ~2052 (activity roughly doubles). This confirms the
*network* — not just the single-neuron model — is doing something, and that
excitation/inhibition balance regulates it.

---

## Where this sits in the real world

Production point-neuron simulators do the same math at enormous scale, with
engineering this teaching version deliberately omits:

- **GeNN** generates custom CUDA kernels from a user's model description, hitting
  real-time simulation of `10^6`-neuron Izhikevich networks on one GPU. **NEST GPU**
  scales across multiple GPUs toward `10^9` neurons; **Brian2CUDA** is a CUDA
  code-generation backend for Brian2; **SpikingJelly** brings SNNs into PyTorch for
  deep spiking networks.
- **Sparse delivery via SpMV.** Real codes represent connectivity as a compressed
  sparse (CSR) matrix and compute synaptic input with **cuSPARSE** SpMV, or use
  per-block spike staging in shared memory to cut atomic contention. Our hash-based
  `synapse_target()` avoids storing the matrix at all — great for a demo, but a real
  connectome (Human Connectome Project) would be a loaded adjacency list.
- **True stochastic input.** Background drive is a per-neuron **Poisson** spike train
  generated with **cuRAND**; we use a deterministic phase-rotating kick so the demo
  output is byte-reproducible.
- **Richer neuron models & plasticity.** The catalog lists Izhikevich and adaptive
  exponential IF (AdEx) neurons, **spike-timing-dependent plasticity (STDP)** for
  learning, and **delay-line spike queues** for realistic, heterogeneous conduction
  delays. Each is a natural extension of `lif_step()` and the delivery step.

### Exercises (see `README.md` for the full list)

1. Swap `lif_step` for the **Izhikevich** model (two state variables, richer
   dynamics) — keep it in `lif.h` so CPU/GPU parity is automatic.
2. Replace the deterministic drive with **cuRAND Poisson** input and verify the
   *statistics* (mean rate) instead of exact counts.
3. Represent the wiring as a **CSR matrix** and deliver spikes with **cuSPARSE**
   SpMV; compare runtime against the atomic scatter as `N` grows.
