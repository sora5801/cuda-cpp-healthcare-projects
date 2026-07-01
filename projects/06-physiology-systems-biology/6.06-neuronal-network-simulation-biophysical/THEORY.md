# THEORY — 6.6 Neuronal Network Simulation (Biophysical)

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A neuron is an electrically excitable cell. Its membrane separates charge, so it
has a **voltage** `V` across it. Embedded in the membrane are **ion channels** —
protein pores that open and close depending on `V` and let specific ions (Na⁺,
K⁺) flow. When enough Na⁺ channels open, `V` shoots up to ~+40 mV and back down
in about a millisecond: an **action potential** (a "spike"). Spikes are the
neuron's output.

Two facts make a *biophysical* network model harder than a toy one:

1. **A neuron has spatial extent.** A real cortical neuron is a tree — a soma
   with an elaborate dendritic arbor. Voltage is not one number; it varies along
   the tree, and current diffuses through the thin, resistive cytoplasm from one
   patch of membrane to the next. We chop the tree into **compartments**, each a
   small isopotential patch, and track a voltage per compartment. A detailed
   layer-5 pyramidal cell has *thousands* of compartments.

2. **Neurons talk through synapses.** When a presynaptic neuron spikes, it
   releases neurotransmitter that briefly opens receptor channels on the
   postsynaptic cell, injecting current. Excitatory (AMPA/NMDA) synapses push `V`
   up; inhibitory (GABA) synapses push it down.

This project builds a **reduced but honest** version: `ncell` neurons, each an
unbranched multi-compartment cable of Hodgkin–Huxley membrane, wired into a ring
by excitatory synapses. Kick one cell and a spike wave circles the ring — the
simplest network phenomenon that exercises *all* the machinery (channel kinetics,
the cable equation, and event-driven synaptic coupling).

---

## 2. The math

### 2.1 Hodgkin–Huxley membrane (per compartment)

For one isopotential patch of membrane, charge conservation gives

```
  C_m dV/dt = -(I_Na + I_K + I_L) + I_coupling + I_syn
```

with the classic Hodgkin–Huxley (1952) ionic currents

```
  I_Na = gNa · m³ · h · (V - E_Na)      (Na⁺, fast; m activates, h inactivates)
  I_K  = gK  · n⁴     · (V - E_K)       (K⁺, delayed rectifier)
  I_L  = gL          · (V - E_L)        (passive leak)
```

Each **gating variable** `x ∈ {m, h, n}` is a fraction in [0,1] obeying

```
  dx/dt = alpha_x(V)·(1-x) - beta_x(V)·x
        = (x_inf(V) - x) / tau_x(V)
```

where `x_inf = alpha/(alpha+beta)` and `tau_x = 1/(alpha+beta)`. The voltage-
dependent rate functions `alpha_x(V)`, `beta_x(V)` are the standard squid-axon
expressions (coded and commented in `src/neuron.h`). Units: `V` in mV, `g` in
mS/cm², `C_m` in µF/cm², time in ms, currents in µA/cm².

### 2.2 The cable equation (compartment coupling)

Adjacent compartments are joined by an axial (intracellular) conductance
`g_axial`. The coupling current into compartment `c` from its neighbours is

```
  I_coupling(c) = g_axial · (V[c-1] - V[c]) + g_axial · (V[c+1] - V[c])
```

This is a discrete Laplacian — a diffusion of voltage along the cable. Because it
couples the compartments, they cannot be integrated independently; they form a
**linear system** each timestep.

### 2.3 Synapse (network coupling)

Each cell has one excitatory synaptic conductance `g_syn` at the soma:

```
  dg_syn/dt = -g_syn / tau_syn ,     g_syn += w_syn on each presynaptic spike
  I_syn = g_syn · (E_syn - V_soma)        (E_syn = 0 mV, excitatory)
```

`g_syn` is a decaying exponential kicked upward by incoming spikes — the standard
single-exponential AMPA model.

### 2.4 Spike detection

A cell "spikes" when its **soma voltage crosses a threshold upward**
(`V_soma` was `< V_thresh`, now `≥ V_thresh`). That boolean, delayed one step, is
the event delivered to its postsynaptic partner.

---

## 3. The algorithm

We advance the whole network by a fixed timestep `dt` using operator splitting,
the same ordering NEURON uses:

**Per neuron, per step:**

1. **Gates (Rush–Larsen).** Freeze `V`, update each gate with the *exact*
   solution of its linear ODE over `dt`:
   ```
   x ← x_inf + (x - x_inf)·exp(-dt/tau_x)
   ```
   Forward Euler can overshoot [0,1] for large `dt`; the exponential rule cannot.
   That unconditional stability is why every production simulator uses it.

2. **Synapse.** Decay `g_syn ← g_syn·exp(-dt/tau_syn)`; if the presynaptic cell
   spiked last step, `g_syn += w_syn`.

3. **Voltages (implicit cable solve).** Treat the linear diffusion + synapse
   terms **implicitly** (backward Euler) and the HH ionic term explicitly
   (evaluated at the old `V`). Backward Euler of

   ```
   C_m/dt·(V_new - V_old) = -I_ion(V_old) + coupling(V_new) + syn(V_new)
   ```

   rearranges, per compartment, into a **tridiagonal** system `A·V_new = d`:

   ```
   a[c] = -g_axial                          (sub-diagonal: left neighbour)
   c[c] = -g_axial                          (super-diagonal: right neighbour)
   b[c] =  C_m/dt + (#neighbours)·g_axial   (+ g_syn on the soma row)
   d[c] =  C_m/dt·V_old[c] - I_ion[c]       (+ g_syn·E_syn on the soma row)
   ```

4. **Hines/Thomas solve.** Solve `A·V_new = d` with the **Thomas algorithm** —
   the Hines solver specialised to a line: one forward sweep eliminating the
   sub-diagonal, one back-substitution sweep. `A` is diagonally dominant (a
   backward-Euler diffusion operator), so no pivoting is needed and the solve is
   unconditionally stable.

5. **Spike test.** Did `V_soma` cross `V_thresh` upward? Record it for delivery
   next step.

**Network coupling** uses a **step-synchronous double buffer**: `spike_prev[]`
holds who fired *last* step and drives synapses *this* step; `spike_now[]` is
filled this step and becomes `spike_prev` next step. Every cell's update depends
only on last step's spikes, never on the order cells are processed — so the
result is **order-independent and deterministic**, which is exactly what lets the
serial CPU loop and the parallel GPU kernel agree bit-for-bit.

### Complexity

Let `N` = neurons, `K` = compartments/cell, `T` = timesteps.

- **Serial (CPU):** `O(N · K · T)` — the Thomas solve is `O(K)` per cell per step.
- **Parallel (GPU):** the `N` cells at each step are independent, so with `P`
  processors the wall-clock work is `O((N/P)·K·T)`; the `T` steps stay serial
  (causal). This is the classic "parallelise the population, march time" pattern
  (docs/PATTERNS.md §1, the ensemble row).

---

## 4. The GPU mapping

```
   grid  = ceil(ncell / 128) blocks      one kernel launch PER timestep
   block = 128 threads
   thread i  ->  neuron i        (state persists in global memory across launches)

   ┌── step t launch ─────────────────────────────────────────────┐
   │ thread i reads spike_prev[pre(i)]      (last step's spike)    │
   │ step_neuron(net[i]):  gates -> synapse -> Hines solve -> spike│
   │ writes spike_now[i]                                           │
   └──────────────────────────────────────────────────────────────┘
   host swaps spike_prev <-> spike_now   (ping-pong)   then launches step t+1
```

**Why one thread per neuron (not per compartment)?** With `K ≤ 8` compartments a
whole neuron's state (voltages + three gates + synapse ≈ a few hundred bytes)
fits comfortably in a thread's registers/local memory, and the Thomas solve is an
inherently *sequential* recurrence along the cable — awkward to split across
threads. One thread per cell keeps the recurrence local and the parallelism where
it is abundant: across cells. (Production simulators with thousands of
compartments per cell instead use **one block per cell** and walk the tree with a
warp; see §7.)

**Why one launch per timestep?** Threads in different blocks cannot synchronise
inside a kernel, but synaptic coupling needs a **grid-wide barrier** between steps
(every cell must finish step `t` before any reads step `t`'s spikes). The kernel
boundary *is* that barrier. The two spike buffers ping-pong across launches. An
advanced alternative — a single persistent kernel with a cooperative-groups grid
barrier — removes the launch overhead and is a suggested exercise.

**Memory spaces.** `net[]`, `results[]`, and the two spike buffers live in
**global memory** (they must persist between launches). Each thread's working
arrays for the Thomas solve are **local/registers**. There is no shared memory in
this teaching version because each thread owns an independent small system;
shared memory would matter for the one-block-per-cell layout.

**Determinism / no atomics.** Nothing is reduced across threads, so there are no
`atomicAdd`s and no float-reordering nondeterminism (docs/PATTERNS.md §3). Each
thread writes only its own cell's outputs.

---

## 5. Numerical considerations

- **Precision: FP64 (double) everywhere.** The CPU and GPU must produce *identical*
  soma-voltage sequences so the integer spike counts match exactly. In double
  precision, with the same operation order in the shared `neuron.h`, the two agree
  to the last bit here — the verification tolerance is literally **0**. (Long
  FP32 iterative solvers would instead diverge by ~1e-5 via FMA differences; we
  avoid that by using FP64 and identical arithmetic — docs/PATTERNS.md §4.)
- **Stability.** Rush–Larsen keeps gates in [0,1] for any `dt`; backward Euler of
  the cable makes the coupling unconditionally stable. The *explicit* HH ionic
  term is the one accuracy-limited piece, which is why we use a small `dt`
  (0.025 ms) — the standard compartmental-modelling choice.
- **The rate-function singularities.** `alpha_m`, `alpha_n` have a removable 0/0
  singularity at specific voltages; we substitute the l'Hôpital limit within a
  1e-6 guard (see `neuron.h`) so the code is finite everywhere.
- **Race conditions.** Avoided structurally: the one-step spike delay + kernel
  boundary means no thread ever reads another thread's *same-step* result.

---

## 6. How we verify correctness

- **CPU reference (`reference_cpu.cpp`)** integrates the same network serially,
  calling the *same* `step_neuron()` in `neuron.h` the GPU calls. This is the
  shared `__host__ __device__` core idiom (docs/PATTERNS.md §2).
- **Exact integer check.** `main.cu` compares every cell's `spike_count` and
  `first_spike` between CPU and GPU; any mismatch is a real bug (a race, a wiring
  error), not floating-point noise, so a single mismatch fails the run.
- **Scientific sanity (beyond CPU==GPU).** The result *should* be a travelling
  wave: with the committed sample the first-spike step increases by ~72 steps
  (~1.8 ms) per cell around the ring, all 16 cells fire, and the mean rate is a
  physiological ~67 Hz. A learner can see the wave in the per-cell output, not
  just trust a PASS.
- **Determinism.** Running the demo twice yields byte-identical `stdout`.

---

## 7. Where this sits in the real world

Production biophysical simulators differ in scale and sophistication, not in
kind:

- **NEURON / CoreNEURON.** NEURON is the de-facto standard; CoreNEURON is its
  GPU/vectorised backend. It handles *branching* dendritic trees — Hines'
  contribution was an ordering of tree compartments so the elimination stays
  `O(K)` even on a branched morphology (our unbranched cable is the special case
  where that ordering is trivial). CoreNEURON uses **cuSPARSE** for batched Hines
  matrices, custom kernels for the gate ODEs, **cuRAND** for stochastic synaptic
  release, a **struct-of-arrays** layout for coalesced gate access, and
  **one block per cell** with a warp walking the tree.
- **Adaptive time-stepping (CVODE).** Real runs often use variable-order,
  variable-`dt` implicit solvers instead of our fixed `dt` — bigger steps when
  the dynamics are smooth, tiny steps through a spike.
- **Richer channels & synapses.** Dozens of ion-channel types, NMDA (voltage-
  dependent Mg²⁺ block), GABA inhibition, short-term plasticity, and realistic
  connectivity from connectomics — all bolt onto the same per-compartment update.
- **Event queues.** With sparse, delayed, weighted connectivity, spike delivery
  becomes an **event-driven priority queue** rather than our dense one-step ring;
  that irregular memory access is a real GPU engineering challenge (the catalog's
  "GPU-side event queues").

Our version keeps the *concepts* — HH kinetics, Rush–Larsen, the Hines/Thomas
solve, event-driven synaptic coupling, one-cell-per-thread parallelism — while
shrinking every axis so the whole thing fits in one readable file and runs in a
second. That is the point: understand the machine before scaling it.
