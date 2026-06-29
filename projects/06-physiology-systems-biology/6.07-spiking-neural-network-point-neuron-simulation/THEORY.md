# THEORY — 6.7 Spiking Neural Network (Point-Neuron) Simulation

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. Diagrams in Mermaid/ASCII
> are welcome. See [README.md](README.md) for the quick tour and build steps.
>
> _Educational only — not for clinical use._

<!-- =======================================================================
     The block below is the verbatim catalog deep-dive for this project,
     stamped in by scaffold.py as raw material. Use it to write the sections
     that follow, then DELETE it (or fold it into "The science"). Every
     TODO(theory) below must be completed before the project is "done".
     ======================================================================= -->

<details>
<summary>Catalog deep-dive (raw source material — fold into the sections below, then remove)</summary>

### 6.7 Spiking Neural Network (Point-Neuron) Simulation 🟡 · Active R&D
- **Deep dive:** Point-neuron SNN models (leaky integrate-and-fire, Izhikevich, adaptive exponential IF) sacrifice morphological detail in exchange for simulating networks of millions to billions of neurons in real time. Each neuron updates a handful of state variables per time step; spikes generate synaptic current injections to thousands of target neurons via a connectivity matrix that is typically sparse (~10 000 synapses/neuron). GeNN generates custom CUDA kernels from user model descriptions, achieving real-time simulation of 10⁶-neuron Izhikevich networks on a single GPU. NEST GPU and Brian2CUDA follow similar kernel-generation approaches.
- **Key algorithms:** Leaky integrate-and-fire (LIF), Izhikevich neuron model, adaptive exponential integrate-and-fire (AdEx), spike-timing-dependent plasticity (STDP), exponential/alpha synapse kernels, delay-line spike queues, random balanced-network (Brunel) connectivity.
- **Datasets:** Allen Brain Observatory — visual cortex spiking data from Neuropixels (https://portal.brain-map.org); DANDI Archive — electrophysiology datasets NWB format (https://dandiarchive.org); OpenNeuro — EEG/MEG recordings for network model validation (https://openneuro.org); Human Connectome Project structural connectivity matrices (https://db.humanconnectome.org).
- **Starter repos/tools:** GeNN (https://github.com/genn-team/genn) — GPU-enhanced SNN code generator (CUDA + HIP), includes Brian2GeNN and ml_genn deep SNN; SpikingJelly (https://github.com/fangwei123456/spikingjelly) — PyTorch-based SNN framework with CUDA extensions; Brian2CUDA (https://github.com/brian-team/brian2cuda) — CUDA code generation backend for Brian2; NEST GPU (https://github.com/nest/nest-simulator) — multi-GPU NEST backend scaling to 10⁹ neurons.
- **CUDA libraries & GPU pattern:** Custom generated CUDA kernels (GeNN/Brian2CUDA), cuSPARSE for synaptic current summation via sparse matrix-vector product, cuRAND for Poisson spike generation; pattern: one thread per neuron for state update, warp-shuffle for local spike detection, atomic-add for synaptic current accumulation.

</details>

---

## 1. The science

TODO(theory): The biology / medicine / physics being modeled — enough for a
reader to understand the *problem* before any math. What real-world question
does computing this answer?

## 2. The math

TODO(theory): The governing equations / formal problem statement, with **every
symbol defined** (units, ranges). State inputs, outputs, and the objective.

## 3. The algorithm

TODO(theory): Step-by-step. Include **complexity analysis**: serial cost vs. the
parallel work/depth. Where is the arithmetic intensity? What is the data-access
pattern?

## 4. The GPU mapping

TODO(theory): How the algorithm becomes **threads / blocks / grids**.
- Thread-to-data mapping (which thread owns which element).
- Launch configuration and the reasoning (block size, grid size).
- Memory hierarchy used and **why**: global / shared / registers / constant /
  texture. Where is the bandwidth bottleneck? What is the occupancy story?
- Which CUDA library (cuBLAS / cuFFT / cuRAND / cuSOLVER / Thrust) does what,
  and what it would take to write that step by hand (no black boxes — §6.1.6).

```
TODO(theory): an ASCII or Mermaid diagram of the grid/block decomposition.
```

## 5. Numerical considerations

TODO(theory): Precision (FP32 vs FP64) and why. Stability. Race conditions and
whether atomics are used. **Determinism**: does the parallel reduction reorder
floating-point sums? If so, say so and quantify the caveat.

## 6. How we verify correctness

TODO(theory): The CPU reference (`src/reference_cpu.cpp`), the **tolerance** and
why that value, and the edge cases checked. Explain why agreement between an
independent serial implementation and the GPU implementation is convincing
evidence of correctness.

## 7. Where this sits in the real world

TODO(theory): How production tools (named in the catalog "Prior art") do this
differently — what they add (scale, accuracy, features) that this teaching
version omits. If this is a 🔴 frontier project shipped as a reduced-scope
teaching version, describe the full approach here.

---

## References

TODO(theory): Papers, docs, and the starter repos from the catalog, with one
line each on what to learn from them.
