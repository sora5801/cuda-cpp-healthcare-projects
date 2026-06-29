# THEORY — 14.15 GPU-Accelerated Neuromorphic Biology

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

### 14.15 GPU-Accelerated Neuromorphic Biology 🔴 · Frontier/Theoretical

- **Deep dive:** Biological neural networks (retina, hippocampus, cortex) integrate spiking dynamics across billions of neurons with trillions of synaptic connections, exhibiting emergent phenomena relevant to neurological disease models and brain-computer interfaces. GPU implementations of spiking neural network (SNN) simulators (GeNN, Brian2CUDA) parallelize over neurons and synaptic update rules, achieving ~1000× speedup over CPU NEST for large-scale cortical column models. GPU neuromorphic simulation of Parkinson's basal ganglia circuits tests deep-brain stimulation parameter spaces in silico. Connection with biology: NVIDIA's H100 NVLink GPU cluster serves as a short-term neuromorphic analog for connectome-scale (C. elegans: 302 neurons, Drosophila: 130K neurons) simulation.
- **Key algorithms:** Leaky integrate-and-fire (LIF), Hodgkin-Huxley conductance-based model, spike-timing-dependent plasticity (STDP), GPU event-driven simulation, surrogate gradient training for SNN backpropagation, structural plasticity, large-scale connectome simulation.
- **Datasets:** FlyWire Drosophila Connectome — 130K neuron wiring diagram (https://flywire.ai/); Allen Brain Connectivity Atlas (https://connectivity.brain-map.org/); Blue Brain Project neocortical data (https://bluebrain.epfl.ch/); OpenNeuromorphic benchmark datasets (verify URL via openneuromorphic.org).
- **Starter repos/tools:** GeNN (GPU-enhanced Neuronal Networks) (https://github.com/genn-team/genn) — GPU SNN simulator; Brian2CUDA (https://github.com/brian-team/brian2cuda) — GPU-compiled Brian2 spiking network simulator; PyNN (https://github.com/NeuralEnsemble/PyNN) — SNN abstraction layer; NEURON (GPU branch) (https://github.com/neuronsimulator/nrn) — biophysically detailed neuron simulation with GPU backend.
- **CUDA libraries & GPU pattern:** CUDA warp-level primitives for parallel synaptic weight updates, cuSPARSE for sparse connectivity matrix (connectome), cuRAND for Poisson spike generation; pattern: connectome adjacency matrix (sparse) → GPU spike-event driven propagation → per-neuron LIF/HH ODE integration → STDP weight update → population firing-rate statistics for disease-state comparison.

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
