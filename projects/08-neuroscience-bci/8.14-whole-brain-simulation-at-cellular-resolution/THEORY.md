# THEORY — 8.14 Whole-Brain Simulation at Cellular Resolution

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

### 8.14 Whole-Brain Simulation at Cellular Resolution 🔴 · Frontier/Theoretical
- **Deep dive:** Simulating the entire mouse brain (~70 million neurons, ~1 trillion synapses) or human brain (~86 billion neurons) at point-neuron resolution requires exascale computing. Current GPU-capable implementations target mouse brain at simplified LIF models and are a grand-challenge benchmark for neuromorphic hardware. Even 1% of the human brain (~860 million neurons) needs ~10 GB of synaptic state alone. GPU cluster approaches (NEST GPU across many nodes, or NVIDIA H100 NVLink cluster) target this regime; the key bottleneck is sparse synaptic event communication.
- **Key algorithms:** Leaky integrate-and-fire / Izhikevich / AdEx at scale, distributed spike event routing (MPI + NCCL), synaptic delay management (distributed ring buffers), STDP online learning at scale, heterogeneous connectivity (random, small-world, structural), balanced E/I network dynamics (Brunel network).
- **Datasets:** Allen Mouse Brain Connectivity Atlas (https://portal.brain-map.org); HCP structural connectivity (https://db.humanconnectome.org); FlyEM Janelia Drosophila connectome for validation (https://neuprint.janelia.org); Blue Brain Cell Atlas (https://portal.brain-map.org).
- **Starter repos/tools:** NEST GPU (https://github.com/nest/nest-simulator) — multi-GPU NEST with CUDA kernel for large network simulation; GeNN (https://github.com/genn-team/genn) — GPU SNN code generation targeting large networks; The Virtual Brain (https://github.com/the-virtual-brain/tvb-root) — whole-brain mean-field at lower resolution; SpikingJelly (https://github.com/fangwei123456/spikingjelly) — PyTorch SNN framework scalable to large populations.
- **CUDA libraries & GPU pattern:** NCCL for multi-GPU spike event all-to-all communication; custom CUDA kernels for per-neuron state update with register-resident state; cuSPARSE for connectivity matrix-vector product; pattern: GPU-direct MPI for spike routing, neuron state in global memory with warp-coalesced access, NVLink for intra-node GPU communication.

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
