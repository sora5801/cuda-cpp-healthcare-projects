# THEORY — 6.6 Neuronal Network Simulation (Biophysical)

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

### 6.6 Neuronal Network Simulation (Biophysical) 🟡 · Active R&D
- **Deep dive:** Simulates networks of morphologically detailed (multi-compartment) neurons using Hodgkin-Huxley-style conductance-based kinetics in each dendritic/axonal segment. A single layer-5 pyramidal cell may have 1 000+ compartments each with 10–30 gating variables, and a cortical column model contains thousands of such cells—resulting in millions of coupled ODEs. The Hines solver (tridiagonal Thomas algorithm along each dendritic tree branch) enables efficient per-cell compartmental integration, but parallelizing across cells and synapses is where GPUs excel. Spike delivery (synaptic event processing) introduces irregular memory access that benefits from GPU-side event queues.
- **Key algorithms:** Hodgkin-Huxley conductance-based kinetics, Hines tridiagonal solver (branching cable equation), Rush-Larsen exponential integration for gates, event-driven spike delivery, exponential synapse models (AMPA/NMDA/GABA), adaptive time-stepping (CVODE).
- **Datasets:** NeuroMorpho.Org — 200 000+ 3D neuronal reconstructions across 900+ species (https://neuromorpho.org); ModelDB / modeldb.science — curated computational neuron models with NEURON/GENESIS files (https://modeldb.science); Allen Brain Cell Atlas — single-cell transcriptomics + patch-seq morpho-electric data (https://portal.brain-map.org); DANDI Archive — neurophysiology datasets in NWB format (https://dandiarchive.org).
- **Starter repos/tools:** NEURON + CoreNEURON GPU (https://github.com/neuronsimulator/nrn) — canonical compartmental simulator with CUDA backend via CoreNEURON; NetPyNE (https://github.com/suny-downstate-medical-center/netpyne) — multiscale network builder on top of NEURON with HPC support; MOOSE (https://github.com/BhallaLab/moose-core) — multiscale OO simulator for neuronal + biochemical networks; Blue Brain / Open Brain Institute (https://github.com/BlueBrain) — production-grade cortical column models.
- **CUDA libraries & GPU pattern:** CoreNEURON uses cuSPARSE for Hines matrix batches; custom CUDA kernels for gate ODEs; cuRAND for stochastic synaptic release; pattern: one CUDA thread-block per cell, warp-level branching for dendritic trees; SOA memory layout for coalesced gating variable access.

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
