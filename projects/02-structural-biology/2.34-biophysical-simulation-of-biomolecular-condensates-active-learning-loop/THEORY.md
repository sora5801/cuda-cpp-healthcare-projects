# THEORY — 2.34 Biophysical Simulation of Biomolecular Condensates (Active Learning Loop)

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

### 2.34 Biophysical Simulation of Biomolecular Condensates (Active Learning Loop) 🔴 · Frontier/Theoretical

- **Deep dive:** Understanding the sequence determinants of biomolecular condensate properties (surface tension, viscosity, partition coefficients of client molecules) requires an active learning loop: GPU CG-MD generates condensate properties, a surrogate model (GNN on sequence) learns the property landscape, and Bayesian optimization proposes new sequences. This closes the loop between sequence, structure, and function for disordered proteins. GPU acceleration enables the necessary throughput (hundreds of condensate simulations per iteration). Applications include designing condensate-targeting therapeutics and understanding IDR evolution.
- **Key algorithms:** Bayesian active learning on sequence space, GNN surrogate for condensate properties, GPU CG-MD with IDP force fields, coexistence concentration estimation, diffusion coefficient estimation from MSD, transfer matrix for condensate-client partition.
- **Datasets:** PhaSePro (https://phasepro.elte.hu); DisProt (https://disprot.org); experimental LLPS partition coefficient datasets (verify URL); published condensate MD trajectory datasets (FUS, TDP-43, hnRNPA1).
- **Starter repos/tools:** CALVADOS 2 (https://github.com/KULL-Centre/CALVADOS) — GPU-compatible residue-level IDP model; OpenMM + GNN surrogate (https://github.com/openmm/openmm) — active learning condensate loop; LAMMPS GPU (https://github.com/lammps/lammps) — large-scale CG condensate simulation; BoTorch (https://github.com/pytorch/botorch) — GPU Bayesian optimization for sequence design.
- **CUDA libraries & GPU pattern:** GPU CG-MD for condensate equilibration; PyTorch GNN surrogate on sequence features; BoTorch GPU Bayesian optimization; multi-GPU ensemble of condensate simulation replicas; GPU MSD calculation for diffusion coefficient.

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
