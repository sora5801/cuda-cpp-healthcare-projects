# THEORY — 1.25 Gaussian-Accelerated MD (GaMD)

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

### 1.25 Gaussian-Accelerated MD (GaMD) 🟡 · Active R&D

- **Deep dive:** GaMD adds a Gaussian-distributed boost potential to the total potential energy without predefined collective variables, enabling unconstrained enhanced sampling. Implemented in AMBER16+ GPU (pmemd.cuda) and NAMD, GaMD can reveal drug binding pathways, allosteric mechanisms, and protein folding on simulation timescales of microseconds. Unlike steered MD, no reaction coordinate is needed — GaMD monitors the system's total potential and boosts when it falls below a threshold. The boost potential statistics are used for free energy reweighting via cumulant expansion.
- **Key algorithms:** Gaussian boost potential with variance threshold, dual-boost GaMD (dihedral + total), free energy reweighting via cumulant expansion to 2nd order, principal component analysis of boosted trajectories, ligand GaMD (LiGaMD).
- **Datasets:** AMBER GaMD tutorials (https://www.med.unc.edu/pharm/miaolab/resources/gamd/); GPCRmd (https://gpcrmd.org); D. E. Shaw Research benchmark systems; PDB structures of drug targets (https://www.rcsb.org).
- **Starter repos/tools:** AMBER pmemd.cuda GaMD (https://ambermd.org) — reference GPU GaMD implementation; NAMD GaMD (https://www.ks.uiuc.edu/Research/namd/) — GaMD in NAMD for GPU simulations; GaMD analysis scripts (https://github.com/MiaoLab20/GaMD) — post-processing and reweighting tools; OpenMM GaMD plugin (verify URL).
- **CUDA libraries & GPU pattern:** Full GPU MD with real-time boost potential evaluation; CUDA kernels for on-the-fly potential monitoring and bias application; memory-efficient running statistics for Gaussian parameters; multi-GPU replica runs.

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
