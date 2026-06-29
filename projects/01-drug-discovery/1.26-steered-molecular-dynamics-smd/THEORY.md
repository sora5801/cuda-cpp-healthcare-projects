# THEORY — 1.26 Steered Molecular Dynamics (SMD)

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

### 1.26 Steered Molecular Dynamics (SMD) 🟡 · Active R&D

- **Deep dive:** SMD applies external forces or velocity constraints to pull a molecule along a predefined coordinate (e.g., unbinding a ligand from a pocket), enabling calculation of work profiles and estimation of free energies via Jarzynski's equality. GPU MD allows many independent SMD trajectories to be run simultaneously, improving statistical convergence of Jarzynski estimates. Applications include estimation of drug residence time, rupture force of protein-ligand bonds, and domain opening mechanisms. NAMD pioneered GPU SMD; OpenMM provides Python-scriptable SMD via external forces.
- **Key algorithms:** Constant-velocity SMD (harmonic spring), constant-force SMD, Jarzynski equality for ΔG, fluctuation theorems, non-equilibrium work analysis, umbrella integration (follow-up).
- **Datasets:** NAMD SMD tutorials (https://www.ks.uiuc.edu/Training/Tutorials/); BindingDB residence time data (https://www.bindingdb.org); PDB force-probe simulation benchmark cases; published SMD studies on ion channels and motor proteins.
- **Starter repos/tools:** NAMD (https://www.ks.uiuc.edu/Research/namd/) — production GPU SMD; GROMACS pull code (https://github.com/gromacs/gromacs) — GPU SMD via pull-coord; OpenMM CustomExternalForce (https://github.com/openmm/openmm) — Python SMD; alchemlyb (https://github.com/alchemistry/alchemlyb) — Jarzynski post-processing.
- **CUDA libraries & GPU pattern:** Full GPU MD; custom CUDA force kernel for harmonic spring SMD; CUDA streams for multiple independent pulling trajectories; GPU memory for storing work accumulated along path.

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
