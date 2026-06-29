# THEORY — 1.21 Polarizable / AMOEBA Force Field MD

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

### 1.21 Polarizable / AMOEBA Force Field MD 🟡 · Active R&D

- **Deep dive:** Classical fixed-charge force fields miss polarization effects crucial for accurate binding free energies and ionic interactions. The AMOEBA force field includes point multipoles (up to quadrupoles) and induced dipoles solved self-consistently at each MD step via an iterative solver (conjugate gradient). This increases cost ~10× over AMBER but GPU implementation in Tinker-HP achieves >200-fold speedup over single-CPU, making microsecond AMOEBA simulations of large proteins feasible. Applications include protein-ligand FEP with AMOEBA and pKa prediction in complex electrostatic environments.
- **Key algorithms:** Induced dipole iteration (conjugate gradient), Ewald summation for multipoles (PME-multipole), AMOEBA water model, HIPPO force field, PIMD with polarizable FF.
- **Datasets:** AMOEBA protein force field parameter files (https://github.com/TinkerTools/tinker); WaterMap/hydration site datasets (Schrodinger, verify URL); BindingDB experimental affinities (https://www.bindingdb.org); NIST thermophysical properties (https://webbook.nist.gov).
- **Starter repos/tools:** Tinker-HP (https://github.com/TinkerTools/tinker-hp) — massively parallel GPU AMOEBA MD; OpenMM AMOEBA plugin (https://github.com/openmm/openmm) — AMOEBA on CUDA; Tinker9 (https://github.com/TinkerTools/tinker9) — GPU-native Tinker rewrite; AMOEBA+ FF parameters (https://github.com/TinkerTools/poltype2).
- **CUDA libraries & GPU pattern:** Custom CUDA conjugate-gradient solver for induced dipoles; cuFFT for multipole PME; warp-synchronous reduction for energy accumulation; multi-GPU via MPI domain decomposition with NCCL.

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
