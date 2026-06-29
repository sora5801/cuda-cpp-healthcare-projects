# THEORY — 2.26 Hydrogen Bond Network & Water Placement Analysis

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

### 2.26 Hydrogen Bond Network & Water Placement Analysis 🟡 · Active R&D

- **Deep dive:** Water molecules mediate protein-ligand interactions at binding sites; their correct placement is critical for accurate docking and scoring. GPU-accelerated MD generates explicit water trajectories from which statistical water occupancy maps (WaterMap, GIST) are computed. The Grid Inhomogeneous Solvation Theory (GIST) requires computing per-voxel thermodynamic quantities (energy, entropy) across millions of trajectory frames — a GPU-parallelizable grid accumulation problem. High-occupancy waters indicate entropically costly displacement sites; displacing them with ligand atoms typically yields affinity gains.
- **Key algorithms:** Grid Inhomogeneous Solvation Theory (GIST), Inhomogeneous Fluid Solvation Theory (IFST), 3D water occupancy map from MD, nearest-neighbor entropy estimation, water bridge H-bond network graph, explicit water clustering.
- **Datasets:** SAMPL water placement challenges (https://github.com/samplchallenges/SAMPL); explicit-solvent PDB structures (https://www.rcsb.org); benchmark sets for WaterMap validation (Schrodinger, verify URL); GIST reference calculations for T4 lysozyme and FKBP12.
- **Starter repos/tools:** GISTPP (https://github.com/liedlgroup/gist-pp) — GIST water thermodynamics analysis; cpptraj GIST (https://github.com/Amber-MD/cpptraj) — AMBER trajectory analysis with GIST; MDAnalysis water analysis (https://github.com/MDAnalysis/mdanalysis) — H-bond and water bridge analysis; WaterMD (verify URL) — GPU-accelerated solvation free energy.
- **CUDA libraries & GPU pattern:** GPU grid accumulation kernels for GIST voxel energy/entropy (atomic updates); custom CUDA nearest-neighbor entropy estimation; MDAnalysis GPU trajectory streaming; GPU-parallel water oxygen occupancy histogramming.

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
