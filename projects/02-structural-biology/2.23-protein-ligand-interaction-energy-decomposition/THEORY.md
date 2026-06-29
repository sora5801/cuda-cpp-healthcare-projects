# THEORY — 2.23 Protein-Ligand Interaction Energy Decomposition

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

### 2.23 Protein-Ligand Interaction Energy Decomposition 🟡 · Active R&D

- **Deep dive:** Per-residue energy decomposition (MM-GBSA per-residue, FEP energy components) identifies which protein residues contribute most to ligand binding, guiding lead optimization and resistance mutation analysis. GPU MD trajectories provide snapshots; GPU-parallel per-residue energy evaluation attributes contributions from each residue. This reveals hot-spot residues for mutational scanning, identifies water-mediated interactions, and explains selectivity across protein family members. Kinase resistance mutation mapping in oncology is a prime application.
- **Key algorithms:** MM-GBSA per-residue energy decomposition, pairwise interaction energy, electrostatic + VDW component separation, water bridge detection, solvent contribution per residue, FEP component analysis.
- **Datasets:** PDB-bind (http://www.pdbbind.org.cn); resistance mutation datasets (ClinVar, https://www.ncbi.nlm.nih.gov/clinvar/); KLIFS kinase binding data (https://klifs.net); ChEMBL activity data for target families (https://www.ebi.ac.uk/chembl/).
- **Starter repos/tools:** AMBER MMPBSA.py decomp (https://ambermd.org/AmberTools.php) — per-residue energy decomposition; gmx_MMPBSA (https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA) — GROMACS MM-GBSA decomposition; MDAnalysis (https://github.com/MDAnalysis/mdanalysis) — pairwise residue-ligand contact analysis; ProLIF (https://github.com/chemosim-lab/ProLIF) — IFP for binding mode decomposition.
- **CUDA libraries & GPU pattern:** GPU MD trajectory generation; CUDA parallel per-residue GB energy evaluation; GPU-batched snapshot processing (N frames × M residues); cuBLAS for energy matrix accumulation.

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
