# THEORY — 1.33 Interaction Fingerprinting & Binding-Mode Clustering

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

### 1.33 Interaction Fingerprinting & Binding-Mode Clustering 🟡 · Active R&D

- **Deep dive:** Protein-ligand interaction fingerprints (IFPs) encode which residues form HBs, hydrophobic contacts, π-stacking, salt bridges, and halogen bonds with a ligand. IFPs enable rapid clustering of thousands of docking poses or MD trajectory frames into distinct binding modes, analogous to chemical fingerprints but for structural biology. GPU-parallel distance/angle evaluation over millions of frame-residue pairs makes real-time IFP generation from MD trajectories feasible. Applications include binding-mode prediction validation and SAR-IFP correlation for lead optimization.
- **Key algorithms:** PLEC (protein-ligand extended connectivity), PLIF (protein-ligand interaction fingerprint), SIFt (structural interaction fingerprint), Tanimoto IFP similarity, GPU-parallel distance/angle kernels, GPU k-means clustering on IFP bit-vectors.
- **Datasets:** PDB-bind complex structures (http://www.pdbbind.org.cn); KLIFS (https://klifs.net); ChEMBL bioactivity with structures (https://www.ebi.ac.uk/chembl/); BindingDB (https://www.bindingdb.org).
- **Starter repos/tools:** ProLIF (https://github.com/chemosim-lab/ProLIF) — protein-ligand interaction fingerprints from MD trajectories; ODDT (https://github.com/oddt/oddt) — open drug discovery toolkit with IFP; Pharmit (https://pharmit.csb.pitt.edu) — pharmacophore + shape screening; KLIFS Python (https://github.com/volkamerlab/kissim) — kinase IFP features.
- **CUDA libraries & GPU pattern:** CUDA kernels for atom-pair distance/angle evaluation over frame×residue grid; GPU popcount for IFP Tanimoto; cuML GPU k-means on IFP matrix; RAPIDS cuDF for MD frame I/O and selection.

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
