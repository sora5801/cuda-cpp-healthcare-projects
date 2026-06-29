# THEORY — 1.14 Conformer Ensemble Generation

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

### 1.14 Conformer Ensemble Generation 🟢 · Established

- **Deep dive:** Drug-like molecules are flexible; binding-relevant conformers must be generated before 3D screening or docking. RDKit ETKDG embeds molecules in 3D using experimental torsion knowledge (ETKDGv3) and distance geometry; generation of thousands of conformers per molecule for a library of millions is a CPU bottleneck. GPU acceleration is achieved by batching conformer embedding across many molecules simultaneously. Alternatively, ML-based conformer predictors (OMEGA-ML, GeoMol, TorsionalDiffusion) use GPU neural networks trained on crystallographic torsion distributions.
- **Key algorithms:** Experimental torsion-angle knowledge distance geometry (ETKDG), MMFF94/UFF energy minimization, breadth-first conformer pruning (RMSD clustering), torsional diffusion (ML), graph neural network conformer prediction.
- **Datasets:** GEOM — 37M conformers of drug-like molecules with DFT energies (https://github.com/learningmatter-mit/geom); CSD torsion library (https://www.ccdc.cam.ac.uk); COD (Crystallography Open Database) — crystal structures for torsion validation (https://www.crystallography.net); PDB small molecule conformations (https://www.rcsb.org).
- **Starter repos/tools:** RDKit ETKDG (https://github.com/rdkit/rdkit) — standard conformer engine, GPU-batched via RDKit-GPU (verify URL); TorsionalDiffusion (https://github.com/gcorso/torsional-diffusion) — GPU diffusion model for conformer sampling; GeoMol (https://github.com/PattanaikL/GeoMol) — ML conformer prediction; Frog2 / OMEGA (OpenEye, commercial) — fast conformer generators.
- **CUDA libraries & GPU pattern:** Batched SVD/distance geometry on GPU via cuSOLVER; custom CUDA kernels for pairwise RMSD computation; GPU-parallel MMFF energy minimization via molecular gradient descent; PyTorch-based diffusion inference with CUDA tensors.

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
