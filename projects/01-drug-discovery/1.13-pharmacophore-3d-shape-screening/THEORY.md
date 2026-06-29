# THEORY — 1.13 Pharmacophore & 3D Shape Screening

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

### 1.13 Pharmacophore & 3D Shape Screening 🟢 · Established

- **Deep dive:** Pharmacophore and shape-based screening compares 3D query features (hydrogen bond donors/acceptors, hydrophobic regions, ionizable groups, molecular shape) against library conformers, capturing complementarity not encoded in 2D fingerprints. ROCS (OpenEye) uses a volumetric Gaussian overlap function (ShapeTanimoto + ColorTanimoto) that is differentiable and GPU-friendly. Screening billions of conformers requires GPU-parallel overlap computation across independent molecule pairs. This is a key pre-filtering step before docking in virtual screening pipelines.
- **Key algorithms:** Gaussian volume overlap (Tversky/Tanimoto), Fast Overlay of Chemical Structures (FOCS), pharmacophore feature matching (HBD/HBA/hydrophobic/aromatic), conformer ensemble generation, rigid body alignment (quaternion-based).
- **Datasets:** ZINC20 conformer libraries (https://zinc20.docking.org); DUD-E (https://dude.docking.org); Enamine REAL conformer sets (https://enamine.net); Directory of Useful Decoys-Enhanced including 3D conformers (verify URL).
- **Starter repos/tools:** ROCS (OpenEye/Cadence) — commercial GPU 3D shape screening (https://www.eyesopen.com/rocs); Open3DQSAR (https://open3dqsar.sourceforge.io) — open 3D-QSAR tool; RDKit shape tools (https://github.com/rdkit/rdkit) — open Gaussian overlap via PyTorch extension (verify URL); Pharmer (https://github.com/dkoes/pharmer) — open pharmacophore search tool.
- **CUDA libraries & GPU pattern:** Warp-parallel Gaussian overlap evaluation over conformer pairs; texture memory for pre-computed atom volumes; GPU-batched rigid-body alignment using quaternion representation; cuBLAS for rotation matrix applications.

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
