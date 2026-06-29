# THEORY — 1.15 Protein-Ligand Binding Affinity Scoring (ML)

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

### 1.15 Protein-Ligand Binding Affinity Scoring (ML) 🟡 · Active R&D

- **Deep dive:** End-to-end ML scoring functions learn protein-ligand interaction energy surrogates directly from structural data, bypassing physics-based force fields. Models range from 3D-CNNs over voxelized complexes to equivariant GNNs over atom graphs to transformer co-folding models (NeuralPLexer3). GPU inference enables rapid rescoring of millions of docking poses in virtual screening — a 3D-CNN scores a pose in ~1 ms on a GPU vs. >1 s for FEP. The fundamental challenge is generalization across chemical space and protein families.
- **Key algorithms:** 3D-CNN on atomic density grids, equivariant graph neural networks (SchNet/DimeNet++), attention-based protein-ligand co-attention, diffusion-based co-folding (NeuralPLexer), Random Forest on PLEC/ECIF interaction fingerprints.
- **Datasets:** PDB-bind v2020 — 19,443 protein-ligand complexes with Kd/Ki (http://www.pdbbind.org.cn); CASF-2016 benchmark (http://www.pdbbind.org.cn/casf.php); ChEMBL activity data (https://www.ebi.ac.uk/chembl/); BindingDB — 2.8M measured binding affinities (https://www.bindingdb.org).
- **Starter repos/tools:** NeuralPLexer (https://github.com/zrqiao/NeuralPLexer) — state-specific co-folding with binding affinity, requires CUDA; GNINA (https://github.com/gnina/gnina) — CNN rescoring in docking pipeline; DiffDock (https://github.com/gcorso/DiffDock) — generative docking with affinity proxy; DeepChem (https://github.com/deepchem/deepchem) — includes AtomicConvolutions and MPNN-based scoring.
- **CUDA libraries & GPU pattern:** cuDNN for 3D-CNN layers; PyTorch Geometric CUDA kernels for equivariant message passing; FP16 mixed precision for throughput; GPU-parallel batch scoring for post-docking rescoring of millions of poses.

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
