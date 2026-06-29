# THEORY — 1.11 QSAR / Property Prediction

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

### 1.11 QSAR / Property Prediction 🟢 · Established

- **Deep dive:** Quantitative structure-activity relationship (QSAR) models predict biological activity from molecular descriptors or learned representations. Modern approaches use message-passing neural networks (MPNNs) over molecular graphs, enabling GPU-batched training on millions of labeled datapoints. The bottleneck shifts from feature computation to batch normalization and message aggregation over irregular graph structures — handled by PyTorch Geometric or DGL with CUDA backends. GPU-accelerated QSAR models at pharmaceutical companies screen hundreds of millions of virtual compounds per hour for ADMET and activity filters.
- **Key algorithms:** Directed message-passing (D-MPNN / Chemprop), graph convolutional networks (GCN), graph attention networks (GAT), transformer on molecular graphs (Uni-Mol), random forest / XGBoost on Morgan fingerprints, uncertainty quantification (ensemble, MCDropout).
- **Datasets:** MoleculeNet — curated ML benchmark for 17+ molecular datasets (https://moleculenet.org); ChEMBL bioactivity data (https://www.ebi.ac.uk/chembl/); TDC (Therapeutics Data Commons) — 66 tasks for drug discovery ML (https://tdcommons.ai); PCBA (PubChem BioAssay) — 128 bioassays on 440k compounds (https://moleculenet.org).
- **Starter repos/tools:** Chemprop (https://github.com/chemprop/chemprop) — D-MPNN for molecular property prediction, GPU training; Uni-Mol (https://github.com/deepmodeling/Uni-Mol) — 3D molecular transformer pre-trained on 209M conformers; DeepChem (https://github.com/deepchem/deepchem) — broad GPU-accelerated ML chemistry toolkit; DGL-LifeSci (https://github.com/awslabs/dgl-lifesci) — graph neural networks for life science on GPU.
- **CUDA libraries & GPU pattern:** PyTorch Geometric CUDA sparse tensor ops for graph batching; cuDNN for feedforward layers; FP16 mixed precision; GPU-accelerated descriptor generation via RDKit CUDA extensions (verify URL).

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
