# THEORY — 7.2 Drug-Target Interaction Prediction (GNN)

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

### 7.2 Drug-Target Interaction Prediction (GNN) 🟡 · Active R&D

- **Deep dive:** Predicts whether a small molecule (drug) will bind to a protein target and estimates binding affinity (Kd/Ki) or binary interaction labels. Molecular graphs have irregular topology, so graph neural message-passing aggregates neighbour features in parallel across thousands of candidate pairs simultaneously on GPU. Protein sequences can be encoded via transformer attention (ESM-2, ProtTrans) whose quadratic attention is accelerated by Flash Attention on CUDA. The bottleneck is the cross-attention between drug graph embeddings and protein sequence embeddings over large virtual screening libraries (millions of compounds), which maps to batched sparse matrix operations. GPU throughput determines how many candidates can be scored per day in drug discovery pipelines.
- **Key algorithms:** Message Passing Neural Networks (MPNN), Graph Attention Networks (GAT), Directed Message Passing (DMPNN), transformer cross-attention, contrastive DTI objectives, Graph Isomorphism Networks (GIN), graph-level pooling, Bayesian hyperparameter optimisation.
- **Datasets:**
  - BindingDB — ~2.9 million measured binding affinities for drug-target pairs (https://www.bindingdb.org/)
  - ChEMBL — curated bioactivity database with >20M activity records (https://www.ebi.ac.uk/chembl/)
  - Davis Kinase Dataset — kinase inhibitor affinities for 442 kinases × 68 drugs (verify URL)
  - KIBA — integrated kinase inhibitor bioactivity benchmark (verify URL)
- **Starter repos/tools:**
  - DeepPurpose (https://github.com/kexinhuang12345/DeepPurpose) — 15 drug/protein encoders, 50+ architectures for DTI
  - TorchDrug (https://github.com/DeepGraphLearning/torchdrug) — GPU-accelerated graph learning library for drug discovery
  - DGL-LifeSci (https://github.com/awslabs/dgl-lifesci) — DGL-based molecular GNN toolkit with CUDA-backed sparse ops
  - DTA-GNN (https://github.com/lennylv/DTA-GNN) — toolkit for target-specific DTA dataset construction and GNN training
- **CUDA libraries & GPU pattern:** DGL/PyG sparse adjacency ops on GPU, Flash Attention 2 for protein encoders, cuDNN for MLP heads; pattern: heterogeneous data parallelism (drug batch × protein batch), optional multi-GPU model parallelism for large protein encoders.

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
