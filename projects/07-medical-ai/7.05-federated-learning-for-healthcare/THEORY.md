# THEORY — 7.5 Federated Learning for Healthcare

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

### 7.5 Federated Learning for Healthcare 🟡 · Active R&D

- **Deep dive:** Trains a single global model across multiple hospitals without sharing raw patient data: each site trains on local data and sends only model gradients or weights to a central aggregator. The GPU bottleneck on each client is identical to standard local training; additional communication cost arises from the aggregation step. NVIDIA FLARE orchestrates GPU-based local training with differential privacy noise injection and secure aggregation. Heterogeneous GPU fleets across hospitals (V100 at one site, A100 at another) require adaptive batch sizing and mixed-precision logic. The primary research challenge is handling statistical data heterogeneity (non-IID distributions) while maintaining convergence.
- **Key algorithms:** FedAvg, FedProx, SCAFFOLD, FedNova, personalised federated learning, differential privacy (Gaussian mechanism, moments accountant), secure aggregation with homomorphic encryption, communication compression (gradient sparsification, quantisation).
- **Datasets:**
  - TCGA (The Cancer Genome Atlas) — multi-institutional genomics + histopathology (https://www.cancer.gov/tcga)
  - MIMIC-IV — EHR data used in federated simulation across synthetic partitions (https://physionet.org/content/mimiciv/)
  - NIH Chest X-ray Dataset — 112,120 chest X-rays for FL benchmarks (https://nihcc.app.box.com/v/ChestXray-NIHCC)
  - Medical Segmentation Decathlon — multi-task dataset used in FL challenges (http://medicaldecathlon.com/)
- **Starter repos/tools:**
  - NVIDIA FLARE (https://github.com/NVIDIA/NVFlare) — production-grade federated learning SDK with GPU-native training loops
  - OpenFL (https://github.com/securefederatedai/openfl) — Intel/Linux Foundation FL framework supporting PyTorch/TF on GPU
  - Flower (https://github.com/adap/flower) — lightweight, framework-agnostic FL with GPU support
  - PySyft (https://github.com/OpenMined/PySyft) — privacy-preserving FL with differential privacy on GPU
- **CUDA libraries & GPU pattern:** cuDNN for local model training, NCCL for efficient intra-site multi-GPU; pattern: data parallelism within site, synchronous or asynchronous gradient aggregation between sites via secure channels.

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
