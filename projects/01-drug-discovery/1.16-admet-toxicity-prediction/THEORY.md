# THEORY — 1.16 ADMET / Toxicity Prediction

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

### 1.16 ADMET / Toxicity Prediction 🟢 · Established

- **Deep dive:** Absorption, Distribution, Metabolism, Excretion, and Toxicity (ADMET) properties gate entry into clinical trials; predicting them computationally early in discovery eliminates costly failures. GPU-trained GNN/MPNN models (Chemprop-based) can screen 100M virtual compounds for ADMET in hours. The ADMET-AI platform (2024) uses a Chemprop-RDKit ensemble achieving best-in-class speed. Multi-task learning on heterogeneous assay data (LogP, hERG, Caco-2, microsomal clearance, Ames mutagenicity) benefits from GPU parallelism across tasks and molecules simultaneously.
- **Key algorithms:** Directed message-passing (D-MPNN), multi-task learning, uncertainty quantification (conformal prediction, evidential learning), Tox21 endpoint models, quantum-chemical descriptor augmentation.
- **Datasets:** Tox21 — 12 toxicity endpoints, 8k compounds (https://tripod.nih.gov/tox21/); TDC ADMET benchmark group (https://tdcommons.ai/benchmark/admet_group/overview/); ClinTox — FDA-approved and failed drugs (https://moleculenet.org); DILI (drug-induced liver injury) databases (verify URL).
- **Starter repos/tools:** Chemprop (https://github.com/chemprop/chemprop) — D-MPNN backbone for ADMET models; ADMET-AI (https://github.com/swansonk14/admet_ai) — GPU-accelerated ADMET platform; DeepChem (https://github.com/deepchem/deepchem) — includes Tox21 models; pkCSM (https://biosig.lab.uq.edu.au/pkcsm/) — web server using graph signatures (verify GPU support).
- **CUDA libraries & GPU pattern:** PyTorch Geometric CUDA sparse ops; cuDNN for feedforward/attention layers; multi-task GPU loss aggregation; FP16 training; batched RDKit fingerprint generation.

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
