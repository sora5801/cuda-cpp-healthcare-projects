# THEORY — 2.15 Antibody Structure Prediction

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

### 2.15 Antibody Structure Prediction 🟡 · Active R&D

- **Deep dive:** Antibody structure prediction is specialized because the CDR-H3 loop is hypervariable and controls antigen specificity. Tools like IgFold, ABodyBuilder3, and IMGT-optimized AlphaFold2 models predict full antibody Fv region structures including flexible CDR loops. GPU inference enables high-throughput prediction for antibody library screening — thousands of sequences per GPU-hour. ABodyBuilder3 uses language model embeddings (ESM-2) and optimized GPU vectorization from OpenFold. Applications include antibody humanization, affinity maturation design, and developability assessment.
- **Key algorithms:** Attention-based CDR loop prediction, language model (ESM-2/IgLM) embeddings for antibody sequences, IMGT-numbered structure prediction, CDR-H3 loop sampling via diffusion, disulfide bond geometry constraints.
- **Datasets:** SAbDab — Structural Antibody Database (https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/); OAS (Observed Antibody Space) — 2B antibody sequences (https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/); CASP-Ab benchmarks; Thera-SAbDab — therapeutic antibody database (https://opig.stats.ox.ac.uk/webapps/newsabdab/therasabdab/).
- **Starter repos/tools:** IgFold (https://github.com/Graylab/IgFold) — fast antibody structure prediction on GPU; ABodyBuilder3 (verify GitHub URL) — GPU-optimized AF2 antibody model; AbNatiV (verify URL) — antibody naturalness scoring; AbDiffuser (verify URL) — antibody sequence+structure diffusion.
- **CUDA libraries & GPU pattern:** cuDNN multi-head attention for ESM-2 backbone; custom CDR attention CUDA kernels; FP16 inference with Flash attention; GPU-batched prediction for antibody library screening; PyTorch distributed for multi-GPU fine-tuning.

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
