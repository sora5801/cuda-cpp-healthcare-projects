# THEORY — 11.1 Protein Engineering / Directed Evolution In Silico

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

### 11.1 Protein Engineering / Directed Evolution In Silico 🟡 · Active R&D

- **Deep dive:** Machine-learning-guided directed evolution replaces physical screening with GPU-accelerated fitness prediction, scoring millions of sequence variants per second using protein language models (ESM-2) or structure-based Rosetta energy functions. EVOLVEpro (Science 2025) demonstrated rapid in silico directed evolution by proposing and filtering variants with GPU-deployed LLM embeddings. Batched GPU inference over combinatorial mutation libraries (10⁸–10¹² sequences) identifies beneficial mutations orders of magnitude faster than laboratory selection. The key parallelism is embarrassingly parallel: each sequence variant scores independently.
- **Key algorithms:** Protein language model (ESM-2) embeddings + fitness regression, directed evolution with Bayesian optimization (GP or Bayesian neural network), structure-based ΔΔG prediction (Rosetta fast-relax, FoldX), zero-shot fitness scoring via masked-language-model log-odds, gradient-based sequence optimization via differentiable fitness surrogate.
- **Datasets:** ProteinGym Substitution Benchmarks — 250+ deep mutational scanning (DMS) datasets across protein families (https://proteingym.org/); Envision (PABP, UBE4B) fitness landscapes; Fluorescent Protein Dataset (GFP) — 56 K variants with fluorescence labels (https://github.com/fhalab/FLIP); FLIP Benchmarks — standardized fitness landscape benchmarks (https://github.com/J-SNACKKB/FLIP).
- **Starter repos/tools:** ESM (https://github.com/facebookresearch/esm) — Meta FAIR ESM-2 + ESMFold GPU protein language model; EVOLVEpro (verify URL at bakerlab.org or GitHub) — in silico directed evolution pipeline; ProteinMPNN (https://github.com/dauparas/ProteinMPNN) — GPU sequence design from backbone; Fitness-Prediction-Benchmark (https://github.com/J-SNACKKB/FLIP) — DMS benchmark datasets and baseline models.
- **CUDA libraries & GPU pattern:** cuDNN for Transformer forward pass over batched sequences, Flash Attention for memory-efficient long-sequence attention, mixed-precision (BF16) for throughput; pattern: encode 10⁶ variants as token batch → GPU LLM forward pass → fitness score vector → Bayesian acquisition function selects next round → iterate.

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
