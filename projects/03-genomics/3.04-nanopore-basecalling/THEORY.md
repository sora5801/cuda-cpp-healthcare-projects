# THEORY — 3.4 Nanopore Basecalling

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

### 3.4 Nanopore Basecalling 🟢 · Established
- **Deep dive:** Nanopore basecalling translates raw ionic-current signal samples (electrical squiggles) from the sequencer into DNA/RNA base sequences. Oxford Nanopore's Dorado uses a recurrent neural network (transformer + CTC decoder in current "SUP" models) trained to map signal windows to base probabilities. The bottleneck is the RNN/transformer inference over millions of signal events per run hour, a perfect GPU workload: batched matrix multiplications across reads mapped to thousands of CUDA cores. Dorado achieves up to 30% speed improvement for HAC models on Ampere/Ada/Blackwell GPUs over previous versions and scales linearly across multiple GPUs. The GPU also powers modified base (methylation) calling simultaneously during basecalling.
- **Key algorithms:** Bidirectional LSTM / Transformer encoder; Connectionist Temporal Classification (CTC) decoding; beam search decoding; adaptive banded event alignment (f5c); Modified base (5mC, 6mA) classification heads.
- **Datasets:** ONT Open Dataset (PromethION human WGS) — available via SRA / ENA (https://www.ncbi.nlm.nih.gov/sra); R9.4.1 and R10.4.1 benchmark datasets released by ONT (https://github.com/GoekeLab/awesome-nanopore); GIAB ONT ultra-long reads — NA12878/HG002 nanopore truth sets (https://www.nist.gov/programs-projects/genome-bottle); ENA Project PRJNA594038 — public multi-species ONT data (https://www.ebi.ac.uk/ena).
- **Starter repos/tools:** Dorado (https://github.com/nanoporetech/dorado) — ONT's official GPU basecaller, multi-GPU, CUDA-optimised, supports MOD calling; f5c (https://github.com/hasindu2008/f5c) — CUDA-accelerated methylation calling and event alignment; awesome-nanopore (https://github.com/GoekeLab/awesome-nanopore) — curated tool index including GPU-enabled callers; Guppy — legacy ONT CUDA basecaller, GPU-only, superseded by Dorado.
- **CUDA libraries & GPU pattern:** cuDNN (RNN/transformer), TensorRT (inference optimisation), cuBLAS (GEMM), CUDA streams (pipelining signal batches); multi-GPU with NVLink/NCCL; persistent thread blocks for stateful RNN across signal chunks.

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
