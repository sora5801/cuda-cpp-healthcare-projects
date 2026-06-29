# THEORY — 11.7 mRNA / Vaccine Sequence Design

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

### 11.7 mRNA / Vaccine Sequence Design 🟡 · Active R&D

- **Deep dive:** mRNA vaccine efficacy depends on optimal codon usage (for high ribosome translation), minimum free-energy (MFE) secondary structure (for stability), and 5'-UTR/3'-UTR element design. LinearDesign finds near-optimal MFE + CAI jointly in 11 minutes via dynamic programming on a lattice (analogous to CYK parsing), and GPU parallelization of the lattice can further accelerate multi-target vaccine design. VaxPress (2024) runs iterative codon optimization with customizable scoring functions including codon adaptation index, GC content, repeat minimization, and vaccine-specific immune stimulation features. Deep generative models (Nature Communications 2025) optimize codon sequences via GPU-trained VAEs, improving translation efficiency measurably in cell-free expression.
- **Key algorithms:** Minimum-free-energy (MFE) RNA folding (Zuker dynamic programming), codon adaptation index (CAI) optimization, LinearDesign lattice-DP algorithm, epitope prediction (MHC-I/II binding), RNA-structure gradient optimization, deep generative codon design (VAE/flow matching).
- **Datasets:** NCBI RefSeq CDS — validated coding sequences for codon usage tables (https://www.ncbi.nlm.nih.gov/refseq/); RNAcentral — non-coding RNA + UTR sequences (https://rnacentral.org/); VaxPress Test Suite — 100 vaccine antigens for benchmarking (https://github.com/ChangLabSNU/VaxPress); IEDB — immune epitope database for T/B cell responses (https://www.iedb.org/).
- **Starter repos/tools:** LinearDesign (https://github.com/LinearDesignSoftware/LinearDesign) — fast MFE+CAI co-optimization; VaxPress (https://github.com/ChangLabSNU/VaxPress) — codon optimizer with LinearDesign integration; VaxLab (https://github.com/ChangLabSNU/VaxLab) — integrated design platform; CodonBERT (verify URL, search "CodonBERT GitHub") — BERT-based codon optimization model (GPU inference).
- **CUDA libraries & GPU pattern:** cuDNN for Transformer-based codon sequence scoring (CodonBERT), CUDA dynamic-programming kernels for parallel MFE computation across sequence windows, Flash Attention for long mRNA sequence context; pattern: target antigen CDS → GPU LinearDesign DP → VaxPress iterative refinement on GPU → GPU epitope scoring → ranked candidates.

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
