# THEORY — 2.1 Protein Structure Prediction Inference (AlphaFold-class)

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

### 2.1 Protein Structure Prediction Inference (AlphaFold-class) 🟢 · Established

- **Deep dive:** AlphaFold2 and its successors (RoseTTAFold, ESMFold, OpenFold, Boltz-1, AlphaFold3) predict atomic-resolution 3D protein structures from amino acid sequences using deep learning. The Evoformer stack processes multiple sequence alignments (MSAs) and pair representations through stacked self-attention and triangle-multiplicative update layers — each requiring enormous GPU memory (an A100 40GB handles ~5000 residues for AF2). GPU inference is mandatory: predicting a 500-residue protein takes ~5 minutes on GPU vs. ~12 hours on CPU. ESMFold bypasses MSA entirely, using a 15B-parameter language model for 10–60× faster prediction.
- **Key algorithms:** Evoformer (MSA row/column attention + triangle updates), Structure Module (invariant point attention, IPA), recycling iterations, template attention, diffusion-based structure generation (AF3), confidence scoring (pLDDT, PAE).
- **Datasets:** AlphaFold Database — 200M+ predicted structures (https://alphafold.ebi.ac.uk/); RCSB PDB — 227k+ experimental structures (https://www.rcsb.org); UniProt/UniRef90 MSA databases (https://www.uniprot.org); CAMEO/CASP15 structure prediction benchmarks (https://www.cameo3d.org).
- **Starter repos/tools:** AlphaFold2 (https://github.com/google-deepmind/alphafold) — official DeepMind implementation; OpenFold (https://github.com/aqlaboratory/openfold) — trainable GPU-friendly PyTorch AF2; ESMFold (https://github.com/facebookresearch/esm) — MSA-free language model structure prediction; Boltz-1 (https://github.com/jwohlwend/boltz) — fully open AF3-level biomolecular complex prediction.
- **CUDA libraries & GPU pattern:** cuDNN multi-head attention for Evoformer; custom CUDA triangle update kernels; FP16/BF16 mixed precision; flash attention (FlashAttention2) for memory-efficient MSA attention; multi-GPU model parallelism for large complexes.

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
