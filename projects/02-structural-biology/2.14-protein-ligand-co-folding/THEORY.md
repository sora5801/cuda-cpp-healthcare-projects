# THEORY — 2.14 Protein-Ligand Co-Folding

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

### 2.14 Protein-Ligand Co-Folding 🟡 · Active R&D

- **Deep dive:** Co-folding models simultaneously predict protein structure and ligand binding pose in a single forward pass, bypassing separate docking steps. Boltz-1 and AlphaFold3 accept ligand SMILES and protein sequence as joint inputs to a diffusion model conditioned on molecular features. GPU inference generates protein-ligand complex structures at near-FEP accuracy for pose prediction in minutes per complex. The GPU bottleneck is the diffusion sampling loop (50–200 denoising steps), each requiring a full attention forward pass over the joint protein-ligand token sequence.
- **Key algorithms:** Joint protein-ligand diffusion (DDPM on 3D positions), conditional atom-type and geometry generation, atom-level self-attention with periodic boundary handling, confidence (pLDDT/iPAE) scoring, cross-attention between protein and ligand tokens.
- **Datasets:** PoseBusters benchmark — 428 recently released PDB complexes (https://github.com/maabuu/posebusters); PDB-bind v2020 (http://www.pdbbind.org.cn); Astex Diverse Set — 85 drug-like ligand complex structures (verify URL); CASF cross-docking benchmarks (http://www.pdbbind.org.cn/casf.php).
- **Starter repos/tools:** Boltz-1 (https://github.com/jwohlwend/boltz) — GPU co-folding of protein-ligand-nucleic acid complexes; NeuralPLexer3 (https://github.com/zrqiao/NeuralPLexer) — state-specific co-folding with CUDA; AlphaFold3 (https://github.com/google-deepmind/alphafold3) — official AF3 with ligand support; DiffDock (https://github.com/gcorso/DiffDock) — diffusion docking without co-folding (complementary).
- **CUDA libraries & GPU pattern:** Flash attention (FlashAttention2) for long joint sequences; cuDNN transformer blocks; GPU diffusion denoising loop with CUDA noise schedules; FP16/BF16 precision; multi-GPU model parallelism for large complexes.

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
