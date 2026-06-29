# THEORY — 3.21 Structural Variant (SV) Calling

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

### 3.21 Structural Variant (SV) Calling 🟡 · Active R&D
- **Deep dive:** Structural variants (deletions, insertions, inversions, translocations ≥50 bp) are detected by read-support signatures: split reads, discordant pairs, and assembly-based breakpoint realignment. GPU acceleration applies at two points: (1) rapid re-alignment of split-read candidates using banded SW to pinpoint breakpoints precisely, and (2) batched deep learning inference (convolutional models on pileup images) to genotype and filter SVs. Sniffles2 uses a fast clustering algorithm for ONT/HiFi; pbsv uses local realignment. GPU-accelerated genotyping (similar to DeepVariant's image-based approach) is an emerging direction for SV filtering at population scale.
- **Key algorithms:** Split-read alignment and breakpoint clustering; discordant pair signature scoring; local assembly with miniasm/hifiasm at breakpoints; convolutional image-based genotyping (DeepSV style); SV merging across samples (SURVIVOR); genotype likelihood calculation.
- **Datasets:** GiaB SV benchmark (HG002) — gold-standard deletion/insertion/inversion calls (https://www.nist.gov/programs-projects/genome-bottle); PacBio SV benchmark (https://github.com/PacificBiosciences/sv-benchmark); 1000 Genomes SV catalog (https://www.internationalgenome.org/data); ENCODE long-read SV studies (https://www.encodeproject.org/).
- **Starter repos/tools:** Sniffles2 (https://github.com/fritzsedlazeck/Sniffles) — fast ONT/HiFi SV caller; PBSV (https://github.com/PacificBiosciences/pbsv) — PacBio SV caller; cuteSV (https://github.com/tjiangHIT/cuteSV) — clustering-based SV caller; NGSEP (https://github.com/NGSEP/NGSEPcore) — variant calling suite with GPU-amenable CNN scoring.
- **CUDA libraries & GPU pattern:** Banded SW CUDA kernels for breakpoint realignment; cuDNN CNN for SV image genotyping; batched pileup image inference; thrust for read cluster sorting; multi-GPU for population-scale SV genotyping.

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
