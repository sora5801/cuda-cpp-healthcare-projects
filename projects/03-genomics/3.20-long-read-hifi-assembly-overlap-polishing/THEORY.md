# THEORY — 3.20 Long-Read HiFi Assembly Overlap & Polishing

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

### 3.20 Long-Read HiFi Assembly Overlap & Polishing 🟡 · Active R&D
- **Deep dive:** PacBio HiFi reads (10–25 kb, >99.5% accuracy) enable near-perfect de novo assemblies but the all-vs-all read overlap step—finding which reads share sequence—is computationally prohibitive: N=20 M reads requires O(N²) comparisons naively. GPU parallelism accelerates the minimiser-based seed look-up and seed chain extension across read pairs. The Darwin read overlapper GPU implementation achieved 109× speedup by storing the minimiser hash table in GPU global memory and resolving seed chains in parallel CUDA blocks. Post-overlap polishing (racon, medaka) is similarly accelerated by GPU POA and RNN inference kernels.
- **Key algorithms:** Minimiser hashing for all-vs-all overlap seeding; sparse chain DP for seed-chain scoring; partial-order alignment (POA) for consensus polishing; string graph simplification and unitig generation; haplotype phasing via heterozygous marker threading.
- **Datasets:** PacBio SMRT Human WGS (HG002/HG003/HG004 trio) (https://www.ncbi.nlm.nih.gov/sra); Vertebrate Genomes Project PacBio HiFi assemblies (https://vertebrategenomesproject.org/); GenomeArk HiFi datasets (https://genomeark.github.io/); CHM13 T2T HiFi reads (https://github.com/marbl/CHM13).
- **Starter repos/tools:** Darwin GPU overlapper (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7495891/) — 109× GPU speedup for PacBio overlap; hifiasm (https://github.com/chhylp123/hifiasm) — state-of-the-art HiFi assembler; racon-GPU (https://github.com/NVIDIA-Genomics-Research/racon-gpu) — GPU consensus polishing; Medaka (https://github.com/nanoporetech/medaka) — RNN-based polishing with GPU inference.
- **CUDA libraries & GPU pattern:** GPU-resident minimiser hash map; custom seed-chain CUDA kernels; POA DP in shared memory per thread block; cuDNN RNN for Medaka polishing; CUDA streams pipelining I/O and compute.

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
