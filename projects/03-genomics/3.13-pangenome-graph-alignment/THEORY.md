# THEORY — 3.13 Pangenome Graph Alignment

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

### 3.13 Pangenome Graph Alignment 🟡 · Active R&D
- **Deep dive:** Pangenome graphs encode the genomic variation of an entire population as a sequence graph (GFA format) rather than a single linear reference; aligning reads to this graph involves generalised DP over a DAG of paths rather than a 1D reference. The vg toolkit's graph alignment applies a generalised Smith-Waterman on the graph DAG, which is harder to parallelise than linear alignment due to irregular memory access. A 2024 SC paper demonstrated GPU-accelerated pangenome layout achieving 57.3× speedup over multi-core CPU for the ODGI layout algorithm by mapping node-force computations to GPU threads. Graph seeding via GBWT/r-index also benefits from parallelised BWT operations.
- **Key algorithms:** Generalised DAG DP alignment; GBWTgraph / r-index graph BWT; pangenome graph layout (force-directed, GPU particles); ODGI path sorting and sorting optimisation; seqwish overlap-to-graph induction; wfmash wavefront alignment for all-to-all seeding.
- **Datasets:** Human Pangenome Reference Consortium (HPRC) — 94 haplotype-resolved assemblies (https://humanpangenome.org/); 1000 Genomes Project GVCFs — variant calls for graph construction (https://www.internationalgenome.org/data); Ensembl Pangenome — multi-species graphs (https://www.ensembl.org/); PGGB tutorial data (https://github.com/pangenome/pggb).
- **Starter repos/tools:** vg (https://github.com/vgteam/vg) — comprehensive variation graph toolkit; PGGB (https://github.com/pangenome/pggb) — Pangenome Graph Builder pipeline; ODGI (https://github.com/pangenome/odgi) — GPU layout algorithms; Rapid GPU-based pangenome layout paper (https://www.csl.cornell.edu/~zhiruz/pdfs/pangenome-layout-sc2024.pdf) — 57× speedup reference.
- **CUDA libraries & GPU pattern:** Custom CUDA force-directed layout kernels (Barnes-Hut approximation on GPU); parallel graph BFS for BWT construction; thrust for node-position sort; cuSPARSE for sparse adjacency matrix traversal; one CUDA thread per node-force computation.

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
