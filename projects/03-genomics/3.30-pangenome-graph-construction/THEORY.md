# THEORY — 3.30 Pangenome Graph Construction

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

### 3.30 Pangenome Graph Construction 🟡 · Active R&D
- **Deep dive:** Building a pangenome variation graph from dozens to thousands of genome assemblies requires all-to-all pairwise alignment (seqwish induces the graph from alignment PAF) and progressive normalisation (smoothxg). At the scale of the HPRC 94-haplotype human pangenome, wfmash all-to-all alignment is the dominant cost. GPU wavefront alignment (WFA) directly accelerates this: the wavefront DP's diagonal-based expansion is anti-diagonal parallel, mapping naturally to GPU threads. ODGI's GPU layout (57.3× speedup over multi-core CPU) demonstrates that force-directed node positioning—a core graph visualisation and ordering step—is highly GPU-amenable via particle-based physics simulation.
- **Key algorithms:** Wavefront alignment (WFA) for all-to-all pairwise seeding; seqwish overlap-to-graph induction; smoothxg POA-based block normalisation; ODGI force-directed layout (SGD / simulated annealing); GFA graph compaction; r-index for graph BWT.
- **Datasets:** HPRC year-1 assemblies — 94 haplotypes, human pangenome (https://humanpangenome.org/); Ensembl non-human pangenome data (https://www.ensembl.org/); Vertebrate Genomes Project assemblies (https://vertebrategenomesproject.org/); NCBI RefSeq complete genomes for bacterial pangenomes (https://ftp.ncbi.nlm.nih.gov/refseq/).
- **Starter repos/tools:** PGGB (https://github.com/pangenome/pggb) — pangenome graph builder pipeline; ODGI with GPU layout (https://github.com/pangenome/odgi) — GPU-accelerated graph layout and operations; wfmash (https://github.com/waveygang/wfmash) — WFA-based all-to-all aligner; vg (https://github.com/vgteam/vg) — comprehensive graph alignment toolkit.
- **CUDA libraries & GPU pattern:** Custom WFA CUDA kernels (anti-diagonal wavefront expansion); GPU-resident pairwise alignment matrix per genome pair; CUDA force-directed layout with node-force parallelism; thrust for wavefront front management; multi-GPU for large-scale all-to-all (N² / num_GPUs pairs per GPU).

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
