# THEORY — 14.1 Whole-Cell Simulation

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

### 14.1 Whole-Cell Simulation 🔴 · Frontier/Theoretical

- **Deep dive:** Whole-cell simulation aspires to mechanistically model every gene, mRNA, protein, metabolite, and organelle in a single bacterium or yeast cell simultaneously. The scale challenge is staggering: E. coli has ~4,300 genes and ~1.5 M ribosomes; a complete stochastic reaction-diffusion simulation at molecular resolution would require centuries on a single CPU. GPU acceleration of spatial SSA (Gillespie/tau-leaping) over a discretized cell volume enables partial whole-cell models (gene expression + metabolism) to run in tractable time. The STEPS simulator (parallel, GPU-accelerated) handles reaction-diffusion on tetrahedral meshes representing subcellular geometry. Achieving true whole-cell simulation likely requires exascale GPU clusters.
- **Key algorithms:** Spatial Gillespie SSA (Next Subvolume Method / ISSA), tau-leaping with error control, next-reaction method (NRM), multiscale hybrid: ODE for deterministic fast species + SSA for rare events, GPU-parallel lattice microbes (LM) algorithm, whole-cell model composition (FBA + transcription/translation + signaling).
- **Datasets:** Mycoplasma genitalium whole-cell model (Karr et al. Cell 2012) parameters (https://simtk.org/projects/wc_models); E. coli K-12 transcriptomics (GEO GSE2198 and related); BioModels Database whole-cell models (https://www.ebi.ac.uk/biomodels/); JCVI Syn3A minimal genome datasets (https://www.jcvi.org/research/first-minimal-synthetic-bacterial-cell).
- **Starter repos/tools:** STEPS (https://github.com/CNS-OIST/STEPS) — GPU-accelerated stochastic spatial reaction-diffusion in tetrahedral meshes; Lattice Microbes (LM) (https://github.com/Luthey-Schulten-Lab/Lattice_Microbes) — GPU spatial stochastic simulator for E. coli; Smoldyn (https://github.com/ssandrews/Smoldyn) — off-lattice particle-based RD simulator (multi-GPU); WholeCellKB (https://github.com/CovertLab/WholeCell) — Karr whole-cell model framework.
- **CUDA libraries & GPU pattern:** CUDA kernels for parallel subvolume SSA reaction firing, cuRAND for per-subvolume random streams, NCCL for multi-GPU spatial domain decomposition; pattern: cell volume partitioned into tetrahedral subvolumes on GPU → parallel SSA firing per subvolume → diffusive transfer between subvolumes via CUDA inter-thread communication → global species count aggregation → repeat at nanosecond timescale.

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
