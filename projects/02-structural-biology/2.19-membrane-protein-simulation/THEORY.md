# THEORY — 2.19 Membrane Protein Simulation

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

### 2.19 Membrane Protein Simulation 🟢 · Established

- **Deep dive:** Membrane proteins (GPCRs, ion channels, transporters, integrins) are embedded in lipid bilayers and represent >50% of current drug targets. Explicit membrane MD requires building asymmetric bilayers with physiological lipid compositions and running microsecond simulations to sample conformational changes. CHARMM-GUI automates system building; GPU GROMACS/NAMD runs production simulations. Key challenges include equilibrating the membrane (~100 ns), maintaining bilayer asymmetry, and capturing slow conformational transitions. GPU-accelerated CG-MARTINI pre-equilibration (1–10 μs) followed by backmapping to all-atom provides a common pipeline.
- **Key algorithms:** CHARMM36 lipid force field, POPE/POPC/cholesterol bilayer assembly, semi-isotropic barostat (NPT-xy coupling), PME for charged bilayer system, CG-to-AA backmapping, k-means clustering of ion channel gate states.
- **Datasets:** MemProtMD — 3133 membrane proteins in lipid bilayers (https://memprotmd.bioch.ox.ac.uk); GPCRdb — GPCR structures and MD data (https://gpcrdb.org); CGMD Platform benchmark systems (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7765266/); OPM — orientations of proteins in membranes (https://opm.phar.umich.edu).
- **Starter repos/tools:** CHARMM-GUI Membrane Builder (https://charmm-gui.org) — automated bilayer + protein setup; GROMACS (https://github.com/gromacs/gromacs) — GPU membrane protein MD; HTMD (https://github.com/Acellera/htmd) — GPU-accelerated membrane protein pipeline; packmol-memgen (https://github.com/memembranes) — AMBER membrane system builder.
- **CUDA libraries & GPU pattern:** GPU semi-isotropic barostat coupling; cuFFT for PME with charged bilayer; custom CUDA PME corrections for 2D slab geometry; multi-GPU domain decomposition along z-axis; GPU neighbor list for heterogeneous lipid-protein system.

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
