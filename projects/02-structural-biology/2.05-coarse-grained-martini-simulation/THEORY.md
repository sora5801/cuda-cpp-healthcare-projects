# THEORY — 2.5 Coarse-Grained / MARTINI Simulation

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

### 2.5 Coarse-Grained / MARTINI Simulation 🟢 · Established

- **Deep dive:** Coarse-grained (CG) force fields like MARTINI map ~4 heavy atoms to a single interaction site, enabling microsecond-to-millisecond simulations of large membrane systems (entire plasma membranes with 63 lipid species, viral capsids, ribosomes). MARTINI3 CG-MD runs in GROMACS with full GPU acceleration, gaining ~100-fold timescale extension over all-atom MD. Membrane protein insertion, lipid scrambling, and vesicle formation are accessible only at CG resolution. The GPU bottleneck is non-bonded CG pair interactions; the coarser grid makes PME and neighbor lists faster than all-atom.
- **Key algorithms:** MARTINI3 force field, Lennard-Jones + shifted electrostatics for CG beads, elastic network overlay (Gō-MARTINI) for protein secondary structure, CG-to-AA backmapping, PME for long-range CG electrostatics.
- **Datasets:** CHARMM-GUI MARTINI membrane builder outputs (https://charmm-gui.org); lipid parameter database (https://cgmartini.nl); membrane-active peptide aggregation benchmarks; EMDB viral capsid reference maps for validation.
- **Starter repos/tools:** GROMACS+MARTINI3 (https://github.com/gromacs/gromacs) — production GPU CG-MD; MARTINI force field files (https://cgmartini.nl) — official parameter repository; TS2CG (https://github.com/weria-pezeshkian/TS2CG) — triangulated surface to CG membrane builder; insane.py (https://github.com/Tsjerk/Insane) — membrane assembly tool for MARTINI.
- **CUDA libraries & GPU pattern:** CUDA kernels for CG non-bonded pair evaluation; cuFFT for CG PME; neighbor list construction with larger cutoffs (1.1–1.2 nm vs 0.9 nm AA); GPU memory efficiency improved by reduced atom count (~4× vs AA).

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
