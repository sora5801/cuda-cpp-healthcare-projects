# THEORY — 14.14 Molecular Machine & Motor Protein Simulation

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

### 14.14 Molecular Machine & Motor Protein Simulation 🔴 · Frontier/Theoretical

- **Deep dive:** Molecular machines — kinesin walking on microtubules, ATP synthase rotating, ribosome translating — operate at nanoscale over microsecond-to-millisecond timescales that are far beyond conventional all-atom MD. GPU-accelerated enhanced sampling methods (metadynamics with PLUMED-CUDA, replica-exchange MD, HTMD adaptive sampling) extend the timescale window by orders of magnitude. Coarse-grained (MARTINI, CGMD) simulations on GPU model the full kinesin power stroke in minutes. The cryo-EM structural database provides high-resolution snapshots of machine conformations that seed GPU MD simulations of the mechanical cycle. Understanding motor protein dysfunction underpins treatments for neurodegeneration, cancer, and rare genetic diseases.
- **Key algorithms:** All-atom MD (GROMACS GPU, OpenMM), coarse-grained MD (MARTINI CGMD), metadynamics / funnel metadynamics with PLUMED-CUDA, replica-exchange MD (REMD), accelerated MD (aMD), elastic network model (ENM) for collective modes, Brownian ratchet mechanochemical models.
- **Datasets:** RCSB PDB motor protein structures — kinesin, dynein, myosin, ATP synthase (https://www.rcsb.org/); CHARMM-GUI membrane builder inputs (https://www.charmm-gui.org/); EMDB cryo-EM maps of conformational states (https://www.ebi.ac.uk/emdb/); GPCRdb for GPCR molecular machine models (https://gpcrdb.org/).
- **Starter repos/tools:** GROMACS (https://github.com/gromacs/gromacs) — GPU MD with CUDA/HIP, fastest production MD engine; OpenMM (https://github.com/openmm/openmm) — Python GPU MD with custom force plugins; PLUMED (https://github.com/plumed/plumed2) — GPU-compatible enhanced sampling (metadynamics) CV library; HTMD (https://github.com/Acellera/htmd) — GPU adaptive sampling for protein conformational exploration.
- **CUDA libraries & GPU pattern:** CUDA bonded/non-bonded force kernels (GROMACS native CUDA), cuFFT for PME long-range electrostatics, GPU neighbor-list Verlet scheme; pattern: cryo-EM structure → CHARMM-GUI parameterization → GPU REMD ensemble (N replicas × GPU) → PLUMED metadynamics bias application → free-energy surface reconstruction via WHAM.

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
