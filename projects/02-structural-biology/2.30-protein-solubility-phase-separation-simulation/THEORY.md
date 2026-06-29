# THEORY — 2.30 Protein Solubility & Phase Separation Simulation

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

### 2.30 Protein Solubility & Phase Separation Simulation 🔴 · Frontier/Theoretical

- **Deep dive:** Liquid-liquid phase separation (LLPS) of intrinsically disordered proteins (IDPs) and RNA-binding proteins underlies formation of biomolecular condensates (stress granules, P-bodies, nucleolus). Simulating LLPS requires system sizes of millions of CG atoms over millisecond timescales — only accessible with GPU CG-MD. FUS, TDP-43, and hnRNPA1 condensate-forming domains have been simulated with MARTINI or HPS (hydrophobicity scale) CG models on GPU. Phase diagrams are computed by running multiple concentration conditions simultaneously. Applications include predicting condensate-forming mutations and designing condensate-disrupting drugs.
- **Key algorithms:** Coarse-grained HPS/Kim-Hummer IDP model, MARTINI IDR parameters, Gibbs ensemble MC for phase coexistence, density functional theory for phase diagram, metadynamics order parameter for condensate formation, finite-size scaling for phase boundary.
- **Datasets:** FuzDB — fuzzy protein complex database (https://fuzdb.org); PhaSePro — proteins undergoing LLPS (https://phasepro.elte.hu); DisProt — intrinsically disordered proteins (https://disprot.org); human proteome LLPS predictor datasets (catGRANULE, PScore).
- **Starter repos/tools:** LAMMPS + HPS model (https://github.com/lammps/lammps) — GPU IDP LLPS simulation; OpenMM HPS (https://github.com/openmm/openmm) — Python IDP CG MD; CALVADOS 2 (https://github.com/KULL-Centre/CALVADOS) — residue-level IDP model for LLPS; GROMACS MARTINI IDR (https://github.com/gromacs/gromacs) — GPU CG LLPS.
- **CUDA libraries & GPU pattern:** GPU CG-MD for multi-million-bead IDP system; CUDA kernel for simplified HPS non-bonded interactions; GPU-parallel concentration ensemble (multiple boxes); GPU-accelerated order parameter clustering for phase detection.

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
