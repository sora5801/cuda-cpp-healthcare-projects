# THEORY — 2.18 NMR Structure Refinement

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

### 2.18 NMR Structure Refinement 🟡 · Active R&D

- **Deep dive:** NMR structure determination requires satisfying distance restraints (NOE: <5 Å), dihedral angle restraints (J-couplings), and RDC (residual dipolar coupling) data via simulated annealing MD. GPU MD accelerates the restrained simulated annealing protocol, especially for large proteins where many restraint evaluations occur per timestep. GPU-accelerated CYANA/XPLOR-NIH can run hundreds of independent SA trajectories simultaneously — essential for ensemble NMR structure determination. Structure validation against chemical shift back-calculation is also GPU-acceleratable.
- **Key algorithms:** Simulated annealing MD with NOE/dihedral/RDC restraints, distance geometry embedding, torsion angle dynamics (CYANA), refinement against CSROSETTA chemical shifts, back-calculation of NMR observables.
- **Datasets:** BMRB — Biological Magnetic Resonance Bank (https://bmrb.io); PDB NMR-derived structures (https://www.rcsb.org); RECOORD — recalculated NMR structures (verify URL); CASD-NMR automated structure determination benchmarks (verify URL).
- **Starter repos/tools:** XPLOR-NIH (https://nmr.cit.nih.gov/xplor-nih/) — restrained MD for NMR with GPU support (via NAMD); CYANA (http://www.cyana.org) — torsion angle dynamics for NMR; AMBER NMR refinement (https://ambermd.org) — pmemd.cuda with NMR restraints; ARIA (http://aria.pasteur.fr) — automated NMR assignment and refinement.
- **CUDA libraries & GPU pattern:** Full GPU MD for restrained SA (pmemd.cuda); CUDA kernel for NOE energy and gradient computation; GPU-parallel independent SA replica array via MPI+CUDA; GPU chemical shift back-calculation via ShiftX2-GPU (verify URL).

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
