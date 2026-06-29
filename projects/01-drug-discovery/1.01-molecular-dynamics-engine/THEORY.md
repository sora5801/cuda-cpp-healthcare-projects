# THEORY — 1.1 Molecular Dynamics Engine

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

### 1.1 Molecular Dynamics Engine 🟢 · Established

- **Deep dive:** Classical MD simulates the time evolution of every atom in a biomolecular system by integrating Newton's equations of motion using empirical force fields (AMBER, CHARMM, GROMOS). Each timestep requires evaluating bonded interactions (bonds, angles, dihedrals) and non-bonded interactions (Lennard-Jones + electrostatics) for millions of atom pairs. GPUs accelerate the embarrassingly parallel pairwise force evaluation, reducing a day of CPU work to minutes on modern A100-class cards. The critical bottleneck — neighbor-list construction and PME reciprocal-space summation — maps cleanly onto CUDA threadblocks. Multi-GPU scaling via domain decomposition allows systems of 10–100 million atoms to be simulated in production.
- **Key algorithms:** Verlet/leapfrog integrator, LINCS/SHAKE bond constraint solvers, Particle-Mesh Ewald (PME) electrostatics, Lennard-Jones cutoff with long-range dispersion correction, Berendsen/Parrinello-Rahman barostat, velocity rescaling/Nosé-Hoover thermostat.
- **Datasets:** CHARMM36m force-field parameter set — comprehensive parameters for proteins, lipids, nucleic acids and carbohydrates (https://mackerell.umaryland.edu/charmm_ff.shtml); AMBER ff19SB — protein force field with improved backbone torsion potentials (https://ambermd.org); GPCRmd database — curated MD trajectories of GPCR proteins (https://gpcrmd.org); MoDEL — molecular dynamics extended library of protein simulations (https://mmb.irbbarcelona.org/MoDEL/).
- **Starter repos/tools:** GROMACS (https://github.com/gromacs/gromacs) — production-grade GPU-accelerated MD engine with CUDA/HIP/SYCL backends; OpenMM (https://github.com/openmm/openmm) — Python-scriptable MD toolkit with CUDA, OpenCL, and CPU platforms; NAMD (https://www.ks.uiuc.edu/Research/namd/) — scalable MD with multi-GPU support via CUDA; AMBER pmemd.cuda (https://ambermd.org/GPUSupport.php) — highly optimized GPU MD engine for AMBER force fields.
- **CUDA libraries & GPU pattern:** cuFFT for PME reciprocal sum, custom CUDA kernels for pairwise force evaluation, thrust for sorted neighbor list, NCCL for multi-GPU halo exchange; pattern is data-parallel threadblocks over atom pairs with shared-memory reductions.

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
