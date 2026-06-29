# THEORY — 1.2 Particle-Mesh Ewald Electrostatics

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

### 1.2 Particle-Mesh Ewald Electrostatics 🟢 · Established

- **Deep dive:** Long-range electrostatics in periodic MD systems cannot be truncated without severe artifacts; PME splits the Coulomb sum into a short-range real-space part (evaluated with cutoff) and a smooth long-range reciprocal-space part evaluated on a 3D grid via FFT. The GPU acceleration opportunity is two-fold: the charge spreading (particle-to-mesh) and force interpolation (mesh-to-particle) steps are data-parallel over atoms, while the 3D FFT is handled by cuFFT. PME scales as O(N log N) and dominates walltime for large biological systems. Achieving double-precision accuracy at float throughput is the main engineering challenge.
- **Key algorithms:** Ewald summation, B-spline charge interpolation (order 4–6), 3D FFT on GPU, real-space erfc damping, smooth PME (SPME), Particle-Particle Particle-Mesh (P3M).
- **Datasets:** CHARMM-GUI solvation benchmark sets — pre-built periodic protein-water boxes (https://charmm-gui.org); D. E. Shaw Research Anton trajectories — ms-scale trajectory archives (available via DE Shaw); ion channel benchmark systems (MemProtMD, https://memprotmd.bioch.ox.ac.uk).
- **Starter repos/tools:** GROMACS CUDA PME (https://github.com/gromacs/gromacs) — reference GPU PME implementation; NAMD GPU PME (https://www.ks.uiuc.edu/Research/namd/) — tiled domain-decomposed PME; OpenMM PME plugin (https://github.com/openmm/openmm) — Python-accessible PME with mixed-precision; cuFFT (https://developer.nvidia.com/cufft) — NVIDIA's FFT library used internally by all above.
- **CUDA libraries & GPU pattern:** cuFFT for 3D FFT; custom CUDA kernels for B-spline charge spreading (atom-parallel) and gradient interpolation; shared-memory tiling to minimize global memory traffic; atomics for scatter-add accumulation on the charge grid.

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
