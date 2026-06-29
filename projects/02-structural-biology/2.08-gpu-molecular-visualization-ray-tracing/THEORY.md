# THEORY — 2.8 GPU Molecular Visualization & Ray Tracing

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

### 2.8 GPU Molecular Visualization & Ray Tracing 🟢 · Established

- **Deep dive:** Interactive visualization of molecular dynamics trajectories, cryo-EM density maps, and protein structures requires real-time rendering of millions of atoms with surface representations (VDW spheres, solvent-accessible surface, cartoon ribbons). VMD uses CUDA for GPU ray tracing (Tachyon/RTX OptiX), generating photorealistic images of molecular systems with ambient occlusion and shadows. Interactive manipulation of multi-million-atom systems at >30 fps is achievable on RTX GPUs. NVIDIA IndeX provides volume rendering of cryo-EM maps directly on GPU.
- **Key algorithms:** GPU ray tracing (OptiX/Embree), CUDA ambient occlusion, isosurface extraction (marching cubes on GPU), molecular surface triangulation, GPU-accelerated MSMS algorithm, volume rendering via GPU compositing, instanced rendering for periodic systems.
- **Datasets:** EMDB cryo-EM maps (https://www.ebi.ac.uk/emdb/); RCSB PDB molecular structures (https://www.rcsb.org); GPCRmd MD trajectories (https://gpcrmd.org); CHARMM-GUI example systems (https://charmm-gui.org).
- **Starter repos/tools:** VMD (https://www.ks.uiuc.edu/Research/vmd/) — GPU-accelerated molecular visualization with CUDA/OptiX ray tracing; PyMOL (https://github.com/schrodinger/pymol-open-source) — GPU-rendered molecular graphics; OVITO (https://www.ovito.org) — GPU-enabled scientific visualization for MD; Mol* (https://github.com/molstar/molstar) — WebGL-accelerated online viewer.
- **CUDA libraries & GPU pattern:** NVIDIA OptiX for hardware ray tracing; custom CUDA marching cubes for isosurface; CUDA sphere/cylinder instanced rendering; GPU-parallel surface normal computation; cuFFT for density map smoothing.

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
