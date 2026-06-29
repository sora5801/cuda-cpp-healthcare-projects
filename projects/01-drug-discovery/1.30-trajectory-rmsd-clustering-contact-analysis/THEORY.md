# THEORY — 1.30 Trajectory RMSD, Clustering & Contact Analysis

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

### 1.30 Trajectory RMSD, Clustering & Contact Analysis 🟢 · Established

- **Deep dive:** Post-MD analysis of multi-microsecond trajectories generates terabytes of coordinate data requiring GPU-accelerated analytics. RMSD calculation requires aligning every frame to a reference (Kabsch algorithm: SVD of 3×3 matrices — trivially parallelized over frames). Pairwise RMSD for clustering requires O(N²) comparisons of millions of frames. H-bond network analysis and contact map generation are similarly parallelizable. MDTraj and cuML enable GPU-accelerated trajectory analysis with RAPIDS. The bottleneck is I/O bandwidth from trajectory files.
- **Key algorithms:** Kabsch RMSD algorithm (SVD), GROMOS/DBSCAN/k-medoids clustering, contact map calculation (distance cutoff), H-bond donor-acceptor angle+distance criteria, radial distribution function (RDF), NMR order parameter S².
- **Datasets:** MDCATH trajectory dataset (https://huggingface.co/datasets/compsciencelab/mdcath); PDB trajectory depositions; GPCRmd (https://gpcrmd.org); MDDB (https://www.mddbr.eu) — molecular dynamics database.
- **Starter repos/tools:** MDTraj (https://github.com/mdtraj/mdtraj) — GPU-accelerated RMSD and trajectory analysis; RAPIDS cuML (https://github.com/rapidsai/cuml) — GPU clustering for MSM construction; MDAnalysis (https://github.com/MDAnalysis/mdanalysis) — trajectory analysis with GPU support; HTMD (https://github.com/Acellera/htmd) — GPU-accelerated adaptive MD analysis.
- **CUDA libraries & GPU pattern:** Custom CUDA kernels for batched 3×3 SVD (Kabsch rotation); GPU pairwise distance matrix via cuBLAS (outer product formulation); atomic contact map via GPU distance thresholding; RAPIDS cuDF for trajectory frame I/O.

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
