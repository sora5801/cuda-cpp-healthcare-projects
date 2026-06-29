# THEORY — 2.33 Structure-Based Pharmacophore Modeling from MD Ensembles

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

### 2.33 Structure-Based Pharmacophore Modeling from MD Ensembles 🟡 · Active R&D

- **Deep dive:** Static pharmacophore models miss receptor flexibility; ensemble pharmacophore modeling derives features from MD trajectory frames, capturing induced-fit and cryptic-pocket binding geometries. GPU-accelerated MD generates the conformational ensemble; GPU-parallel feature extraction (H-bond donor/acceptor, hydrophobic contact maps) across millions of frames clusters into a consensus pharmacophore. The resulting ensemble pharmacophore is used for 3D similarity screening with GPU ROCS/FastROCS against billion-compound libraries, bridging MD insights with ultra-large-scale screening.
- **Key algorithms:** Dynamic pharmacophore feature extraction from MD, ensemble pharmacophore clustering (DBSCAN on feature vectors), 3D Gaussian overlap scoring (ROCS), pharmacophore SMARTS matching, common hits approach (CHA), water-displacement pharmacophore.
- **Datasets:** GPCRmd trajectory archive (https://gpcrmd.org); DUD-E actives/decoys for validation (https://dude.docking.org); PDB structures of target classes (https://www.rcsb.org); ZINC drug-like library for screening (https://zinc20.docking.org).
- **Starter repos/tools:** Pharmer (https://github.com/dkoes/pharmer) — pharmacophore screening tool; MDpocket (https://github.com/Discngine/fpocket) — pocket detection across MD trajectories; HTMD pharmacophore (https://github.com/Acellera/htmd) — ensemble pharmacophore from GPU MD; OpenEye ROCS (https://www.eyesopen.com/rocs) — GPU 3D shape+pharmacophore screening.
- **CUDA libraries & GPU pattern:** GPU Gaussian overlap for ROCS pharmacophore scoring; CUDA H-bond/hydrophobic feature extraction over MD frames; cuML DBSCAN for pharmacophore cluster detection; GPU batch pharmacophore matching over compound library.

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
