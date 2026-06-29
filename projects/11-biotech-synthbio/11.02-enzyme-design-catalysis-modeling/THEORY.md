# THEORY — 11.2 Enzyme Design & Catalysis Modeling

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

### 11.2 Enzyme Design & Catalysis Modeling 🟡 · Active R&D

- **Deep dive:** Computational enzyme design requires evaluating active-site geometry, transition-state stabilization, and substrate binding simultaneously. GPU-accelerated QM/MM (quantum mechanics / molecular mechanics) couples a DFT or semi-empirical QM region around the catalytic residues with a classical MM region of the full enzyme, enabling thousands of candidate enzyme structures to be ranked. Rosetta enzyme design generates theozyme scaffolds and then repacks surrounding residues on GPU. AlphaFold-2 structure prediction + ProteinMPNN sequence design creates novel enzyme candidates at scale. De novo enzyme design for non-natural reactions (Diels-Alder, retro-aldol) has been demonstrated computationally; GPU acceleration is the bottleneck to scaling to large combinatorial searches.
- **Key algorithms:** Rosetta enzyme design (RIF docking, match/scaffold search), QM/MM (ONIOM, pDynamo), transition-state theory rate prediction, directed evolution fitness landscape modeling, SE(3)-equivariant active-site design (BindCraft/RFdiffusion), Monte Carlo backrub for enzyme refinement.
- **Datasets:** BRENDA Enzyme Database — kinetics, substrates, organisms (https://www.brenda-enzymes.org/); SABIO-RK — enzyme kinetic parameters (https://sabiork.h-its.org/); UniProt/SwissProt enzyme entries (https://www.uniprot.org/); M-CSA Mechanism and Catalytic Site Atlas (https://www.ebi.ac.uk/thornton-srv/m-csa/).
- **Starter repos/tools:** RFdiffusion (https://github.com/RosettaCommons/RFdiffusion) — diffusion-based active-site design; PyRosetta (https://github.com/RosettaCommons/pyrosetta) — GPU-compatible Rosetta Python bindings; GROMACS (https://github.com/gromacs/gromacs) — GPU QM/MM enzyme MD via ORCA/CP2K coupling; DeepMind AlphaFold2 (https://github.com/google-deepmind/alphafold) — structure prediction for enzyme scaffold validation.
- **CUDA libraries & GPU pattern:** cuDNN for Rosetta energy term neural-network surrogate, CUDA ONIOM QM/MM kernels via GROMACS GPU engine, cuFFT for periodic electrostatics (PME); pattern: RFdiffusion generates active-site scaffold on GPU → ProteinMPNN designs sequence → GPU MD relaxation → GPU QM/MM ΔG‡ evaluation → rank and select.

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
