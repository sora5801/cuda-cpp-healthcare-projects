# THEORY — 13.19 Antibody Pharmacokinetics & FcRn-Mediated Recycling

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

### 13.19 Antibody Pharmacokinetics & FcRn-Mediated Recycling 🔴 · Frontier/Theoretical

- **Deep dive:** Models the complex PK of monoclonal antibodies and bispecifics, which are dominated by FcRn-mediated endosomal recycling, target-mediated drug disposition, and antigen sink effects. The multi-compartment antibody PK model (plasma, interstitial, endosome, target tissue) coupled with FcRn binding dynamics is a stiff ODE system with widely separated time scales (hours vs. weeks). Population simulation of thousands of virtual patients with variable FcRn expression, antigen expression, and body composition requires GPU-parallel stiff ODE integration. Antibody engineering to optimise FcRn affinity and pH-dependent binding can be virtually screened at scale on GPU.
- **Key algorithms:** Two-compartment antibody model with FcRn recycling submodel (Dhanarajan-Meibohm), TMDD with high-affinity target binding, pH-dependent FcRn binding kinetics (endosomal pH 6.0 vs. plasma pH 7.4), neonatal clearance model, multi-target bispecific PK (dual TMDD), stiff LSODA/CVODE integration, population NLME for biologics.
- **Datasets:**
  - Published mAb PK datasets — Phase I first-in-human PK from IND/NDA submissions (verify via FDA label search)
  - BioModels antibody models (https://www.ebi.ac.uk/biomodels/)
  - Open Systems Pharmacology mAb PBPK library (https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library)
  - DrugBank biologic PK data (https://www.drugbank.com/)
- **Starter repos/tools:**
  - Pumas (https://pumas.ai/) — GPU biologics PK modelling in Julia
  - PK-Sim mAb models (https://github.com/Open-Systems-Pharmacology/PK-Sim) — PBPK for antibodies with FcRn
  - mrgsolve (https://github.com/metrumresearchgroup/mrgsolve) — R-based simulation of PKPD ODEs, parallelisable with OpenMP
  - nvQSP (https://github.com/NVIDIA-Digital-Bio/nvQSP) — GPU stiff ODE solver for antibody PK virtual populations
- **CUDA libraries & GPU pattern:** Custom CUDA CVODE stiff ODE kernels with FcRn endosomal binding, cuBLAS for Jacobian LU factorisation, cuRAND for virtual patient FcRn expression sampling; pattern: one CUDA block per virtual patient, pH-dependent binding kinetics in thread-local registers.

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
