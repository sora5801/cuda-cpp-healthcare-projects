# THEORY — 14.10 GPU-Accelerated Bayesian Inference Engine for Biomedicine

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

### 14.10 GPU-Accelerated Bayesian Inference Engine for Biomedicine 🟡 · Active R&D

- **Deep dive:** Bayesian inference over high-dimensional biomedical models (pharmacokinetic, genetic, epidemiological) requires Markov chain Monte Carlo (MCMC) or variational inference (VI) that is historically slow. GPU-accelerated Hamiltonian Monte Carlo (HMC/NUTS) in NumPyro or PyMC-JAX achieves 10–100× speedup over CPU Stan, enabling inference in population PKPD models with 10⁴ parameters and >10⁶ observations. GPU batch parallelism runs independent MCMC chains simultaneously, and GPU-accelerated gradients via JAX/autograd make HMC feasible for complex ODEs. Clinical trial simulation (tens of thousands of virtual patients) is a key use case.
- **Key algorithms:** Hamiltonian Monte Carlo (HMC) + No-U-Turn Sampler (NUTS), variational inference (ADVI, normalizing flows), sequential Monte Carlo (SMC), population PKPD (NONMEM-equivalent), Gaussian process inference, integrated nested Laplace approximation (INLA).
- **Datasets:** NONMEM Pharmacokinetic Reference Dataset (Holford NHG, verify URL); UK Biobank phenome-wide association studies (https://www.ukbiobank.ac.uk/); OpenFDA Drug Adverse Event database (https://open.fda.gov/apis/drug/event/); CDISC SDTM clinical trial datasets (verify URL via cdisc.org).
- **Starter repos/tools:** NumPyro (https://github.com/pyro-ppl/numpyro) — GPU HMC/NUTS via JAX; PyMC (https://github.com/pymc-devs/pymc) — probabilistic programming with JAX/GPU backend; BlackJAX (https://github.com/blackjax-devs/blackjax) — GPU MCMC kernels in JAX; Stan (https://github.com/stan-dev/stan) — reference Bayesian inference (CPU; GPU via GPU-compatible backend research).
- **CUDA libraries & GPU pattern:** JAX XLA GPU compilation for HMC gradient computation, cuBLAS for covariance matrix operations in GP inference, cuFFT for spectral MCMC methods; pattern: prior + likelihood specification in NumPyro → GPU JIT-compiled HMC kernel → parallel chains on GPU → posterior diagnostics (R-hat, ESS) → posterior predictive check.

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
