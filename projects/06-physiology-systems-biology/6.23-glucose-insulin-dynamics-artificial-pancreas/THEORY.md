# THEORY — 6.23 Glucose-Insulin Dynamics & Artificial Pancreas

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

### 6.23 Glucose-Insulin Dynamics & Artificial Pancreas 🟡 · Active R&D
- **Deep dive:** Type 1 diabetes management via a closed-loop artificial pancreas requires real-time simulation of glucose-insulin dynamics (Bergman minimal model, UVA/Padova T1D simulator) for controller design, in-silico trial, and reinforcement learning (RL) training. GPU acceleration enables parallel virtual patient cohort simulation for RL policy optimization and Monte Carlo variability analysis. The UVA/Padova simulator has been FDA-accepted for in-silico clinical trials.
- **Key algorithms:** Bergman minimal model (3-compartment ODE), UVA/Padova T1D simulator (13-compartment ODE), PID and model-predictive control (MPC), deep RL (PPO, SAC) for insulin dosing policy, glucose meal appearance (Gastric Emptying model), Kalman filter for CGM noise filtering.
- **Datasets:** OhioT1DM dataset — 12-week CGM + insulin data for 12 T1D subjects (https://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html); JAEB CGMS datasets (https://public.jaeb.org); simglucose simulator virtual patient population (https://github.com/jxx123/simglucose); DirecNet CGM datasets (https://public.jaeb.org/direcnet).
- **Starter repos/tools:** simglucose (https://github.com/jxx123/simglucose) — Python UVA/Padova T1D simulator, gym environment for RL; GluCoEnv (https://github.com/chirathyh/GluCoEnv) — GPU-accelerated glucose control RL environment (PyTorch); G2P2C (https://github.com/RL4H/G2P2C) — RL artificial pancreas; OpenAPS oref0 (https://github.com/openaps/oref0) — open-source reference algorithm.
- **CUDA libraries & GPU pattern:** Batched ODE integration (cusolve / custom RK4 kernel) for ensemble of virtual patients; cuRAND for meal disturbance sampling; PyTorch GPU for RL policy network training; pattern: embarrassingly parallel virtual patient simulation—one CUDA thread per patient per time step.

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
