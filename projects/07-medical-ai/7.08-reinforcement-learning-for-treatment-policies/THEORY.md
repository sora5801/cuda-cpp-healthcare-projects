# THEORY — 7.8 Reinforcement Learning for Treatment Policies

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

### 7.8 Reinforcement Learning for Treatment Policies 🟡 · Active R&D

- **Deep dive:** Learns optimal dynamic treatment regimens — sepsis fluid and vasopressor dosing, mechanical ventilation settings, chemotherapy scheduling — from retrospective EHR trajectories using offline reinforcement learning. The GPU bottleneck is the batch Q-network or policy gradient updates across thousands of patient trajectories with hundreds of time steps each. Offline RL (Conservative Q-Learning, BEAR, TD3+BC) requires sampling large replay buffers and computing bootstrapped targets in parallel. Digital twin environments for safe exploration run population-level ODE simulations accelerated on GPU. Each policy evaluation step scores all actions for all patients simultaneously on GPU.
- **Key algorithms:** Conservative Q-Learning (CQL), Behaviour Constrained Policy Optimisation (BCPO), TD3+BC, Proximal Policy Optimisation (PPO) in simulation, Dueling DQN, Soft Actor-Critic (SAC), inverse RL, doubly-robust off-policy evaluation, OGSRL (Offline Guarded Safe RL).
- **Datasets:**
  - MIMIC-IV — ICU trajectories for sepsis, ventilation, and medication studies (https://physionet.org/content/mimiciv/)
  - eICU-CRD — multi-site ICU cohort for cross-hospital policy generalisation (https://eicu-crd.mit.edu/)
  - MIMIC-Sepsis benchmark (https://arxiv.org/abs/2510.24500) — curated sepsis trajectory benchmark from MIMIC
  - AmsterdamUMCdb — 23k ICU patients, open-access (https://amsterdammedicaldatascience.nl/)
- **Starter repos/tools:**
  - d3rlpy (https://github.com/takuseno/d3rlpy) — offline RL library with CUDA-accelerated Q-learning (PyTorch)
  - MIMIC-Extract (https://github.com/MLforHealth/MIMIC_Extract) — standardised MIMIC-III/IV feature extraction for RL
  - AI Clinician (https://github.com/matthieukomorowski/AI_Clinician) — seminal offline RL sepsis treatment repo
  - HealthGym (https://github.com/healthylaife/healthgym) — clinical offline RL environments built on MIMIC data
- **CUDA libraries & GPU pattern:** cuDNN for policy/Q-networks, cuBLAS for experience replay batch matmuls, custom CUDA kernels for parallelised Bellman backup over large replay buffers; pattern: GPU replay buffer sampling with pinned memory for fast CPU→GPU transfer.

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
