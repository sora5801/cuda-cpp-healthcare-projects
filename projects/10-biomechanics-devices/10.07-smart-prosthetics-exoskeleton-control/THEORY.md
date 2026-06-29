# THEORY — 10.7 Smart Prosthetics & Exoskeleton Control

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

### 10.7 Smart Prosthetics & Exoskeleton Control 🟡 · Active R&D

- **Deep dive:** Myoelectric prosthetics and powered exoskeletons decode surface EMG or EEG in real time to predict user intent, then execute low-latency torque commands. GPU acceleration runs deep CNNs and recurrent networks for intent classification in under 5 ms, meeting the ~50 ms end-to-end control loop budget. Reinforcement-learning-trained controllers for exoskeleton gait assistance require millions of simulated steps during training (parallelized in GPU physics engines like IsaacGym), then deploy on edge GPUs. Impedance control and admittance control loops for compliant interaction simulate full body-device co-dynamics, with contact forces between limb and exoskeleton computed via GPU FSI.
- **Key algorithms:** CNN / LSTM / Transformer EMG intent classification, model-predictive control (MPC), impedance/admittance control, reinforcement learning (PPO/SAC in IsaacGym), Kalman/extended-Kalman observer for joint state estimation, proportional myoelectric control, adaptive gain scheduling.
- **Datasets:** NinaPro DB5 — 10-DOF hand gestures, surface EMG + IMU from 53 subjects (http://ninapro.hevs.ch/); PhysioNet Lower Limb Prosthetics — transtibial amputee locomotion (https://physionet.org/); BCI Competition IV — motor imagery EEG for upper-limb control (https://www.bbci.de/competition/iv/); exo-H3 IMU dataset — powered exoskeleton kinematics (verify URL via IEEE DataPort).
- **Starter repos/tools:** NVIDIA IsaacGym (https://developer.nvidia.com/isaac-gym) — GPU-parallel RL training for robotic/exoskeleton control; legged_gym (https://github.com/leggedrobotics/legged_gym) — GPU-parallel locomotion RL on IsaacGym; Biopatrec (https://github.com/g-guo/biopatrec) — EMG pattern recognition benchmark platform; BioSig (https://biosig.sourceforge.net/) — open biosignal toolbox with EMG classifiers (CPU, GPU-extensible).
- **CUDA libraries & GPU pattern:** cuDNN (CNN inference for EMG/EEG classification), CUDA kernels for batch EMG windowing and feature extraction, IsaacGym GPU physics for RL training; pattern: 4096 parallel simulated human-exoskeleton environments in IsaacGym → policy gradient update on GPU → policy distillation → edge GPU (Jetson) deployment.

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
