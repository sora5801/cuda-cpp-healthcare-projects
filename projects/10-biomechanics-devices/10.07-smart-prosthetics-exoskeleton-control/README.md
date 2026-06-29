# 10.7 — Smart Prosthetics & Exoskeleton Control

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biomechanics%2C%20Biomedical%20Devices%20%26%20Surgery-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 10: Biomechanics, Biomedical Devices & Surgery · Catalog ID `10.7`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

<!-- =======================================================================
     SCAFFOLD STATUS: this README was stamped from the catalog. The prose
     fields below (Deep dive / Algorithms / Datasets / Prior art) are filled
     in from the catalog. Sections marked TODO(impl)/TODO(theory) must be
     completed by the project author before this project is "done"
     (see CLAUDE.md §4.1 and tools/verify_project.py).
     ======================================================================= -->

## Summary

TODO(impl): One paragraph, plain language — what this project does and why a
learner should care. (Seed from the deep dive below.)

## What this computes & why the GPU helps

Myoelectric prosthetics and powered exoskeletons decode surface EMG or EEG in real time to predict user intent, then execute low-latency torque commands. GPU acceleration runs deep CNNs and recurrent networks for intent classification in under 5 ms, meeting the ~50 ms end-to-end control loop budget. Reinforcement-learning-trained controllers for exoskeleton gait assistance require millions of simulated steps during training (parallelized in GPU physics engines like IsaacGym), then deploy on edge GPUs. Impedance control and admittance control loops for compliant interaction simulate full body-device co-dynamics, with contact forces between limb and exoskeleton computed via GPU FSI.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

CNN / LSTM / Transformer EMG intent classification, model-predictive control (MPC), impedance/admittance control, reinforcement learning (PPO/SAC in IsaacGym), Kalman/extended-Kalman observer for joint state estimation, proportional myoelectric control, adaptive gain scheduling.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/smart-prosthetics-exoskeleton-control.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/smart-prosthetics-exoskeleton-control.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\smart-prosthetics-exoskeleton-control.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the result, shows the
GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/` — a tiny, offline input so the demo runs
  with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented, idempotent).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: NinaPro DB5 — 10-DOF hand gestures, surface EMG + IMU from 53 subjects (http://ninapro.hevs.ch/); PhysioNet Lower Limb Prosthetics — transtibial amputee locomotion (https://physionet.org/); BCI Competition IV — motor imagery EEG for upper-limb control (https://www.bbci.de/competition/iv/); exo-H3 IMU dataset — powered exoskeleton kinematics (verify URL via IEEE DataPort).

## Expected output

Success looks like `demo/expected_output.txt`. The program computes the result on
both the **GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`)
and asserts they agree within the documented tolerance — that agreement is the
correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
2. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
3. [`src/kernels.cu`](src/kernels.cu) — the kernel(s) and host wrapper.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline.
5. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

NVIDIA IsaacGym (https://developer.nvidia.com/isaac-gym) — GPU-parallel RL training for robotic/exoskeleton control; legged_gym (https://github.com/leggedrobotics/legged_gym) — GPU-parallel locomotion RL on IsaacGym; Biopatrec (https://github.com/g-guo/biopatrec) — EMG pattern recognition benchmark platform; BioSig (https://biosig.sourceforge.net/) — open biosignal toolbox with EMG classifiers (CPU, GPU-extensible).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN (CNN inference for EMG/EEG classification), CUDA kernels for batch EMG windowing and feature extraction, IsaacGym GPU physics for RL training; pattern: 4096 parallel simulated human-exoskeleton environments in IsaacGym → policy gradient update on GPU → policy distillation → edge GPU (Jetson) deployment. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
