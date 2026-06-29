# 7.8 — Reinforcement Learning for Treatment Policies

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20AI%20%26%20Clinical%20Deep%20Learning-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 7: Medical AI & Clinical Deep Learning · Catalog ID `7.8`
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

Learns optimal dynamic treatment regimens — sepsis fluid and vasopressor dosing, mechanical ventilation settings, chemotherapy scheduling — from retrospective EHR trajectories using offline reinforcement learning. The GPU bottleneck is the batch Q-network or policy gradient updates across thousands of patient trajectories with hundreds of time steps each. Offline RL (Conservative Q-Learning, BEAR, TD3+BC) requires sampling large replay buffers and computing bootstrapped targets in parallel. Digital twin environments for safe exploration run population-level ODE simulations accelerated on GPU. Each policy evaluation step scores all actions for all patients simultaneously on GPU.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Conservative Q-Learning (CQL), Behaviour Constrained Policy Optimisation (BCPO), TD3+BC, Proximal Policy Optimisation (PPO) in simulation, Dueling DQN, Soft Actor-Critic (SAC), inverse RL, doubly-robust off-policy evaluation, OGSRL (Offline Guarded Safe RL).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/reinforcement-learning-for-treatment-policies.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/reinforcement-learning-for-treatment-policies.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\reinforcement-learning-for-treatment-policies.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: MIMIC-IV — ICU trajectories for sepsis, ventilation, and medication studies (https://physionet.org/content/mimiciv/) eICU-CRD — multi-site ICU cohort for cross-hospital policy generalisation (https://eicu-crd.mit.edu/) MIMIC-Sepsis benchmark (https://arxiv.org/abs/2510.24500) — curated sepsis trajectory benchmark from MIMIC AmsterdamUMCdb — 23k ICU patients, open-access (https://amsterdammedicaldatascience.nl/)

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

d3rlpy (https://github.com/takuseno/d3rlpy) — offline RL library with CUDA-accelerated Q-learning (PyTorch) MIMIC-Extract (https://github.com/MLforHealth/MIMIC_Extract) — standardised MIMIC-III/IV feature extraction for RL AI Clinician (https://github.com/matthieukomorowski/AI_Clinician) — seminal offline RL sepsis treatment repo HealthGym (https://github.com/healthylaife/healthgym) — clinical offline RL environments built on MIMIC data

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN for policy/Q-networks, cuBLAS for experience replay batch matmuls, custom CUDA kernels for parallelised Bellman backup over large replay buffers; pattern: GPU replay buffer sampling with pinned memory for fast CPU→GPU transfer. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
