# THEORY — 7.9 Real-Time Edge Inference for Medical Devices

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

### 7.9 Real-Time Edge Inference for Medical Devices 🟡 · Active R&D

- **Deep dive:** Deploys neural networks on embedded GPUs (NVIDIA Jetson, AMD Versal) or medical device SoCs for real-time inference — ECG arrhythmia detection, pulse oximetry anomaly, surgical robot vision, ultrasound B-mode AI. The challenge is matching model latency to physiological sampling rates (e.g., 500 Hz ECG requires <2 ms inference). TensorRT INT8 quantisation reduces model size 4× with minimal accuracy loss. Layer fusion fuses sequential convolutions, activations, and normalisation into single CUDA kernels, eliminating memory bandwidth bottlenecks. NVIDIA Jetson Orin delivers 275 TOPS at 60 W, enabling full clinical-grade CNN inference locally without cloud round-trips.
- **Key algorithms:** Post-training quantisation (INT8, INT4), knowledge distillation, neural architecture search for edge (MobileNetV3, EfficientDet-Lite), layer fusion, structured pruning, TensorRT engine optimisation, latency-aware NAS.
- **Datasets:**
  - PhysioNet Challenge datasets — ECG, SpO2, EEG for device validation (https://physionet.org/)
  - CAMUS cardiac ultrasound segmentation dataset (https://www.creatis.insa-lyon.fr/Challenge/camus/)
  - EyePACS retinal fundus — used for on-device DR screening validation (verify URL)
  - MIMIC-III Waveform Database — high-freq bedside monitor signals (https://physionet.org/content/mimicdb/)
- **Starter repos/tools:**
  - TensorRT (https://github.com/NVIDIA/TensorRT) — inference optimisation with INT8 calibration and layer fusion
  - NVIDIA Jetson SDK (https://developer.nvidia.com/embedded/jetpack) — Jetson-optimised libraries for edge GPU inference
  - MONAI Deploy (https://github.com/Project-MONAI/monai-deploy) — clinical AI deployment framework with TensorRT backend
  - OpenVINO (https://github.com/openvinotoolkit/openvino) — Intel edge inference toolkit for x86+iGPU devices
- **CUDA libraries & GPU pattern:** TensorRT INT8 for quantised inference, cuDNN for on-device convolutions, Triton Inference Server for multi-model serving; pattern: streaming inference pipeline with zero-copy pinned memory between sensor DMA and GPU.

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
