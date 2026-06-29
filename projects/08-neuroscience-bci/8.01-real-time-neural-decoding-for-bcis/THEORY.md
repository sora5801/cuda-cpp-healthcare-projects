# THEORY — 8.1 Real-Time Neural Decoding for BCIs

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

### 8.1 Real-Time Neural Decoding for BCIs 🟡 · Active R&D
- **Deep dive:** Brain-computer interfaces decode motor intent, speech, or cognitive state from simultaneous recordings of 100–1 000+ neural channels at 30 kHz sampling. The decoding pipeline—bandpass filtering, spike detection, feature extraction, Kalman/population vector decode, output command generation—must complete within 5–50 ms to feel natural to the user. GPU acceleration allows running deep neural decoder networks (1D-CNN, transformer, WaveNet) directly in the decode loop without sacrificing latency through CUDA stream pipelining.
- **Key algorithms:** Kalman filter (linear decoder), population vector algorithm (PVA), Wiener filter, linear discriminant analysis (LDA), point-process filter, recurrent neural networks (GRU/LSTM), convolutional temporal decoder, optimal linear estimator (OLE), variational autoencoder latent space decoding.
- **Datasets:** BrainGate clinical trial data (https://www.braingate.org — access via collaboration); DANDI Archive intracortical array datasets (https://dandiarchive.org); Allen Brain Observatory Neuropixels data (https://portal.brain-map.org); NLB (Neural Latents Benchmark) — standardized BCI decode benchmarks (https://neurallatents.github.io).
- **Starter repos/tools:** BrainFlow (https://github.com/brainflow-dev/brainflow) — unified BCI SDK with real-time GPU-compatible data streaming; OpenBCI GUI (https://github.com/OpenBCI/OpenBCI_GUI) — open-source BCI hardware + software; NLB challenge tools (https://github.com/neurallatents/nlb_tools) — neural latents benchmark evaluation; NDT2/FALCON BCI decode benchmark (verify URL on neurallatents.github.io).
- **CUDA libraries & GPU pattern:** cuBLAS for real-time matrix multiply in Kalman predict/update; TensorRT for inference-optimized deep decoder; CUDA streams for pipelined acquire→decode→output with <5 ms latency; pattern: producer-consumer stream pipeline with pinned host memory for zero-copy data ingestion from acquisition hardware.

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
