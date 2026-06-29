# THEORY — 12.5 Real-Time Sequencing Analysis / Adaptive Sampling

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

### 12.5 Real-Time Sequencing Analysis / Adaptive Sampling 🟡 · Active R&D
- **Deep dive:** Oxford Nanopore adaptive sampling (ReadUntil API) allows the sequencer to reject reads in real time (within 200 ms per read) based on a computational decision—requiring GPU basecalling and alignment to complete in under ~100 ms per read chunk. The pipeline: raw signal → GPU basecalling (Dorado, HAC model) → GPU seed-extension to reference → accept/reject decision → signal to sequencer. GPU processing is not optional; CPU pipelines are too slow for the 200 ms window. This enables on-target enrichment without library preparation: unwanted chromosomal regions are skipped by reversing the voltage to eject the DNA strand.
- **Key algorithms:** GPU CTC basecalling (Dorado transformer); approximate hash seed alignment (minimap2 GPU); streaming input buffer management; read-until decision tree; pore blocking prediction; real-time sequence classification (pathogen typing).
- **Datasets:** ONT open datasets with ReadUntil metadata (https://github.com/GoekeLab/awesome-nanopore); NCBI SRA real-time sequencing runs (https://www.ncbi.nlm.nih.gov/sra); ENA clinical nanopore studies (https://www.ebi.ac.uk/ena); Oxford Nanopore public data portal (https://labs.epi2me.io/dataindex/).
- **Starter repos/tools:** Dorado (https://github.com/nanoporetech/dorado) — GPU basecaller with low-latency streaming mode; ReadFish (https://github.com/looselab/readfish) — ReadUntil adaptive sampling controller; Icarust (https://github.com/LooseLab/Icarust) — real-time nanopore simulator for pipeline testing; MinKNOW (ONT proprietary) — sequencer control with GPU basecalling integration.
- **CUDA libraries & GPU pattern:** TensorRT for ultra-low-latency RNN inference; CUDA streams for overlapping signal decode and alignment; persistent GPU kernel for continuous signal ingestion; GPU ring buffer for streaming POD5 signal; multi-GPU for PromethION multi-flow-cell setups.

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
