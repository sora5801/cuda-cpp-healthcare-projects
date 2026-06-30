# Demo — 3.18 Protein Language Model Inference

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/protein_sample.txt` (a 24-residue synthetic peptide
   through one multi-head self-attention block: `d_model=32, heads=4, d_head=8`).
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`): the
   output embeddings, the head-0 attention map, and the discrete "most-attended
   residue" readout must all agree. Prints a clear `PASS`/`FAIL`.
4. **Time** the kernels (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Canonical output

See [`expected_output.txt`](expected_output.txt). For each residue the program
prints its output-embedding **L2 norm** and the residue **head 0 attends to most**.
Two things are worth noticing, and both are real teaching points:

- **Identical residues get identical summaries.** Every `A` (positions 3, 6) has
  norm `0.342107`; every `K` (1, 7, 15) has `0.328776`. With no positional
  encoding, a residue's projection depends only on its amino-acid embedding, so
  equal residues are interchangeable — exactly why real PLMs *add* positional
  information (here, rotary embeddings — see THEORY).
- **A few residues act as hubs.** Many queries' head-0 attention peaks at residue
  `17` (`H`), and `RESULT: PASS` confirms the GPU's `softmax(QKᵀ/√d)·V` matches
  the CPU's to within `~1.5e-8` (well under the `1e-4` tolerance).

> The sequence and all model weights are **synthetic** (the weights are generated
> from an integer hash in `src/attention_math.h`). This demonstrates the
> self-attention *computation*, not a real protein prediction.
