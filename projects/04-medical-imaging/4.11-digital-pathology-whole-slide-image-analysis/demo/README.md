# Demo — 4.11 Digital Pathology / Whole-Slide Image Analysis

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/slide_sample.txt` — one synthetic slide of 64 tiles
   × 8 features, with 6 planted "tumor" tiles (label 1).
3. **Verify** that the GPU attention-MIL forward pass matches the CPU reference —
   attention weights, pooled embedding, and slide probability all within `1e-9`
   (the fixed-point pooling makes the embedding and probability match *exactly*).
4. **Report** the pooled slide embedding, the tumor probability, the top-attention
   tile, a top-5 attention ranking, and the `@0.5` slide call.

stdout (the result) is deterministic and diffed against
[`expected_output.txt`](expected_output.txt); the timing and raw error magnitudes
are on stderr only.

## Canonical output

See [`expected_output.txt`](expected_output.txt). The interesting thing to look at:

- **Attention concentrates on the tumor tiles.** The top-5 attention tiles are
  `#13, #33, #43, #3, #23` — exactly five of the six planted tumor tiles (at
  indices 3, 13, 23, 33, 43, 53), each with weight ~0.16 versus the uniform
  `1/64 ≈ 0.0156`. The model "looks at" the diagnostic tiles, ignoring background.
- **The slide is called `TUMOR`** (probability ≈ 0.556 ≥ 0.5), matching the
  ground-truth label 1. Regenerate a benign slide with
  `python scripts/make_synthetic.py --tumor-frac 0` and the call flips to `benign`
  (probability ≈ 0.08) with a flat, uninformative attention map.
- **`RESULT: PASS`** means the GPU and CPU produced the same attention, embedding,
  and probability within tolerance. The stderr `[verify]` line shows the raw
  differences (`max embed diff = 0`, `prob diff = 0` — the fixed-point trick at
  work; `max attn diff ~1e-16` — device vs host transcendentals).

> The data is a **synthetic** feature bag, not real histology — a demonstration of
> GPU attention-MIL, **not** a clinical analysis.
