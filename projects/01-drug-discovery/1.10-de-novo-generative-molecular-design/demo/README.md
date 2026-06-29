# Demo — 1.10 De Novo Generative Molecular Design

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/smiles_corpus_sample.txt` corpus.
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`):
   every generated molecule's score and length must match **exactly** (the RNG,
   sampling loop, and scorer are shared host/device code, so molecule *i* is
   bit-identical on both). Prints a clear `PASS`/`FAIL`.
4. **Time** the GPU kernel (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the run-varying detail (model size, best-index
  cross-check), so it is shown but never diffed.

## What you are seeing

The demo trains a tiny **first-order Markov language model** on 16 synthetic
SMILES strings, then samples **4096 novel molecules** in parallel on the GPU
(one thread per molecule, each with its own reproducible RNG stream). It reports:

- **drug-like hits** — how many generated molecules clear a fixed toy score
  threshold (a stand-in for "fraction of valid, useful candidates");
- **mean reward** — the average toy drug-likeness score (the signal an RL loop
  would push upward — see THEORY §"distribution learning vs goal-directed");
- **best molecule** — the single highest-scoring SMILES (the "goal-directed"
  pick), reconstructed deterministically from its index.

## Expected result

```
1.10 -- De Novo Generative Molecular Design
trained first-order Markov model on 16 SMILES; generated 4096 molecules
drug-like hits (score >= 500): 1289 / 4096
mean reward: 44 milli-units
best molecule: idx=88  SMILES=CNOCO4#2=2Ncc1CCCCCCcOn4Occ1#c4221  score=1500 milli-units
RESULT: PASS (GPU matches CPU exactly: 4096/4096 molecules identical)
```

> The "best molecule" string is intentionally not a real drug — a first-order
> Markov model over a permissive alphabet and a **toy** scorer (see
> `THEORY.md`) produce structurally loose SMILES. The point is the *pipeline*
> (learn → sample → score → select) and its exact GPU/CPU parity, not the
> chemistry of any one output.
