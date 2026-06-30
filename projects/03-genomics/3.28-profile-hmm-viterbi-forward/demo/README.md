# Demo — 3.28 Profile HMM (Viterbi / Forward)

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/phmm_sample.fasta` input — building a
   profile HMM from the consensus and scoring the 7 database sequences with both
   **Viterbi** (best path) and **Forward** (all paths).
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`) for
   *both* algorithms and print a clear `PASS`/`FAIL`.
4. **Time** the kernels (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## What to look for

- **The ranking.** `homolog` (a lightly mutated copy of the consensus) sits at
  **rank 1** with a Viterbi score of `-31.85` nats, ~54 nats above the best random
  decoy. That margin is the whole point: a profile HMM separates true family
  members from noise. This is the scientific check, beyond CPU==GPU agreement.
- **Viterbi vs Forward.** Both columns are printed. Forward (sum over all paths)
  is always ≥ Viterbi (best single path) for the same sequence — visible in the
  numbers (e.g. homolog `-31.13` Forward vs `-31.85` Viterbi).
- **Exact agreement.** The stderr line shows `max_abs_err: Viterbi 0.000e+00
  Forward 0.000e+00` — the shared `__host__ __device__` core makes GPU and CPU
  bit-identical (THEORY §6).

## Expected result (stdout)

```
3.28 -- Profile HMM (Viterbi / Forward)
profile: 24 match columns (consensus 'MKTAYIAKQRQISFVKSHFSRQLE')
database: 7 sequences scored (Viterbi + Forward, log-prob in nats)
rank by Viterbi score (best path):
  rank name            viterbi      forward
  1    homolog        -31.8517     -31.1334
  2    decoy3         -85.7719     -81.1212
  3    decoy6         -85.7719     -81.4085
  4    decoy2         -87.8149     -82.8099
  5    decoy1         -88.4636     -82.7063
  6    decoy5         -88.4636     -83.3374
  7    decoy4         -92.0621     -84.7652
top hit: homolog  (Viterbi -31.8517, 53.9202 nats above runner-up)
RESULT: PASS (GPU matches CPU within tol=1.0e-04)
```

(Note `decoy3`/`decoy6` tie on Viterbi and are ordered by lower index — a
deterministic tie-break, so stdout is reproducible. Forward distinguishes them.)
