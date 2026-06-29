# Demo — 1.20 Reaction Yield / Retrosynthesis Scoring

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/routes_sample.txt` (24 synthetic candidate
   retrosynthetic routes + a shared logistic yield model).
3. **Verify** the GPU per-route scores against the CPU reference and print a
   clear `PASS`/`FAIL`.
4. **Report** the **top-5 most synthesizable routes** (highest score first), and
   time the kernel (CUDA events) vs. the CPU baseline.

Output is split: the deterministic **top-K + PASS** goes to **stdout** (diffed
against [`expected_output.txt`](expected_output.txt)); the **timing** and the
numeric error go to **stderr** (shown, not diffed).

## Canonical output

See [`expected_output.txt`](expected_output.txt) for the exact stdout the demo
asserts:

```
1.20 -- Reaction Yield / Retrosynthesis Scoring
Scored 24 candidate retrosynthetic routes (<= 6 steps, 4 features each)
top-5 most synthesizable routes (higher score = better):
  #1  route[0]  score = 0.949571
  ...
RESULT: PASS (GPU matches CPU within tol=1e-06)
```

`route[0]` is the **planted best route** (short, high-prior, low-condition-penalty,
fully in-stock), so its #1 ranking is by construction — a sanity check that the
scoring and ranking work. A green `PASS` means the GPU scores matched the CPU
reference within `1e-6` (they share the same `route_score()`; the only difference
is a few-times-`1e-8` single-precision `expf`/FMA rounding — see `THEORY.md`).

> The scores reflect the **synthetic** sample; they carry no chemical meaning.
