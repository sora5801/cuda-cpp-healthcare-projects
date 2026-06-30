# Demo — 2.16 ΔΔG Stability Prediction

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed synthetic protein in `data/sample/`.
3. **Verify** the GPU saturation-mutagenesis scan against the CPU reference
   (`reference_cpu.cpp`) and print a clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## What you are looking at

The program runs a **deep mutational / saturation-mutagenesis scan**: for each of
the 24 residues it predicts the ΔΔG (folding-stability change, kcal/mol) of
mutating that residue to all 20 amino acids — a 24 × 20 = 480-cell grid. Each
cell is one independent GPU thread (PATTERNS.md §1, "batched masked prediction").

The synthetic protein has a designed buried hydrophobic core, so the most
**destabilising** predicted mutations are exactly the buried bulky residues
(Trp/Tyr/Phe) mutated to tiny, flexible **glycine** — destroying core packing.
That recovers the planted signal, which is the point of the demo.

> **Synthetic & didactic.** The model is a transparent physics-inspired scoring
> function, not a trained predictor; the protein is synthetic. See the project
> README "Limitations & honesty" and THEORY.md. Not for any real use.

## Expected result

```
2.16 -- Delta-Delta-G Stability Prediction
Saturation mutagenesis scan: protein 'synthetic_core_helix', 24 residues x 20 AA = 480 mutations
top-5 most DESTABILISING mutations (most negative ddG):
  #1  W17G   ddG =  -7.9373 kcal/mol
  #2  Y9G   ddG =  -7.7487 kcal/mol
  #3  F13G   ddG =  -7.7326 kcal/mol
  #4  Y5G   ddG =  -7.7283 kcal/mol
  #5  F18G   ddG =  -7.7137 kcal/mol
summary: 376 of 456 non-self mutations are destabilising (ddG<0); mean ddG = -1.7295 kcal/mol
RESULT: PASS (GPU matches CPU within tol=1.0e-03 kcal/mol)
```

Mutation codes read `<wild-type><1-based position><mutant>`, e.g. `W17G` =
Trp at position 17 → Gly. The `RESULT: PASS` line means the GPU grid agreed with
the CPU reference within the documented 1e-3 kcal/mol tolerance.
