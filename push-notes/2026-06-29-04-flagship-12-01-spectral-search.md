# Push 2026-06-29 #04 -- flagship 12.01 spectral-search

> Push-note (CLAUDE.md §7.1). Eleventh Phase 1 flagship — analytical & omics.

## 1. Summary

The omics flagship is done: **12.01 Mass-Spectrometry Proteomics Search**, a complete, verified GPU spectral
library search. It scores one observed MS/MS spectrum against thousands of theoretical spectra by cosine
similarity (normalized dot product) — one GPU thread per library spectrum, query in constant memory. Eleventh
distinct GPU pattern: **batched dot-product scoring** (the real-valued cousin of 1.12's Tanimoto search).

## 2. What changed

- [`projects/12-omics-data-processing/12.01-mass-spectrometry-proteomics-search/`](../projects/12-omics-data-processing/12.01-mass-spectrometry-proteomics-search) — fully implemented:
  - `src/kernels.cu` — `cosine_kernel` (constant-memory query, double-accumulated dot product) + wrapper.
  - `src/reference_cpu.cpp` / `.h` — norms + the serial cosine reference.
  - `src/main.cu` — load → norms → CPU + GPU score → verify → top-K + target rank.
  - `THEORY.md`, `README.md`, `data/` (synthetic query + 1024 library spectra), `scripts/`, `demo/`.
- `docs/STATUS.md` — `12.01` → **done** (11/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb)

**12.01 spectral search** teaches **batched dot-product scoring**: each library spectrum gets a thread that
computes a normalized dot product against the query (read from constant memory, broadcast warp-wide). The
double-accumulated dot product makes CPU and GPU agree exactly. The standout file is `src/kernels.cu`: why
the query belongs in constant memory and how cosine normalization makes the score intensity-invariant.

## 4. How to build & run

```powershell
cd projects/12-omics-data-processing/12.01-mass-spectrometry-proteomics-search
msbuild build/mass-spectrometry-proteomics-search.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> top-5 matches + target rank + RESULT: PASS
```

## 5. What to study here

Reading path: `THEORY.md` (§2 cosine, §4 constant-memory + numerics) → `src/kernels.cu` →
`src/reference_cpu.cpp`. Then try README **Exercises**: sparse peak-list spectra, fragment-ion indexing
(the MSFragger/GiCOPS idea), on-GPU top-K, or scoring a batch of queries.

## 6. Verification

- ✅ `Release|x64` **and** `Debug|x64` build with **zero errors / zero warnings**.
- ✅ Demo **PASS**: deterministic top-5 + target rank match `expected_output.txt`.
- ✅ **GPU == CPU exactly** (`max_abs_err = 0`; double-accumulated dot product).
- ✅ Search is correct: query (from library 7) recovered at **rank 1**, cosine 0.993; decoys ≈ 0.3.
- ✅ `verify_project.py` → **DONE** (comment ratio **0.62**, no TODOs).
- **GPU win:** CPU ~0.18 ms vs GPU ~0.08 ms on 1024 spectra; grows with library size (real: 10^6 peptides).
- **Environment:** RTX 2080 (`sm_75`), CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- **Dense binned cosine** on synthetic random spectra; real search uses sparse peaks, fragment-ion indexing,
  precursor filtering, decoys/FDR, and PTMs. Query broadcast from constant memory (≤1024 bins); top-K on host.

## 8. Next push preview

Next flagship: **13.02 PBPK ODE ensemble over virtual patients** (pharmacology) — a multi-compartment
pharmacokinetic ODE solved for thousands of virtual patients in parallel (an ensemble-RK4 variant of 9.02,
with a richer compartment model).
