# Push 2026-06-28 #01 -- flagship 1.12 tanimoto

> Push-note (CLAUDE.md §7.1). First Phase 1 flagship — the drug-discovery domain.

## 1. Summary

The first **flagship** is done: **1.12 Molecular Fingerprint Similarity Search**, a complete, verified CUDA
Tanimoto similarity search that replaces the SAXPY placeholder with a real biomedical computation. It sets
the Phase 1 quality bar — real kernel, CPU reference, full `THEORY.md`, synthetic data pipeline, and a demo
that verifies GPU == CPU bit-for-bit. This is the canonical "embarrassingly parallel" GPU pattern and a
clean first real project to study.

## 2. What changed

- [`projects/01-drug-discovery/1.12-molecular-fingerprint-similarity-search/`](../projects/01-drug-discovery/1.12-molecular-fingerprint-similarity-search) — fully implemented:
  - `src/kernels.cu` — `tanimoto_kernel` (constant-memory query, `__popcll`, grid-stride) + host wrapper.
  - `src/reference_cpu.cpp` / `.h` — data model (`FingerprintSet`, `FP_WORDS=32`), loader, serial reference.
  - `src/main.cu` — load → CPU + GPU → verify → top-5 report (deterministic stdout / timing on stderr).
  - `THEORY.md`, `README.md` (all sections), `data/` (synthetic sample + provenance), `scripts/`, `demo/`.
- `docs/STATUS.md` — `1.12` → **done** (1/301).
- `CHANGELOG.md` — indexed this push-note.

## 3. New projects (didactic blurb)

**1.12 Tanimoto fingerprint similarity** teaches the most fundamental GPU pattern: *one independent job per
thread*. A molecule becomes a 2048-bit fingerprint; similarity is `popcount(A&B)/popcount(A|B)`. The query
goes in **constant memory** (broadcast warp-wide), the library streams from global memory, and each thread
scores one molecule with the single-instruction **`__popcll`** popcount inside a **grid-stride loop**. The
most interesting thing to look at is `src/kernels.cu`: why the query belongs in constant memory and how the
grid-stride loop lets a fixed grid cover a library of any size.

## 4. How to build & run

```powershell
cd projects/01-drug-discovery/1.12-molecular-fingerprint-similarity-search
msbuild build/molecular-fingerprint-similarity-search.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> top-5 hits + RESULT: PASS (GPU matches CPU)
```

## 5. What to study here

Reading path: `THEORY.md` (§2 the math, §4 the GPU mapping) → `src/main.cu` (the 5-step shape) →
`src/kernels.cu` (constant memory + `__popcll` + grid-stride) → `src/reference_cpu.cpp` (the trusted
baseline). Then try the README **Exercises**: move top-K onto the GPU (`cub::DeviceRadixSort`), or transpose
the library to column-major and measure the coalescing effect.

## 6. Verification

- ✅ `Release|x64` **and** `Debug|x64` build with **zero errors / zero warnings** (MSBuild).
- ✅ Demo **PASS**: deterministic stdout matches `demo/expected_output.txt`; top-5 hits reported.
- ✅ **GPU == CPU bit-for-bit**: `max_abs_err = 0.000e+00` (popcount is exact integer; division is IEEE).
- ✅ `verify_project.py` → **DONE** (all structure/README/THEORY gates; comment ratio **0.85**; no TODOs).
- **Environment:** RTX 2080 (`sm_75`), CUDA 13.3, VS 2026 (`v145`). Kernel ~0.011 ms vs CPU ~0.061 ms on the
  tiny 64-molecule sample (a teaching artifact — the GPU's real advantage is at library scale).

## 7. Known limitations / TODOs

- Sample is **synthetic** (labeled everywhere); similarities carry no chemical meaning.
- Top-K is computed on the host (the parallelized bottleneck — scoring — is on the GPU). On-device top-K is
  left as Exercise 1.
- Exact brute force only — no LSH/approximate indexing, and the whole library must fit in GPU memory.

## 8. Next push preview

Next flagship: **3.01 Smith-Waterman / Needleman-Wunsch alignment** (genomics) — a dynamic-programming
kernel with an anti-diagonal wavefront, a very different GPU pattern (data dependencies, not embarrassing
parallelism). Continuing through the 14 flagships, pushing per flagship.
