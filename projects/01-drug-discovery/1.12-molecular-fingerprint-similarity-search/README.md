# 1.12 — Molecular Fingerprint Similarity Search

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.12`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Given one **query** molecule and a large **library** of molecules, find the most
chemically similar library members. Each molecule is encoded as a 2048-bit
**fingerprint**, and similarity is the **Tanimoto coefficient**
`popcount(A & B) / popcount(A | B)`. Every query-vs-library comparison is
independent, so we put one GPU thread on each library molecule — the textbook
"embarrassingly parallel" workload, and the foundation of billion-compound
virtual screening.

## What this computes & why the GPU helps

Tanimoto similarity over Morgan/ECFP bit-vectors is the standard metric for
chemical similarity searching. A brute-force scan of a query against 100M
compounds is ~10¹⁰ bit-AND + popcount operations — and every one is independent.
That is a perfect fit for GPU SIMD: a single 2048-bit fingerprint is 32 × 64-bit
words, and a thread evaluates one query-vs-library pair in a few nanoseconds.
Production tools (Schrödinger's `gpusimilarity`) load an entire library into GPU
memory and retrieve from billion-compound sets in under a second.

**The parallel bottleneck** is the per-pair `AND`/`OR`+popcount over the library;
it is `O(N · FP_WORDS)` and dominates runtime. We parallelize it across the
library dimension `N` (one thread per molecule), keep the shared query in
constant memory, and use the `__popcll` hardware popcount intrinsic.

## The algorithm in brief

- **Fingerprint:** a fixed 2048-bit vector; bit *k* = "substructure *k* present".
- **Tanimoto / Jaccard:** `T(A,B) = |A ∧ B| / |A ∨ B|` via `popcount`.
- **Search:** score the query against all `N` library fingerprints, then take the
  **top-K** highest scores.

See [THEORY.md](THEORY.md) for the science → math → algorithm → GPU-mapping derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3** (see
[docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/molecular-fingerprint-similarity-search.sln`.
2. Select **`Release|x64`**.
3. **Build → Build Solution** → `build/x64/Release/molecular-fingerprint-similarity-search.exe`.

CLI: `msbuild build\molecular-fingerprint-similarity-search.sln /p:Configuration=Release /p:Platform=x64`

## Run the demo

```powershell
./demo/run_demo.ps1
```

Builds if needed, runs on the committed sample, prints the **top-5 hits**, the
**GPU-vs-CPU agreement** check, and a timing line.

## Data

- **Sample (committed):** `data/sample/fingerprints_sample.txt` — 1 query + 64
  **synthetic** 2048-bit fingerprints (deterministic; scores span a wide range).
- **Full dataset:** generate ECFP4 fingerprints from ChEMBL/ZINC/PubChem with
  **RDKit** — see `scripts/download_data.ps1` and [data/README.md](data/README.md).
- For a library-scale synthetic set: `python scripts/make_synthetic.py --n 1000000`.

## Expected output

`demo/expected_output.txt` holds the deterministic stdout. The program computes
all `N` scores on the **GPU** (`src/kernels.cu`) and on a **CPU reference**
(`src/reference_cpu.cpp`) and asserts they agree within `1e-6`. Because popcount
is exact integer arithmetic and the division is IEEE, the two agree **bit-for-bit**
(`max_abs_err = 0`).

## Code tour

1. [`src/main.cu`](src/main.cu) — load fingerprints, run CPU + GPU, verify, print top-K.
2. [`src/reference_cpu.h`](src/reference_cpu.h) — the data model (`FP_WORDS`, `FingerprintSet`) + loader/reference prototypes.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the constant-memory / `__popcll` idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel (grid-stride, constant-memory query) + host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline.

## Prior art & further reading

- **gpusimilarity** (<https://github.com/schrodinger/gpusimilarity>) — CUDA/Thrust brute-force fingerprint search; the production analogue of this project.
- **FPSim2** (<https://github.com/chembl/FPSim2>) — fast similarity search (PyTables + popcount).
- **RDKit** (<https://github.com/rdkit/rdkit>) — generates the Morgan/ECFP fingerprints we consume.
- **Faiss** (<https://github.com/facebookresearch/faiss>) — GPU ANN search for molecular embeddings (the approximate cousin of brute force).

Study these for the production approach; reimplement didactically rather than copying (CLAUDE.md §2).

## CUDA pattern used here

Constant memory (broadcast query) · `__popcll` 64-bit popcount intrinsic ·
grid-stride loop over the library · independent outputs (no atomics/shared mem) ·
top-K on the host (the production path uses `cub::DeviceRadixSort`).

## Exercises

1. **Top-K on the GPU.** Replace the host `partial_sort` with a Thrust
   `sort_by_key` or a `cub::DeviceRadixSort` on the device scores. Does it help
   at `n = 64`? At `n = 1,000,000`?
2. **Warp-shuffle reduction.** Assign one *warp* per molecule and reduce the 32
   word-popcounts with `__shfl_down_sync`. When is that better than one thread
   per molecule?
3. **Memory layout.** Try a *column-major* (transposed) library so word `w` of
   all molecules is contiguous. Measure the effect on memory coalescing.
4. **Threshold search.** Add a `--min-sim` filter that returns *all* hits above a
   similarity threshold (a stream-compaction problem — try `thrust::copy_if`).
5. **Bigger fingerprints.** Change `FP_WORDS` to 16 (1024-bit) or 64 (4096-bit),
   regenerate the sample, and confirm CPU/GPU still agree.

## Limitations & honesty

- The sample is **synthetic**; similarities carry **no chemical meaning**.
- Top-K is computed on the host (fine here; the real bottleneck — scoring — is on
  the GPU). At true library scale you would keep the reduction on-device.
- We load the whole library into device memory; production tools stream/sharded
  libraries that exceed GPU memory. No LSH/approximate indexing is implemented —
  this is exact brute force, the clearest thing to learn from.
