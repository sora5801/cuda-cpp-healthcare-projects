# 12.01 — Mass-Spectrometry Proteomics Search

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Analytical%20%26%20Omics%20Data%20Processing-lightgrey)

> **🟢 Beginner · Established** — Domain 12: Analytical & Omics Data Processing · Catalog ID `12.01`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Identify a peptide by **spectral library search**: score one observed MS/MS
spectrum (the query) against thousands of theoretical spectra and return the best
matches. Each spectrum is a binned intensity vector; the match score is the
**cosine similarity** (normalized dot product). Every query-vs-library comparison
is independent, so one GPU thread scores one library spectrum, with the query in
constant memory. Eleventh distinct GPU pattern: **batched dot-product scoring**.

## What this computes & why the GPU helps

Database peptide search — correlating each observed spectrum against a peptide
database — is the most time-consuming step in proteomics: ~10⁵ spectra against
~10⁶ peptides is 10¹¹ comparisons. Each score is an independent dot product, so
GPUs parallelize scoring thousands of theoretical spectra per observed spectrum
(GiCOPS, Tempest). The query is read by every thread but never changes → constant
memory broadcasts it.

**The parallelized work** is the per-library cosine score; the small top-K
selection is a host step.

## The algorithm in brief

- **Binning:** each spectrum → a fixed-length intensity vector.
- **Score:** `cosine(q, lib_i) = dot(q, lib_i) / (‖q‖·‖lib_i‖)` (norms precomputed).
- **Rank:** report the top-K highest-scoring library spectra.

See [THEORY.md](THEORY.md) for spectral matching, normalization, and the sparse/indexed approaches.

## Build

Requires **Visual Studio 2026** (v145) + **CUDA 13.3** ([docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/mass-spectrometry-proteomics-search.sln`.
2. **`Release|x64`** → **Build** → `build/x64/Release/mass-spectrometry-proteomics-search.exe`.

CLI: `msbuild build\mass-spectrometry-proteomics-search.sln /p:Configuration=Release /p:Platform=x64`

## Run the demo

```powershell
./demo/run_demo.ps1
```

Scores the query against the library on CPU + GPU and verifies they match.

## Data

- **Sample (committed):** `data/sample/spectra_sample.txt` — 1 query + 1024 **synthetic** spectra.
- **Real data:** mzML from ProteomeXchange/PRIDE searched against a peptide DB — see
  `scripts/download_data.ps1` and [data/README.md](data/README.md).
- Bigger synthetic set: `python scripts/make_synthetic.py --N 8192`.

## Expected output

`demo/expected_output.txt` holds the deterministic top-5 matches and the rank of
the known target. The GPU (`src/kernels.cu`) and CPU (`src/reference_cpu.cpp`)
compute the dot product in double, so their cosine scores agree to `~0`. The query
(derived from library spectrum 7) is recovered at **rank 1** (cosine ≈ 0.993).

## Code tour

1. [`src/main.cu`](src/main.cu) — load, compute norms, CPU + GPU score, verify, top-K, print.
2. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the constant-memory-query idea.
3. [`src/kernels.cu`](src/kernels.cu) — the cosine kernel (one thread per library spectrum).
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — norms + the serial cosine reference.

## Prior art & further reading

- **GiCOPS** (<https://github.com/pcdslab/gicops>) — GPU database peptide search (fragment indexing).
- **MSFragger** (<https://github.com/Nesvilab/MSFragger>) — ultra-fast hash-indexed search.
- **OpenMS** (<https://github.com/OpenMS/OpenMS>) — proteomics toolkit (mzML, scoring).
- SpectraST / spectral-library search; the **spectral contrast angle** score.

Study these for the production approach; reimplement didactically (CLAUDE.md §2).

## CUDA pattern used here

**Batched dot-product scoring**: one thread per library spectrum, query in
**constant memory** (warp broadcast), double-accumulated dot product, precomputed
norms · top-K on the host. The same shape as `1.12` (Tanimoto), with real-valued
intensities and a cosine score.

## Exercises

1. **Sparse spectra.** Store each spectrum as (bin, intensity) peak lists and score
   by merging sorted peaks — the representation real engines use (most bins are 0).
2. **Fragment-ion indexing.** Invert the library (bin → list of spectra with a peak
   there) so a query only touches candidate spectra (the MSFragger/GiCOPS idea).
3. **On-GPU top-K.** Replace the host `partial_sort` with `cub::DeviceRadixSort`.
4. **Better scores.** Implement the dot-product **spectral contrast angle** or a
   cross-correlation (XCorr) score and compare rankings.
5. **Many queries.** Score a batch of observed spectra at once (a 2-D launch) — the
   real throughput case.

## Limitations & honesty

- **Dense binned cosine** on synthetic random spectra; real search uses sparse
  peaks, fragment-ion indexing, precursor-mass filtering, decoys/FDR, and PTMs.
- The query is broadcast from constant memory (fits ≤ 1024 bins); top-K is on the host.
- Data is synthetic and has no biological meaning.
