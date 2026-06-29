# THEORY — 12.01 Mass-Spectrometry Proteomics Search

> For a reader who knows C++ but is new to CUDA and to proteomics. See
> [README.md](README.md) for the tour and build. _Educational only._

## 1. The science

In shotgun proteomics, proteins are digested into peptides, separated, and
fragmented in a mass spectrometer, producing **MS/MS spectra**: each is a set of
(mass-to-charge, intensity) peaks from the peptide's fragment ions. To identify
the peptide behind an observed spectrum, we **search** it against a database of
**theoretical** spectra predicted from a protein sequence database. The peptide
whose theoretical spectrum best matches the observation is the identification.
This database search is the central, and most expensive, computation in
proteomics.

## 2. The math

Bin each spectrum to a fixed-length intensity vector (peaks fall into m/z bins).
For an observed spectrum `q` and a theoretical spectrum `r`, the **cosine
similarity** (a.k.a. normalized dot product, or the spectral contrast angle's
cosine) is

```
cos(q, r) = (q · r) / (‖q‖ ‖r‖) = ( Σ_b q_b r_b ) / ( sqrt(Σ q_b²) · sqrt(Σ r_b²) )
```

It lies in `[0,1]` for non-negative intensities: `1` = identical peak pattern, `0`
= no shared peaks. Normalizing by the norms makes the score independent of overall
intensity scale, so a faint and a strong copy of the same peptide score alike.

## 3. The algorithm

```
precompute ‖q‖ and ‖r_i‖ for every library spectrum     # once
for each library spectrum i:                            # PARALLEL
    score_i = (q · r_i) / (‖q‖ ‖r_i‖)
report the top-K highest scores
```

**Complexity.** Scoring is `Θ(N · bins)` for `N` library spectra. Dense binning
makes every comparison touch every bin; sparse/indexed methods (§7) cut this
dramatically. The top-K selection is `Θ(N log K)`.

## 4. The GPU mapping

**Decomposition.** One thread per library spectrum (a 1-D grid over `N`). Thread
`i` reads its spectrum's row from global memory and the query from **constant
memory**, accumulates the dot product, and divides by the precomputed norms.

**Constant memory for the query.** Every thread reads all `bins` query values and
none writes them — identical to project 1.12's Tanimoto search. Constant memory's
broadcast cache serves one address to a whole warp in a single transaction, so the
query is not re-fetched from global memory by every thread. (Here `bins ≤ 1024`
fits the 4 KB we reserve, well within the 64 KB constant bank.)

**Numerics.** The dot product accumulates in **double** even though the data is
`float`, which keeps the score accurate and — crucially — lets the CPU and GPU
agree: both do the same double accumulation in the same bin order. With the norms
precomputed identically, the cosine scores match to `~0` (no reduction across
threads, so nothing reorders). The score is deterministic and reproducible.

**Why this is fast.** The kernel is a streaming dot product: it reads each library
spectrum once (coalesced across threads for a fixed bin) and does `bins`
multiply-adds — memory-bandwidth bound, the GPU's strength. The advantage grows
with `N` (and with batching many query spectra).

## 5. Numerical considerations

- **Precision.** `float` storage, `double` accumulation — standard for dot
  products of many terms.
- **Determinism.** Each thread writes its own score; no atomics or cross-thread
  reduction, so the result is reproducible and CPU-matching.
- **Zero norms.** An empty spectrum has norm 0; we guard the division to return 0
  rather than NaN.

## 6. How we verify correctness

`main.cu` scores the query on CPU (`cosine_cpu`) and GPU (`cosine_gpu`) and
compares all `N` scores (`max_abs_err ≈ 0`). Beyond CPU/GPU parity, the search is
*correct*: the query was synthesized from library spectrum 7 (with intensity
jitter and a little noise), and it is recovered at **rank 1** with cosine ≈ 0.993,
while unrelated spectra score ≈ 0.3 — the method actually finds the right peptide,
not just two codes agreeing.

## 7. Where this sits in the real world

Production engines (MSFragger, GiCOPS, Comet, SpectraST) add what this teaching
version omits: **sparse** spectra (peaks, not dense bins — most bins are empty),
**fragment-ion indexing** (invert the library so a query only scores *candidate*
spectra sharing peaks — the key to scaling to 10⁶ peptides), **precursor-mass
filtering**, **decoy databases + FDR** control, **post-translational
modifications**, and richer scores (XCorr, hyperscore). The per-comparison dot
product you parallelize here is their innermost loop; GiCOPS' contribution is
exactly moving that loop, plus the indexing, onto the GPU.

## References

- Eng, McCormack & Yates (1994) — SEQUEST / the cross-correlation score.
- Kong et al. (2017), *MSFragger* — fragment-ion indexing for fast search.
- Haseeb & Saeed, *GiCOPS* — GPU-accelerated database peptide search.
- NVIDIA CUDA C++ Programming Guide — constant memory and coalescing.
