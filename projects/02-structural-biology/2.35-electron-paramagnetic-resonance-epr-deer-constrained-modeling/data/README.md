# Data — 2.35 Electron Paramagnetic Resonance (EPR/DEER) Constrained Modeling

> **Everything in `sample/` is SYNTHETIC.** It contains no real protein, no real
> spin labels, and no real EPR measurements. It is engineered so the demo has a
> *known answer* you can check (CLAUDE.md §8). Nothing here is for clinical or
> research use.

## Committed sample (`sample/`)

| Field | Value |
|---|---|
| File | `sample/deer_sample.txt` |
| Origin | **Synthetic** — produced by `scripts/make_synthetic.py` (seed 2025) |
| License | Public domain (CC0) — it is synthetic |
| Size | ~80 KB (well under the 50 MB commit limit) |
| Contents | 64 ensemble frames × two 24-point spin-label rotamer clouds + a target P(r) |

This tiny file lets `demo/run_demo` run **offline, with zero downloads**, a hard
requirement for every project (CLAUDE.md §8).

### File format (parsed by `src/reference_cpu.cpp :: load_ensemble`)

Whitespace-separated text; lines beginning with `#` are comments and ignored.

```
M ROTAMERS NBINS                      # header (must match src/deer_params.h)
# ---- repeated M times, one block per ensemble frame ----
truth_flag                            # 1 if this frame is a synthetic "true" match, else 0
x y z                                 # site-A rotamer endpoint 1 (nanometres)
 ... (ROTAMERS lines total) ...
x y z                                 # site-B rotamer endpoint 1 (nanometres)
 ... (ROTAMERS lines total) ...
# ---- after all frames ----
P_exp(bin 0)                          # experimental target distribution, NBINS values
 ... (NBINS lines total) ...          # (re-normalized to sum 1 on load)
```

**Field meanings.**
- `M` — number of ensemble members (MD frames). `ROTAMERS` — spin-label rotamers
  modelled per site per frame. `NBINS` — number of distance bins in `P(r)`.
- `truth_flag` — *synthetic ground truth only*: marks the frames whose design
  distance equals the target's peak. The demo reports how much population
  reweighting puts back onto these frames; the algorithm never sees the flag.
- `x y z` — the 3-D position (in **nanometres**) of one spin-label rotamer's
  unpaired-electron site (the nitroxide N–O midpoint). Site A and site B are the
  two labelled residues; the spin–spin distance is `|A_i − B_j|`.
- `P_exp` — the "measured" DEER distance distribution we are fitting to (a
  Gaussian centred on the true distance in the synthetic case).

How it was built: each frame places its two rotamer clouds a chosen distance
apart, so its back-calculated `P_m(r)` is a narrow bump at that distance. The
first 16 frames are at the true distance (3.5 nm); the rest are decoys at other
distances. See `scripts/make_synthetic.py` for the exact recipe. Regenerate (or
make it bigger) with:

```
python scripts/make_synthetic.py                # the committed 64-frame sample
python scripts/make_synthetic.py --frames 400   # a larger ensemble
```

## Full / real dataset

There is no single "EPR dataset" to download for this teaching project — real
DEER-restrained ensemble modelling combines **a protein structure**, **a spin-
label rotamer library**, and **an experimental P(r)**. `scripts/download_data.*`
prints pointers to the real resources; it never bypasses any registration.

Catalog dataset notes (verbatim):

> SASBDB EPR-constrained structures (verify URL); published DEER datasets for
> membrane transporters; EPR.cxls community datasets (verify URL); PDB structures
> refined with EPR data.

Useful real-world resources (verify URLs before relying on them):

- **SASBDB** — small-angle scattering / EPR-constrained structures: <https://www.sasbdb.org/>
- **MMM** (Multiscale Modeling of Macromolecules) — MTSSL rotamer libraries and
  DEER back-calculation: <https://www.epr.ethz.ch/software/mmm.html>
- **DEER-PREdict** — DEER/PRE prediction from MD ensembles (search the Lindorff-
  Larsen lab / GitHub; verify URL).
- **BioEn / EnsembleFit** — Bayesian ensemble reweighting: <https://github.com/bio-phys/BioEN>
- **PDB** — many entries are refined against EPR restraints: <https://www.rcsb.org/>

To use real data, export your model's two label sites' rotamer clouds (e.g. from
MMM) and your experimental `P(r)` into the format above, matching the header to
`src/deer_params.h`.

## Provenance & honesty

The committed sample is **100% synthetic** and labelled as such in the file
header, in `make_synthetic.py`, and here. It exists only to make the kernel and
the reweighting verifiable offline. It does **not** represent any real protein or
measurement and must never be presented as clinically or scientifically valid.
