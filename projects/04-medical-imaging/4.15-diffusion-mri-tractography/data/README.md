# Data — 4.15 Diffusion MRI & Tractography

## Committed sample (`sample/dwi_sample.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** (forward-simulated by `scripts/make_synthetic.py`) |
| License | Public domain (CC0) — it is synthetic |
| Size | 16×16×4 = 1024 voxels × 13 measurements, ~150 KB text |
| Why synthetic | Real dMRI (HCP/ABCD/UK Biobank) requires a data-use agreement and is far too large to commit. We embed a **known ground-truth** so the demo is interpretable and verifiable. |

The sample is generated **deterministically** (no RNG, no noise) so
`demo/expected_output.txt` is stable. It runs the demo **offline, with zero
downloads** — a hard requirement (CLAUDE.md §8).

### What the sample contains

A tiny 3-D volume with a **curved fiber bundle** (a quarter-circle arc, repeated
in each z-slice) of highly anisotropic tissue embedded in an isotropic
background. Each voxel's diffusion-weighted signal is computed from the
**Stejskal–Tanner** equation `S_k = S0 · exp(−b_k · gᵏᵀ D gᵏ)` for a chosen
ground-truth tensor `D`:

- **On the bundle:** an axially-symmetric ("cigar") tensor with eigenvalues
  λ∥ = 1.7e-3, λ⊥ = 0.3e-3 mm²/s, fast axis along the local arc tangent → high
  anisotropy (FA ≈ 0.80).
- **Background:** an isotropic tensor λ = 0.9e-3 mm²/s → FA ≈ 0.

So the fit should recover FA ≈ 0.80 along the bundle with the principal
eigenvector v1 pointing along the arc, and tractography should reconstruct the
curve. That built-in answer is what makes the demo output *checkable*.

### File format

```
<nx> <ny> <nz> <nmeas>                 # e.g. "16 16 4 13"  (nmeas MUST be 13)
<mask> S_0 S_1 ... S_12                # one line per voxel, x fastest then y then z
...                                    # nx*ny*nz voxel lines total
```

- `nmeas = 13` = 1 non-diffusion-weighted (b=0) image + 12 diffusion directions.
- `S_0` is the b=0 signal; `S_1..S_12` are the diffusion-weighted signals at the
  fixed **1 + 12 icosahedral** gradient scheme (b = 1000 s/mm², defined in
  `src/reference_cpu.cpp::make_gradient_scheme` and mirrored in
  `scripts/make_synthetic.py`).
- `mask` is 1 for tissue (fit + seed tractography here), 0 for background.
- The loader (`src/reference_cpu.cpp::load_dwi`) rejects a file whose `nmeas`
  differs from the compiled `NMEAS`.

## Full (real) dataset

Real diffusion MRI comes as a 4-D NIfTI volume plus a `bvec`/`bval` table (the
gradient scheme). Public sources — each requiring a free account and a data-use
agreement (which we do **not** bypass):

- **Human Connectome Project (HCP)** — 3T/7T multi-shell dMRI:
  <https://db.humanconnectome.org/>
- **ABCD Study** dMRI: <https://abcdstudy.org/>
- **UK Biobank** dMRI: <https://www.ukbiobank.ac.uk/>

`scripts/download_data.ps1` / `.sh` print the links and a conversion recipe
(DIPY/nibabel or MRtrix3 `mrconvert`) to write a small ROI into this project's
text format. For a larger synthetic volume instead:

```
python scripts/make_synthetic.py --nx 64 --ny 64 --nz 32
```

## Provenance & honesty

The committed sample is **synthetic** and labeled as such everywhere. The signals
are forward-simulated, noise-free, and carry **no clinical meaning**. Nothing here
may be used for diagnosis or any real medical decision (repository-wide rule).
