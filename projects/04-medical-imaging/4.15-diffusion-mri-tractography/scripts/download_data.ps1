# ===========================================================================
# scripts/download_data.ps1  --  How to get REAL diffusion MRI (Windows)
# ---------------------------------------------------------------------------
# Project 4.15 : Diffusion MRI & Tractography
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. Every real dMRI dataset below requires
# a data-use agreement, so this script only prints instructions + links and
# defers to scripts/make_synthetic.py for an offline stand-in. It downloads
# nothing by itself.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.15 -- Diffusion MRI & Tractography"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "Real diffusion MRI comes as NIfTI volumes plus a b-vectors/b-values"
Write-Host "table (the gradient scheme). All of the public sources require a free"
Write-Host "account and a data-use agreement -- respect them; this script will NOT"
Write-Host "bypass any login:"
Write-Host "  * Human Connectome Project (HCP) 1200-subject release, 3T/7T multi-shell"
Write-Host "    dMRI:                 https://db.humanconnectome.org/"
Write-Host "  * ABCD Study dMRI:      https://abcdstudy.org/"
Write-Host "  * UK Biobank dMRI:      https://www.ukbiobank.ac.uk/"
Write-Host ""
Write-Host "Recommended tooling to read/convert a real dataset (study, do not"
Write-Host "copy wholesale -- see README 'Prior art'):"
Write-Host "  * MRtrix3 (mrconvert, dwi2tensor, tckgen): https://www.mrtrix.org/"
Write-Host "  * DIPY (Python; read_bvals_bvecs, TensorModel): https://dipy.org/"
Write-Host "  * FSL (dtifit, bedpostx):  https://fsl.fmrib.ox.ac.uk/"
Write-Host ""
Write-Host "To convert a NIfTI DWI + bvec/bval into THIS project's text format"
Write-Host "(one b0 + 12 directions per voxel; see data/README.md), a short DIPY or"
Write-Host "nibabel script suffices: load the 4-D volume, pick a small ROI, and write"
Write-Host "'<mask> S_0 .. S_12' per voxel. Keep NMEAS = 13 to match the compiled"
Write-Host "gradient scheme (src/dti_core.h) or regenerate make_gradient_scheme."
Write-Host ""
Write-Host "Offline stand-in (no download, fully reproducible, SYNTHETIC):"
Write-Host "  python scripts/make_synthetic.py                 # the committed 16x16x4 sample"
Write-Host "  python scripts/make_synthetic.py --nx 64 --ny 64 --nz 32   # a bigger volume"
