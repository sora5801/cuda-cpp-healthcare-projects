#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  How to get REAL diffusion MRI (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 4.15 : Diffusion MRI & Tractography
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. Every real dMRI dataset below requires
# a free account and a data-use agreement, so this script only prints instructions
# and defers to scripts/make_synthetic.py for an offline stand-in. It downloads
# nothing by itself.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.15 -- Diffusion MRI & Tractography"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "Real diffusion MRI comes as NIfTI volumes plus a b-vectors/b-values table"
echo "(the gradient scheme). All public sources require a free account and a"
echo "data-use agreement -- respect them; this script will NOT bypass any login:"
echo "  * Human Connectome Project (HCP), 3T/7T multi-shell: https://db.humanconnectome.org/"
echo "  * ABCD Study dMRI:      https://abcdstudy.org/"
echo "  * UK Biobank dMRI:      https://www.ukbiobank.ac.uk/"
echo
echo "Recommended tooling to read/convert a real dataset (study, do not copy"
echo "wholesale -- see README 'Prior art'):"
echo "  * MRtrix3 (mrconvert, dwi2tensor, tckgen): https://www.mrtrix.org/"
echo "  * DIPY (Python; read_bvals_bvecs, TensorModel): https://dipy.org/"
echo "  * FSL (dtifit, bedpostx):  https://fsl.fmrib.ox.ac.uk/"
echo
echo "To convert a NIfTI DWI + bvec/bval into THIS project's text format (one b0 +"
echo "12 directions per voxel; see data/README.md), a short DIPY/nibabel script"
echo "suffices: load the 4-D volume, pick a small ROI, and write '<mask> S_0 .. S_12'"
echo "per voxel. Keep NMEAS = 13 to match the compiled gradient scheme"
echo "(src/dti_core.h) or regenerate make_gradient_scheme."
echo
echo "Offline stand-in (no download, fully reproducible, SYNTHETIC):"
echo "  python scripts/make_synthetic.py                 # the committed 16x16x4 sample"
echo "  python scripts/make_synthetic.py --nx 64 --ny 64 --nz 32   # a bigger volume"
