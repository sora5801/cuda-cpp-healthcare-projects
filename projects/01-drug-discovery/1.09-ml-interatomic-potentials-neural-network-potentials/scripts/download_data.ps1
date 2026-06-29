# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 1.9 : ML Interatomic Potentials (Neural Network Potentials)
#
# CONTRACT (CLAUDE.md sec 8): idempotent, documented, prints the source URL +
# expected size, and NEVER bypasses credentials/registration. The committed
# synthetic sample is enough to run the demo offline; the real NNP training sets
# below are large research datasets you fetch only if you want to go further.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.9 -- ML Interatomic Potentials (NNP)"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The committed tiny sample (data/sample/water_cluster.xyzc) is SYNTHETIC and"
Write-Host "is all the demo needs. For a larger synthetic cluster, run:"
Write-Host "    python scripts/make_synthetic.py --mols 64"
Write-Host ""
Write-Host "REAL NNP training datasets (quantum-chemistry energies/forces):"
Write-Host "  ANI-1ccx  CCSD(T) energies, ~500k conformers   https://github.com/isayev/ANI1ccx_dataset"
Write-Host "  SPICE     DFT energies+forces, drugs+peptides   https://github.com/openmm/spice-dataset"
Write-Host "  rMD17     revised MD17 force benchmark           https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038"
Write-Host "  OE62      62k organic molecules, DFT energetics  (verify current URL)"
Write-Host ""
Write-Host "These are HDF5/archive downloads (hundreds of MB to GB). This teaching"
Write-Host "project does NOT train on them -- it uses a fixed, manufactured network to"
Write-Host "demonstrate the descriptor + per-atom MLP pipeline. To USE the real data you"
Write-Host "would: (1) read geometries/energies, (2) train weights (TorchANI/NequIP/MACE),"
Write-Host "(3) export the weights into AtomicNet. See README 'Prior art & further reading'."
Write-Host ""
Write-Host "Idempotent fetch pattern when wiring a real set:"
Write-Host "  1) skip download if the file already exists with the right checksum"
Write-Host "  2) print source URL + expected size + SHA256"
Write-Host "  3) for credentialed sets, print registration instructions ONLY (never bypass)"
