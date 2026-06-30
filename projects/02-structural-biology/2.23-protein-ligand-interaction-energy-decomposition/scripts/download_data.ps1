# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 2.23 : Protein-Ligand Interaction Energy Decomposition
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# expected size + checksum, and NEVER bypasses credentials/registration. If a
# dataset needs an account, this script only prints instructions + links and
# defers to scripts/make_synthetic.py for an offline stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.23 -- Protein-Ligand Interaction Energy Decomposition"
Write-Host "[download_data] Target data dir: $DataDir"

# This teaching project runs entirely on the committed SYNTHETIC sample (no real
# structure/force-field parser is shipped). Real per-residue MM-GBSA needs a full
# MD+parameter stack (AMBER prmtop or GROMACS top + a trajectory), so this script
# prints where to obtain real complexes and how to feed them through the proper
# tools, rather than pretending to download a ready-to-run file.
Write-Host ""
Write-Host "This project ships a SYNTHETIC sample only (data/sample/complex_sample.txt)."
Write-Host "The committed tiny sample is enough to run the demo offline."
Write-Host ""
Write-Host "For a larger SYNTHETIC system, regenerate with more residues/frames:"
Write-Host "    python scripts/make_synthetic.py --residues 200 --frames 500"
Write-Host ""
Write-Host "To work with REAL protein-ligand complexes, obtain structures from:"
Write-Host "    PDBbind   : http://www.pdbbind.org.cn   (curated complexes + affinities)"
Write-Host "    KLIFS     : https://klifs.net           (kinase-ligand structures)"
Write-Host "    ChEMBL    : https://www.ebi.ac.uk/chembl/ (activity data for target families)"
Write-Host "    ClinVar   : https://www.ncbi.nlm.nih.gov/clinvar/ (resistance mutations)"
Write-Host ""
Write-Host "Then produce per-residue MM-GBSA inputs with a proper toolchain (study these):"
Write-Host "    AMBER MMPBSA.py decomp  : https://ambermd.org/AmberTools.php"
Write-Host "    gmx_MMPBSA              : https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA"
Write-Host "Respect each source's license; do not redistribute restricted structures."
