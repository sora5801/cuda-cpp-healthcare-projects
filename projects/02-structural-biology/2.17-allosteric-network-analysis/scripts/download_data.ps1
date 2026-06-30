# ===========================================================================
# scripts/download_data.ps1  --  Fetch a REAL trajectory dataset (Windows)
# ---------------------------------------------------------------------------
# Project 2.17 -- Allosteric Network Analysis
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. The real allosteric-trajectory
# archives are large and account-gated, so this script PRINTS INSTRUCTIONS only
# and defers to scripts/make_synthetic.py for the offline stand-in the demo uses.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.17 -- Allosteric Network Analysis"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The committed tiny sample (data/sample/trajectory.txt) is SYNTHETIC and is"
Write-Host "all the demo needs. To study a REAL allosteric trajectory, obtain one of:"
Write-Host ""
Write-Host "  * GPCRmd allosteric trajectory archive  https://gpcrmd.org"
Write-Host "      Browse to a GPCR system, download the trajectory (.xtc/.dcd) +"
Write-Host "      topology (.pdb/.psf). Free, but registration is required -- this"
Write-Host "      script does NOT log in for you."
Write-Host "  * MDAnalysis test trajectories          https://github.com/MDAnalysis/mdanalysis"
Write-Host "  * ProDy benchmark structures/ensembles  https://github.com/prody/ProDy"
Write-Host "  * Allosteric Database (ASD)             http://mdl.shsmu.edu.cn/ASD/"
Write-Host ""
Write-Host "CONVERT a real trajectory into this project's plain-text format (the"
Write-Host "loader expects '# SITE_ALLO i', '# SITE_ACTIVE j', then 'N T', then T*N"
Write-Host "lines of 'x y z' Calpha coordinates, frame-major) using MDAnalysis, e.g.:"
Write-Host ""
Write-Host "    import MDAnalysis as mda"
Write-Host "    u = mda.Universe('topology.pdb', 'traj.xtc')"
Write-Host "    ca = u.select_atoms('name CA')"
Write-Host "    # write N T header, then loop frames writing ca.positions ..."
Write-Host ""
Write-Host "For a larger SYNTHETIC problem instead, run:"
Write-Host "    python scripts/make_synthetic.py --residues 200 --frames 1000"
