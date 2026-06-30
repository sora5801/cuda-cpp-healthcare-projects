# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 2.19 : Membrane Protein Simulation   (reduced-scope teaching version)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL, and
# NEVER bypasses credentials/registration. This teaching project does NOT need a
# downloaded dataset -- it BUILDS its own tiny synthetic coarse-grained membrane
# patch (scripts/make_synthetic.py + the in-code build_system()). This script
# therefore only points at the real-world membrane databases the catalog names,
# for a learner who wants to go further (those need force-field setup tools and
# are far beyond this model). Nothing here is required to run the demo.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.19 -- Membrane Protein Simulation"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project runs entirely on a SYNTHETIC sample -- no download needed."
Write-Host "  Regenerate / resize the committed sample with:"
Write-Host "    python scripts/make_synthetic.py --n-lipids 32 --n-prot 7 --steps 400"
Write-Host ""
Write-Host "Real-world membrane-protein resources (for further study; not auto-fetched"
Write-Host "because they need force-field setup tools like CHARMM-GUI):"
Write-Host "  * MemProtMD  -- 3133 membrane proteins in bilayers : https://memprotmd.bioch.ox.ac.uk"
Write-Host "  * GPCRdb     -- GPCR structures and MD data         : https://gpcrdb.org"
Write-Host "  * OPM        -- orientations of proteins in membranes: https://opm.phar.umich.edu"
Write-Host "  * CGMD Platform benchmark systems                   : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7765266/"
Write-Host ""
Write-Host "  These are atomistic/structural sets. Building a runnable MD system from"
Write-Host "  them requires CHARMM-GUI Membrane Builder or packmol-memgen (see README)."
