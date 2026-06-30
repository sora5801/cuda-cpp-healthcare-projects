# ===========================================================================
# scripts/download_data.ps1  --  Fetch / point at the real datasets (Windows)
# ---------------------------------------------------------------------------
# Project 2.30 : Protein Solubility & Phase Separation Simulation
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. The LLPS resources below are sequence/
# annotation DATABASES, not ready-to-run particle configurations -- so there is
# no single binary to download for this simulation. This script prints where the
# real data lives (for the curious) and defers to make_synthetic.py for the
# offline, runnable coarse-grained system the demo uses.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.30 -- Protein Solubility & Phase Separation Simulation"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "  Real-world LLPS / IDP resources (databases, not simulation inputs):"
Write-Host "    * PhaSePro   -- proteins undergoing LLPS    https://phasepro.elte.hu"
Write-Host "    * PhaSepDB   -- phase-separation database    http://db.phasep.pro"
Write-Host "    * DisProt    -- intrinsically disordered     https://disprot.org"
Write-Host "    * FuzDB      -- fuzzy protein complexes       https://fuzdb.org"
Write-Host "    * CALVADOS   -- residue-level IDP force field https://github.com/KULL-Centre/CALVADOS"
Write-Host ""
Write-Host "  These give SEQUENCES and per-residue stickiness scales (Kapcha-Rossky,"
Write-Host "  HPS, CALVADOS). To build a runnable system from a sequence you map each"
Write-Host "  residue to its lambda, place beads on a chain, and feed the loader format"
Write-Host "  in data/README.md -- exactly what make_synthetic.py does with synthetic"
Write-Host "  stickiness values."
Write-Host ""
Write-Host "  The committed tiny sample in data/sample/system.txt runs the demo offline."
Write-Host "  For a larger SYNTHETIC system, run:"
Write-Host "    python scripts/make_synthetic.py --chains 16 --len 12 --box 12.0"
