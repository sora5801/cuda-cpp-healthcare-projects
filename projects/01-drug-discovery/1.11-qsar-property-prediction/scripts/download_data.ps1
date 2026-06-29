# ===========================================================================
# scripts/download_data.ps1  --  Point at the FULL QSAR datasets (Windows)
# ---------------------------------------------------------------------------
# Project 1.11 : QSAR / Property Prediction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. The real QSAR benchmarks ship as
# SMILES + labels and must be featurized into atom/bond graphs with RDKit, which
# is a Python step outside this C++ demo. So this script does NOT auto-download:
# it prints exactly where to get each dataset and how to convert it, and defers
# to scripts/make_synthetic.py for the offline stand-in that the demo uses.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.11 -- QSAR / Property Prediction"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The committed sample (data/sample/molecules_sample.txt) is a tiny SYNTHETIC"
Write-Host "molecule batch and is all the demo needs. To study REAL QSAR data:"
Write-Host ""
Write-Host "  MoleculeNet  (curated ML benchmarks; ESOL, FreeSolv, Lipophilicity, BBBP, ...)"
Write-Host "    https://moleculenet.org   (CSV of SMILES + labels; no login)"
Write-Host "  ChEMBL  (measured bioactivities for ~2.4M compounds)"
Write-Host "    https://www.ebi.ac.uk/chembl/   (bulk download; no login)"
Write-Host "  Therapeutics Data Commons (TDC)  (66 ready-made drug-discovery ML tasks)"
Write-Host "    https://tdcommons.ai   (pip install PyTDC; programmatic access)"
Write-Host "  PCBA  (128 PubChem BioAssays over ~440k compounds)"
Write-Host "    https://moleculenet.org"
Write-Host ""
Write-Host "  These ship as SMILES + labels. To turn them into the CSR graph format"
Write-Host "  this project reads (see data/README.md), featurize with RDKit, e.g.:"
Write-Host "    pip install rdkit pandas"
Write-Host "    # for each SMILES: atoms -> 6-dim feature rows, bonds -> edge list,"
Write-Host "    # then emit 'num_mols num_nodes num_edges' + features + counts + edges."
Write-Host ""
Write-Host "  For a larger SYNTHETIC batch without any download, run:"
Write-Host "    python scripts/make_synthetic.py"
Write-Host ""
Write-Host "  When wiring a real fetch, keep it idempotent: skip if the file exists"
Write-Host "  with the right SHA256; print URL + size + checksum; never store secrets."
