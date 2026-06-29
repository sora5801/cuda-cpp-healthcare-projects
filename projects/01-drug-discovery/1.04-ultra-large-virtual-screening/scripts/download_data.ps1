# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 1.4 : Ultra-Large Virtual Screening
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs +
# the recipe, and NEVER bypasses credentials/registration. Real screening
# libraries are huge and license-bound, so this script does NOT auto-download a
# multi-billion-compound set; it prints the RDKit recipe to turn a SMILES list
# into this project's descriptor format, and defers to make_synthetic.py for an
# offline stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.4 -- Ultra-Large Virtual Screening"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "Real virtual-screening libraries (descriptors + features):"
Write-Host "  Enamine REAL  >6B make-on-demand : https://enamine.net/compound-collections/real-compounds"
Write-Host "  ZINC20        ~2B purchasable     : https://zinc20.docking.org"
Write-Host "  ChEMBL        bioactivity ref     : https://www.ebi.ac.uk/chembl/"
Write-Host "  ExCAPE-DB     chemogenomics       : https://solr.ideaconsult.net/search/excape/"
Write-Host ""
Write-Host "These sets are huge and license-bound -- download a SMILES subset from the"
Write-Host "links above (respect each license), then compute this project's columns with"
Write-Host "RDKit (mw logp_x100 hbd hba rotb psa feat_hex):"
Write-Host ""
Write-Host "  from rdkit import Chem"
Write-Host "  from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors"
Write-Host "  m = Chem.MolFromSmiles(smiles)"
Write-Host "  mw   = round(Descriptors.MolWt(m))"
Write-Host "  logp = round(Descriptors.MolLogP(m) * 100)"
Write-Host "  hbd  = Lipinski.NumHDonors(m);  hba = Lipinski.NumHAcceptors(m)"
Write-Host "  rotb = Descriptors.NumRotatableBonds(m)"
Write-Host "  psa  = round(Descriptors.TPSA(m))"
Write-Host "  # feat = a 32-bit pharmacophore/Morgan bitmask folded to 32 bits"
Write-Host ""
Write-Host "The committed tiny sample in data/sample/ is enough to run the demo."
Write-Host "For a larger SYNTHETIC problem (no download, fully offline), run:"
Write-Host "    python scripts/make_synthetic.py --n 1000000"
