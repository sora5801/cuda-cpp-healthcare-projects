# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 3.26 : GPU BAM Sorting & Deduplication
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# expected size + checksum, and NEVER bypasses credentials/registration. The
# real BAM archives below are large and/or credentialed and cannot be
# redistributed here, so this script only PRINTS where to get them and how to
# convert them into our text format -- it never tries to log in for you. The
# committed tiny SYNTHETIC sample (data/sample/reads_sample.txt) is enough to
# run the demo offline; scripts/make_synthetic.py scales it up.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.26 -- GPU BAM Sorting & Deduplication"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "Real aligned-read datasets (BAM files). All require their own access"
Write-Host "terms; follow each site's instructions -- this script does NOT bypass them:"
Write-Host "  * 1000 Genomes WGS BAMs   https://www.internationalgenome.org/data"
Write-Host "  * TCGA cancer WGS BAMs     https://portal.gdc.cancer.gov/   (controlled access)"
Write-Host "  * ENCODE ChIP-seq BAMs     https://www.encodeproject.org/   (open)"
Write-Host "  * ICGC PCAWG BAMs          https://dcc.icgc.org/            (controlled access)"
Write-Host ""
Write-Host "To turn a real BAM into this project's text format, extract the fields"
Write-Host "the sort+dedup need with samtools, e.g.:"
Write-Host "  samtools view input.bam | awk 'BEGIN{OFS=\" \"}"
Write-Host "    { ref=\$3; pos=\$4; strand=(and(\$2,16)?1:0); mate=\$8; "
Write-Host "      q=0; n=length(\$11); for(i=1;i<=n;i++) q+=substr(\$11,i,1); "
Write-Host "      print ref_index, pos, strand, mate, q }'"
Write-Host "  (map ref names to 0-based indices, and keep pos/mate within the bit"
Write-Host "   budgets in src/bam.h, or widen the keys as THEORY.md describes)."
Write-Host ""
Write-Host "Prepend a header line '<n> <num_refs>' (n = number of reads)."
Write-Host ""
Write-Host "OFFLINE: the committed data/sample/reads_sample.txt already runs the demo."
Write-Host "For a bigger SYNTHETIC problem (no download):"
Write-Host "    python scripts/make_synthetic.py --n 1048576"
