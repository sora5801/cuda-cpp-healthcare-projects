#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.26 : GPU BAM Sorting & Deduplication
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + expected
# size + checksum, and NEVER bypasses credentials/registration. The real BAM
# archives below are large and/or credentialed and cannot be redistributed here,
# so this script only PRINTS where to get them and how to convert them into our
# text format. The committed SYNTHETIC sample runs the demo offline;
# scripts/make_synthetic.py scales it up.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 3.26 -- GPU BAM Sorting & Deduplication"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "Real aligned-read datasets (BAM files). All require their own access terms;"
echo "follow each site's instructions -- this script does NOT bypass them:"
echo "  * 1000 Genomes WGS BAMs   https://www.internationalgenome.org/data"
echo "  * TCGA cancer WGS BAMs     https://portal.gdc.cancer.gov/   (controlled access)"
echo "  * ENCODE ChIP-seq BAMs     https://www.encodeproject.org/   (open)"
echo "  * ICGC PCAWG BAMs          https://dcc.icgc.org/            (controlled access)"
echo
echo "To turn a real BAM into this project's text format, extract the fields the"
echo "sort+dedup need with samtools, e.g.:"
echo '  samtools view input.bam | awk '"'"'BEGIN{OFS=" "}'
echo '    { ref=$3; pos=$4; strand=(and($2,16)?1:0); mate=$8;'
echo '      q=0; n=length($11); for(i=1;i<=n;i++) q+=substr($11,i,1);'
echo '      print ref_index, pos, strand, mate, q }'"'"
echo "  (map ref names to 0-based indices; keep pos/mate within the bit budgets"
echo "   in src/bam.h, or widen the keys as THEORY.md describes)."
echo
echo "Prepend a header line '<n> <num_refs>' (n = number of reads)."
echo
echo "OFFLINE: the committed data/sample/reads_sample.txt already runs the demo."
echo "For a bigger SYNTHETIC problem (no download):"
echo "    python scripts/make_synthetic.py --n 1048576"
