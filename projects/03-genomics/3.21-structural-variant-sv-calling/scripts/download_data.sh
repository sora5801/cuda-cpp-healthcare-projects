#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Pointers to real SV benchmarks (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.21 : Structural Variant (SV) Calling
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. This teaching project ships a SYNTHETIC
# sample (data/sample/sv_sample.txt) and does not require any download to run the
# demo. This script only points at the real gold-standard benchmarks and defers
# to scripts/make_synthetic.py for a larger offline stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 3.21 -- Structural Variant (SV) Calling"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project runs entirely on the committed SYNTHETIC sample:"
echo "    data/sample/sv_sample.txt   (one planted deletion; demo needs no download)"
echo
echo "For a LARGER synthetic problem (offline, no credentials):"
echo "    python scripts/make_synthetic.py --reads 200000"
echo
echo "Real, gold-standard SV benchmarks (study these; respect each license):"
echo "  * GiaB HG002 SV benchmark (NIST) -- deletion/insertion/inversion truth set:"
echo "      https://www.nist.gov/programs-projects/genome-bottle"
echo "      (Tier-1 VCF + BED; download via the GiaB FTP/S3 mirrors linked there.)"
echo "  * PacBio sv-benchmark -- HiFi long-read SV truth + tooling:"
echo "      https://github.com/PacificBiosciences/sv-benchmark"
echo "  * 1000 Genomes structural-variant catalog:"
echo "      https://www.internationalgenome.org/data"
echo "  * ENCODE long-read SV studies:"
echo "      https://www.encodeproject.org/"
echo
echo "These are BAM/CRAM + VCF, not the toy text format this teaching demo parses."
echo "When wiring a real dataset, follow this idempotent pattern:"
echo "  1) skip the download if the file already exists with the right SHA256"
echo "  2) print the source URL + expected size + checksum before fetching"
echo "  3) for any credentialed set, print registration instructions ONLY -- never"
echo "     attempt to bypass authentication (CLAUDE.md §8)."
