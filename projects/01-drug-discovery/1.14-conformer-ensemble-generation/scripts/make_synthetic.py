#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic run-parameter sample
# ---------------------------------------------------------------------------
# Project 1.14 : Conformer Ensemble Generation
#
# WHAT THIS WRITES (and why it is tiny)
#   The molecule itself is fixed at COMPILE TIME in src/conformer.h (a short
#   flexible chain). The program enumerates every conformer of that molecule
#   internally, so the only thing a "dataset" needs to supply is a couple of RUN
#   PARAMETERS for the RMSD-pruning step:
#
#       <rmsd_threshold_angstrom>   <num_representatives_to_print>
#
#   That is what this script emits to data/sample/conformer_params.txt. It is
#   SYNTHETIC by construction -- there is no patient or proprietary data here, just
#   two demo knobs -- and it is labelled synthetic in data/README.md.
#
#   (The real-world datasets for this topic -- GEOM, the CSD torsion library, COD --
#   are large and/or license-restricted; scripts/download_data.* point you to them.
#   They are NOT needed to run this teaching demo.)
#
# USAGE
#   python scripts/make_synthetic.py                       # default 0.50 A, 5 reps
#   python scripts/make_synthetic.py --rmsd 0.75 --top 8   # custom knobs
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "conformer_params.txt"


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic conformer run parameters.")
    ap.add_argument("--rmsd", type=float, default=1.00,
                    help="RMSD threshold in Angstrom: conformers closer than this "
                         "are treated as duplicates and pruned (1.0 A is a typical "
                         "RDKit conformer-dedup value)")
    ap.add_argument("--top", type=int, default=5,
                    help="how many lowest-energy representatives to print")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    # The file is two whitespace-separated numbers, matching load_params() in
    # src/main.cu. A leading comment line is allowed because the C++ reader skips
    # non-numeric tokens... but to keep the loader dead-simple we emit numbers only.
    text = f"{args.rmsd:g} {args.top}\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(rmsd_threshold={args.rmsd} A, top={args.top}; SYNTHETIC)")


if __name__ == "__main__":
    main()
