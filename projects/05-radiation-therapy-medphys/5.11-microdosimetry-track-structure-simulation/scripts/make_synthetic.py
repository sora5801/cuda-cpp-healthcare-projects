#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic sample dataset
# ---------------------------------------------------------------------------
# Project 5.11 : Microdosimetry & Track-Structure Simulation
#
# WHY THIS EXISTS
#   Real track-structure inputs (Geant4-DNA cross-section tables, measured
#   microdosimetric spectra) are large and their licensing/provenance varies.
#   For a self-contained, offline demo we generate a tiny, clearly-SYNTHETIC
#   parameter file describing one microdosimetry scenario. The C++ program reads
#   it and runs the Monte Carlo; the parameters here are physically MOTIVATED
#   (nm-scale water box, DNA proximity, LET spread) but NOT validated radiobiology.
#
# THE ONE-LINE FORMAT (whitespace-separated, order matters; see data/README.md):
#   box_nm sigma_ion let_spread quantum_eV quanta_per_ion p_delta delta_quanta
#   dna_radius_nm n_dna_segments n_y_bins y_max_keV_um n_tracks seed
#
#   The default scenario is a moderately HIGH-LET, MIXED field: sigma_ion=1.0 /nm
#   mean ionization density with a lognormal LET spread (let_spread=0.5) so the
#   lineal-energy spectrum has a realistic high-y tail, DNA within 3 nm of the
#   track scores strand breaks, and clustered breaks form DSBs. Tuned to give an
#   interpretable result: a broad f(y), SSB/DSB ~ 3, and DSB/track ~ 0.08.
#
# USAGE
#   python scripts/make_synthetic.py                 # writes the default sample
#   python scripts/make_synthetic.py --n-tracks 20000 --let-spread 0.8
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "track_params.txt"


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic microdosimetry sample.")
    ap.add_argument("--box-nm", type=float, default=100.0, help="cubic water box side (nm)")
    ap.add_argument("--sigma-ion", type=float, default=1.0, help="mean ionizations per nm (LET class)")
    ap.add_argument("--let-spread", type=float, default=0.5, help="lognormal LET spread (mixed field)")
    ap.add_argument("--quantum-ev", type=float, default=30.0, help="energy per quantum (eV)")
    ap.add_argument("--quanta-per-ion", type=int, default=2, help="quanta per primary ionization")
    ap.add_argument("--p-delta", type=float, default=0.5, help="prob. an ionization emits a delta ray")
    ap.add_argument("--delta-quanta", type=int, default=3, help="quanta in a delta-ray cluster")
    ap.add_argument("--dna-radius-nm", type=float, default=3.0, help="strand-break capture radius (nm)")
    ap.add_argument("--n-dna-segments", type=int, default=25, help="DNA scoring segments along y")
    ap.add_argument("--n-y-bins", type=int, default=12, help="lineal-energy histogram bins")
    ap.add_argument("--y-max", type=float, default=480.0, help="upper edge of y histogram (keV/um)")
    ap.add_argument("--n-tracks", type=int, default=4000, help="number of primary tracks")
    ap.add_argument("--seed", type=int, default=20240517, help="base RNG seed")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    # Emit the fields in the exact order the C++ loader expects. Using %g keeps
    # the line compact; integers stay integral so the loader parses them cleanly.
    fields = [
        f"{args.box_nm:g}", f"{args.sigma_ion:g}", f"{args.let_spread:g}",
        f"{args.quantum_ev:g}", f"{args.quanta_per_ion:d}", f"{args.p_delta:g}",
        f"{args.delta_quanta:d}", f"{args.dna_radius_nm:g}", f"{args.n_dna_segments:d}",
        f"{args.n_y_bins:d}", f"{args.y_max:g}", f"{args.n_tracks:d}", f"{args.seed:d}",
    ]
    # The C++ loader parses bare whitespace-separated numbers (ifstream >> ...),
    # so the file itself carries no comment header -- the field meanings live in
    # data/README.md. We still LABEL the data as synthetic there and in every doc.
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(" ".join(fields) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  (n_tracks={args.n_tracks}; SYNTHETIC)")


if __name__ == "__main__":
    main()
