#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic phantom sample dataset
# ---------------------------------------------------------------------------
# Project 5.10 : Secondary Cancer Risk & Stray-Dose Monte Carlo
#
# WHY THIS EXISTS
#   The real inputs for stray-dose MC are 3-D ICRP-110 voxel phantoms and NIST
#   photon cross-section tables (see data/README.md). Those are large and/or
#   redistribution-restricted, so this script deterministically generates a
#   clearly-SYNTHETIC, reduced 1-D "organ-stack" phantom that matches the loader
#   (reference_cpu.cpp::load_stray_problem) and makes the demo runnable offline.
#   Synthetic data is always LABELED synthetic (CLAUDE.md section 8).
#
# THE FILE FORMAT (read by load_stray_problem)
#   Line 1 (11 numbers):
#     field_end mu organ_cm scatter_frac sidescatter leakage_frac neutron_frac
#     roulette_floor roulette_survive n_histories seed
#   Then one line per organ:  "<name> <risk_coeff>"
#   n_organs is inferred from the number of organ lines.
#
# THE ENGINEERED SAMPLE (so the result is interpretable, PATTERNS.md section 6)
#   organ 0 = "Target" is in-field (field_end=1) and gets the large primary dose;
#   organs 1..N receive ONLY stray dose (scatter + leakage + neutron surrogate),
#   which falls off with distance -- exactly the out-of-field pattern the science
#   predicts. Risk coefficients are illustrative BEIR-VII-style values ordered so
#   radiosensitive organs (marrow, colon, lung) contribute more risk per unit dose.
#
# USAGE
#   python scripts/make_synthetic.py                 # writes data/sample/phantom.txt
#   python scripts/make_synthetic.py --histories 400000 --seed 7
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "phantom.txt"

# Reduced 1-D organ stack (head -> foot). The first entry is the treated target.
# risk_coeff = illustrative relative cancer-risk sensitivity (NOT clinical).
ORGANS = [
    ("Target",     0.00),   # treated volume; secondary-cancer risk in-field is not summed
    ("RedMarrow",  1.20),   # highly radiosensitive
    ("Colon",      1.00),
    ("Lung",       0.85),
    ("Stomach",    0.70),
    ("Bladder",    0.55),
    ("Breast",     0.80),
    ("Thyroid",    0.40),
    ("Skin",       0.10),   # distal, low sensitivity
]


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic 1-D phantom sample.")
    ap.add_argument("--histories", type=int, default=200000,
                    help="number of primary photon histories")
    ap.add_argument("--seed", type=int, default=12345, help="base RNG seed")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    # Physics + variance-reduction parameters (documented in data/README.md).
    field_end        = 1        # only organ 0 is in-field
    mu               = 0.070    # tissue attenuation (1/cm), ~ high-energy photons
    organ_cm         = 8.0      # slab thickness per organ (cm)
    scatter_frac     = 0.90     # 90% of an interaction's weight scatters (survives)
    sidescatter      = 8.0e-4   # forced-detection lateral-scatter fraction per organ
    leakage_frac     = 6.0e-5   # machine-head leakage per primary (uniform)
    neutron_frac     = 2.0e-5   # secondary-neutron surrogate (distance-weighted)
    roulette_floor   = 1.0e-3   # play roulette below this weight
    roulette_survive = 0.25     # survival probability in roulette

    header = (f"{field_end} {mu:g} {organ_cm:g} {scatter_frac:g} {sidescatter:g} "
              f"{leakage_frac:g} {neutron_frac:g} {roulette_floor:g} "
              f"{roulette_survive:g} {args.histories} {args.seed}")

    lines = [
        "# Project 5.10 SYNTHETIC 1-D phantom (NOT clinical). "
        "Fields: field_end mu organ_cm scatter_frac sidescatter leakage_frac "
        "neutron_frac roulette_floor roulette_survive n_histories seed",
        header,
        "# organ  risk_coeff  (illustrative BEIR-VII-style sensitivities)",
    ]
    for name, rc in ORGANS:
        lines.append(f"{name} {rc:g}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(organs={len(ORGANS)}, histories={args.histories}, seed={args.seed}; SYNTHETIC)")


if __name__ == "__main__":
    main()
