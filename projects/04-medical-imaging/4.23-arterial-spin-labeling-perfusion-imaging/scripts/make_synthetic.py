#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write the synthetic multi-delay ASL study
# ---------------------------------------------------------------------------
# Project 4.23 : Arterial Spin Labeling & Perfusion Imaging
#
# WHAT THIS GENERATES
#   A tiny, fully-SYNTHETIC multi-delay ASL dataset the demo fits. For each voxel
#   we pick a GROUND-TRUTH (CBF, ATT) spanning physiologically plausible tissue
#   (grey matter ~60, white matter ~20 mL/100g/min; transit times ~0.5-1.5 s),
#   then evaluate the NOISE-FREE Buxton kinetic curve delta-M(PLD) at the shared
#   PLD schedule. Because the curves are noise-free, a converged Gauss-Newton fit
#   must recover the ground truth -- that "embed a known answer" design
#   (docs/PATTERNS.md §6) is what makes the demo's result interpretable and
#   verifiable.
#
#   The Buxton model here is a BYTE-FOR-BYTE port of asl.h::asl_buxton with the
#   consensus constants from asl.h::asl_default_constants -- so the synthesized
#   signal and the C++ fit's forward model agree exactly (up to float printing).
#
# OUTPUT FORMAT (data/README.md), a small text file:
#   line 1:  n_voxels  n_plds  max_iters  f_init  att_init
#   line 2:  pld_0 ... pld_{n_plds-1}
#   then n_voxels lines:  true_cbf  true_att  s_0 ... s_{n_plds-1}
#
# USAGE
#   python scripts/make_synthetic.py                 # default 6-voxel study
#   python scripts/make_synthetic.py --voxels 100000 # a bigger synthetic map
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "asl_sample.txt"

# --- Acquisition constants: MUST mirror asl.h::asl_default_constants() ---
T1_BLOOD = 1.65   # s
ALPHA    = 0.85   # pCASL inversion efficiency
LAMBDA   = 0.90   # mL/g water partition coefficient
TAU      = 1.80   # s labeling duration
M0       = 1.0    # MR units


def buxton(pld, f_phys, att):
    """Noise-free pCASL Buxton delta-M at one PLD. Mirrors asl.h::asl_buxton."""
    f = f_phys / 6000.0          # mL/100g/min -> 1/s
    if pld < att:                # regime (A): labeled blood not yet arrived
        return 0.0
    q = 2.0 * ALPHA * M0 * (f / LAMBDA) * T1_BLOOD
    decay_transit = math.exp(-att / T1_BLOOD)
    if pld < att + TAU:          # regime (B): bolus still arriving
        return q * decay_transit * (1.0 - math.exp(-(pld - att) / T1_BLOOD))
    # regime (C): full bolus delivered, blood-T1 decay only
    return (q * decay_transit
              * math.exp(-(pld - att - TAU) / T1_BLOOD)
              * (1.0 - math.exp(-TAU / T1_BLOOD)))


def default_voxels():
    """Six named tissue types with plausible (CBF, ATT). Recovered in the demo."""
    return [
        # (true_cbf mL/100g/min, true_att s)   -- description
        (60.0, 0.70),   # cortical grey matter, short transit
        (55.0, 1.00),   # grey matter, longer transit
        (22.0, 1.20),   # white matter, low flow
        (18.0, 1.40),   # deep white matter, long transit
        (80.0, 0.50),   # highly perfused grey, fast arrival
        (40.0, 0.90),   # mixed / border-zone tissue
    ]


def main():
    ap = argparse.ArgumentParser(description="Write the synthetic ASL study.")
    ap.add_argument("--voxels", type=int, default=0,
                    help="if >0, generate this many voxels by tiling the 6 defaults")
    ap.add_argument("--max-iters", type=int, default=30, help="Gauss-Newton cap")
    ap.add_argument("--f-init", type=float, default=30.0, help="initial CBF guess")
    ap.add_argument("--att-init", type=float, default=0.70, help="initial ATT guess")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    # Multi-delay PLD schedule (s): a typical 7-delay pCASL sampling that brackets
    # the transit times so both CBF (amplitude) and ATT (onset) are identifiable.
    plds = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]

    base = default_voxels()
    if args.voxels > 0:
        voxels = [base[i % len(base)] for i in range(args.voxels)]
    else:
        voxels = base

    lines = []
    lines.append(f"{len(voxels)} {len(plds)} {args.max_iters} "
                 f"{args.f_init:g} {args.att_init:g}")
    lines.append(" ".join(f"{p:g}" for p in plds))
    for (cbf, att) in voxels:
        sig = [buxton(p, cbf, att) for p in plds]
        # Fixed precision so the file is compact and reproducible.
        lines.append(f"{cbf:g} {att:g} " + " ".join(f"{s:.10g}" for s in sig))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"({len(voxels)} voxels, {len(plds)} PLDs, noise-free Buxton curves)")


if __name__ == "__main__":
    main()
