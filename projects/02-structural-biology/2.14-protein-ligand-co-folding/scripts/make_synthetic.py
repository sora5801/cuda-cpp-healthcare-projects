#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Write a synthetic protein-ligand complex
# ---------------------------------------------------------------------------
# Project 2.14 : Protein-Ligand Co-Folding (reduced-scope teaching version)
#
# WHAT THIS WRITES (and why it is SYNTHETIC, not a real PDB structure)
#   This repo is study material; the committed sample must run offline and carry
#   a KNOWN answer so the demo is verifiable (PATTERNS.md §6). So we hand-build a
#   tiny "complex": a ring of protein backbone (C-alpha) tokens forming a binding
#   pocket, plus a few ligand heavy-atom tokens sitting inside it. These NATIVE
#   coordinates x* are the planted answer. The program noises them to x_T and the
#   reverse diffusion must fold them back -> recovered RMSD ~ 0.
#
#   This is NOT a real protein and NOT a real ligand pose. No clinical meaning.
#   For real complexes, see scripts/download_data.* (PoseBusters / PDBbind).
#
# OUTPUT FORMAT (one token per line; see data/README.md):
#   line 1 : n_protein n_ligand steps temp step_frac type_bias seed noise_scale
#   line k : type x* y* z*        (type 0 = protein backbone, 1 = ligand atom)
#
# USAGE
#   python scripts/make_synthetic.py
#   python scripts/make_synthetic.py --n-protein 16 --noise 1.5 --steps 200
# ===========================================================================
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "complex_sample.txt"

TYPE_PROTEIN = 0
TYPE_LIGAND = 1


def build_complex(n_protein, n_ligand, pocket_radius):
    """Return (types, coords) for the planted native complex.

    Protein tokens lie on a circle (the pocket rim) in the z=0 plane; ligand
    tokens sit in a small cluster near the pocket center, slightly above the
    plane. Deterministic -- no randomness here (the noise is added at run time
    by the C++ init_positions for reproducibility)."""
    types, coords = [], []

    # Protein backbone ring: a crude binding pocket. Evenly spaced on a circle.
    for i in range(n_protein):
        ang = 2.0 * math.pi * i / n_protein
        x = pocket_radius * math.cos(ang)
        y = pocket_radius * math.sin(ang)
        z = 0.0
        types.append(TYPE_PROTEIN)
        coords.append((x, y, z))

    # Ligand atoms: a WELL-SPACED zig-zag chain threaded across the pocket and
    # lifted in z so the pose is genuinely 3-D. We space the atoms ~1.8 A apart
    # (centered on the origin) so each -- including the two chain ENDPOINTS --
    # has a DISTINCT geometric neighbourhood. Geometric (distance-kernel)
    # attention can then resolve every atom individually; a tightly-clustered
    # ligand would instead average to its centroid, a real failure mode we
    # deliberately avoid here and discuss in THEORY "Numerical considerations".
    dx = 1.8                              # inter-atom spacing along x (Angstrom)
    x0 = -dx * (n_ligand - 1) / 2.0       # center the chain on x = 0
    for j in range(n_ligand):
        x = x0 + dx * j                   # walk across the pocket in x
        y = 0.5 * (1 if j % 2 == 0 else -1)   # alternate +/- in y (zig-zag)
        z = 1.0 + 0.8 * j                 # rise in z (out of the pocket plane)
        types.append(TYPE_LIGAND)
        coords.append((x, y, z))

    return types, coords


def main():
    ap = argparse.ArgumentParser(description="Write a synthetic protein-ligand complex.")
    ap.add_argument("--n-protein", type=int, default=12, help="protein backbone tokens (pocket rim)")
    ap.add_argument("--n-ligand", type=int, default=5, help="ligand heavy-atom tokens")
    ap.add_argument("--pocket-radius", type=float, default=4.0, help="pocket rim radius (Angstrom)")
    ap.add_argument("--steps", type=int, default=160, help="reverse-diffusion (denoising) steps")
    ap.add_argument("--temp", type=float, default=0.5, help="attention bandwidth (smaller=sharper)")
    ap.add_argument("--step-frac", type=float, default=0.1, help="DDIM step fraction in (0,1]")
    ap.add_argument("--type-bias", type=float, default=1.0, help="same-type attention bonus")
    ap.add_argument("--seed", type=int, default=20260628, help="RNG seed for the initial noise")
    ap.add_argument("--noise", type=float, default=1.0, help="std-dev of the start noise (Angstrom)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    types, coords = build_complex(args.n_protein, args.n_ligand, args.pocket_radius)

    lines = []
    # Header: dimensions + the diffusion schedule, in the C++ loader's order.
    lines.append(f"{args.n_protein} {args.n_ligand} {args.steps} {args.temp:g} "
                 f"{args.step_frac:g} {args.type_bias:g} {args.seed} {args.noise:g}")
    # One row per token: type then the native x* y* z* (4 decimals, deterministic).
    for t, (x, y, z) in zip(types, coords):
        lines.append(f"{t} {x:.4f} {y:.4f} {z:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {out}  "
          f"({args.n_protein} protein + {args.n_ligand} ligand tokens, "
          f"{args.steps} steps, noise={args.noise} A)  [SYNTHETIC -- not a real structure]")


if __name__ == "__main__":
    main()
