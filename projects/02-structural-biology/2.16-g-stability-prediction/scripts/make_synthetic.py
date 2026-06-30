#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the tiny synthetic protein sample
# ---------------------------------------------------------------------------
# Project 2.16 : Delta-Delta-G Stability Prediction (reduced-scope teaching)
#
# WHY SYNTHETIC  (CLAUDE.md §8, PATTERNS.md §6)
#   Real experimental Delta-Delta-G data (Protherm, the Megascale set) is great
#   study material but is licensed/large and needs download + parsing. For an
#   OFFLINE, deterministic demo we ship a hand-crafted miniature "protein": a
#   sequence of wild-type residues plus a per-residue BURIAL FRACTION (1 = core,
#   0 = surface). We engineer it so the saturation scan recovers an obvious,
#   checkable signal: buried hydrophobic-core positions are highly sensitive
#   (mutating them away from hydrophobic packing is strongly destabilising),
#   while flexible exposed-loop positions are nearly neutral.
#
#   This is SYNTHETIC and is labelled synthetic everywhere (data/README.md). It
#   has no relation to any real protein structure -- it exists only to make the
#   GPU lesson runnable and the result interpretable.
#
# OUTPUT FORMAT (consumed by src/reference_cpu.cpp::load_protein):
#   line 1 : <name>             single token label
#   line 2 : <L>                residue count
#   next L : <AA> <buried>      one-letter wild-type residue + burial fraction
#
# USAGE
#   python scripts/make_synthetic.py                  # default 24-residue toy
#   python scripts/make_synthetic.py --residues 40 --out data/sample/protein_sample.txt
#
# DETERMINISM: a fixed seed makes the file byte-identical on every machine, so
# demo/expected_output.txt stays valid. Re-run only if you intend to change the
# committed sample (then regenerate expected_output.txt from a real run).
# ===========================================================================
import argparse
import os
import random

# The 20 standard amino acids in one-letter code. We split them into a
# "hydrophobic core" set (good for buried positions) and a "polar/charged
# surface" set (good for exposed positions), so a designed protein looks
# physically plausible: core = hydrophobic, surface = polar.
HYDROPHOBIC = list("AILMFVWY")     # like the buried core
POLAR_CHARGED = list("RNDQEKHST")  # like the solvent-exposed surface


def build_protein(n_residues: int, seed: int):
    """Return (name, [(aa, buried), ...]) for a synthetic n_residues protein.

    Design: an alternating pattern of a buried hydrophobic 'core stripe' and an
    exposed polar 'surface stripe', mimicking the periodicity of a buried helix
    face. Burial is high (0.85-1.0) on the core stripe and low (0.0-0.2) on the
    surface stripe, with two fully-buried hydrophobic anchors planted so the most
    destabilising mutations are stable and recognisable in the demo output.
    """
    rng = random.Random(seed)
    residues = []
    for i in range(n_residues):
        # A period-4 stripe: positions 0,1 mod 4 are core-facing; 2,3 surface.
        on_core_face = (i % 4) in (0, 1)
        if on_core_face:
            # Buried core: pick a hydrophobic residue, high burial.
            aa = rng.choice(HYDROPHOBIC)
            buried = round(rng.uniform(0.85, 1.00), 3)
        else:
            # Exposed surface: pick a polar/charged residue, low burial.
            aa = rng.choice(POLAR_CHARGED)
            buried = round(rng.uniform(0.00, 0.20), 3)
        residues.append((aa, buried))

    # Plant two deterministic, fully-buried hydrophobic anchors that are
    # CRITICAL: mutating them to charged/proline is the most destabilising hit,
    # giving the demo a stable, recognisable "top mutations" list.
    if n_residues > 12:
        residues[6]  = ("L", 1.0)   # buried Leu  -> L7  : core anchor
        residues[12] = ("F", 1.0)   # buried Phe  -> F13 : core anchor
    return ("synthetic_core_helix", residues)


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic protein for the Delta-Delta-G scan.")
    ap.add_argument("--residues", type=int, default=24, help="number of residues (default 24)")
    ap.add_argument("--seed", type=int, default=216, help="RNG seed (default 216, for project 2.16)")
    here = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(here, "..", "data", "sample", "protein_sample.txt")
    ap.add_argument("--out", default=default_out, help="output path")
    args = ap.parse_args()

    name, residues = build_protein(args.residues, args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="ascii", newline="\n") as f:
        f.write(name + "\n")
        f.write(f"{len(residues)}\n")
        for aa, buried in residues:
            f.write(f"{aa} {buried:.3f}\n")
    print(f"[make_synthetic] wrote {len(residues)}-residue synthetic protein to {args.out}")


if __name__ == "__main__":
    main()
