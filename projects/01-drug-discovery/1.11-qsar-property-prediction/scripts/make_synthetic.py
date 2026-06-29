#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic molecule-batch sample
# ---------------------------------------------------------------------------
# Project 1.11 : QSAR / Property Prediction
#
# WHY THIS EXISTS
#   The real QSAR benchmarks (MoleculeNet, ChEMBL, TDC; see data/README.md) need
#   RDKit to featurize SMILES into atom/bond graphs, and some carry licenses we
#   will not redistribute. So the committed sample is a small batch of CLEARLY
#   SYNTHETIC "molecules": graphs whose atoms carry a 6-dim feature vector and
#   whose bonds form simple chemical-looking topologies (chains, rings, a star).
#   The features are NOT real chemistry and the predicted property is a synthetic
#   demonstration number -- never a real ADMET/activity value.
#
# THE FILE FORMAT  (parsed by src/reference_cpu.cpp::load_graph)
#   line 1            : num_mols num_nodes num_edges        (edges = REAL bonds,
#                       self-loops are added by the loader, not listed here)
#   next num_nodes    : GCN_F_IN(=6) feature floats per atom (one atom per line)
#   next num_mols     : atom_count for each molecule (must sum to num_nodes)
#   next num_edges    : "u v" undirected bond, GLOBAL node indices (0-based)
#
# THE 6 ATOM FEATURES (a tiny, didactic descriptor vector)
#   f0 : 1.0 if the atom is "carbon-like"     (one-hot element bit)
#   f1 : 1.0 if the atom is "nitrogen-like"   (one-hot element bit)
#   f2 : 1.0 if the atom is "oxygen-like"     (one-hot element bit)
#   f3 : normalized local degree (#bonds / 4) -- a cheap topological descriptor
#   f4 : 1.0 if the atom is in a ring          (synthetic ring flag)
#   f5 : a constant 1.0 bias channel           (lets the net learn an offset)
#   Real featurizers use dozens of such bits (hybridization, charge, aromaticity,
#   H-count, chirality, ...). Six keeps the sample human-readable.
#
# DETERMINISM
#   Everything here is fixed (no RNG), so re-running reproduces byte-identical
#   data and thus a byte-identical demo/expected_output.txt.
#
# USAGE
#   python scripts/make_synthetic.py            # writes data/sample/molecules_sample.txt
#   python scripts/make_synthetic.py --out other.txt
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "molecules_sample.txt"

# Element one-hot helpers (C, N, O). Index: 0=C, 1=N, 2=O.
def elem_onehot(e):
    v = [0.0, 0.0, 0.0]
    v[e] = 1.0
    return v


def build_molecules():
    """Return a list of molecules. Each molecule is a dict with:
         'elems' : list of element codes (0=C,1=N,2=O) per atom
         'ring'  : list of 0/1 ring flags per atom
         'bonds' : list of (i,j) LOCAL atom-index bonds (0-based within the mol)
    The topologies are chosen to look chemically plausible and to give a clear
    spread of predicted properties (chains vs rings vs branched stars)."""
    mols = []

    # mol 0: a 4-atom carbon chain  C-C-C-C  (like butane).
    mols.append({
        "elems": [0, 0, 0, 0],
        "ring":  [0, 0, 0, 0],
        "bonds": [(0, 1), (1, 2), (2, 3)],
    })

    # mol 1: a 6-membered carbon ring (like cyclohexane / benzene skeleton).
    mols.append({
        "elems": [0, 0, 0, 0, 0, 0],
        "ring":  [1, 1, 1, 1, 1, 1],
        "bonds": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
    })

    # mol 2: a small N/O-containing chain  N-C-C-O  (like ethanolamine).
    mols.append({
        "elems": [1, 0, 0, 2],
        "ring":  [0, 0, 0, 0],
        "bonds": [(0, 1), (1, 2), (2, 3)],
    })

    # mol 3: a branched "star"  central C bonded to 3 outer atoms (C,N,O).
    mols.append({
        "elems": [0, 0, 1, 2],
        "ring":  [0, 0, 0, 0],
        "bonds": [(0, 1), (0, 2), (0, 3)],
    })

    # mol 4: a 5-membered ring with one nitrogen (like pyrrolidine skeleton).
    mols.append({
        "elems": [1, 0, 0, 0, 0],
        "ring":  [1, 1, 1, 1, 1],
        "bonds": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
    })

    return mols


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic molecule batch.")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    mols = build_molecules()

    # Flatten to global node indices and compute per-atom local degree.
    atom_counts = [len(m["elems"]) for m in mols]
    starts = [0]
    for c in atom_counts:
        starts.append(starts[-1] + c)
    num_nodes = starts[-1]
    num_mols = len(mols)

    # local degree per atom (count REAL bonds; loader adds the self-loop later).
    feats = []          # one [6] vector per global atom
    edges = []          # global (u,v) bonds
    for mi, m in enumerate(mols):
        base = starts[mi]
        deg = [0] * len(m["elems"])
        for (i, j) in m["bonds"]:
            deg[i] += 1
            deg[j] += 1
            edges.append((base + i, base + j))
        for a in range(len(m["elems"])):
            e = m["elems"][a]
            f = elem_onehot(e)                      # f0..f2 element one-hot
            f.append(deg[a] / 4.0)                  # f3 normalized degree
            f.append(float(m["ring"][a]))           # f4 ring flag
            f.append(1.0)                           # f5 bias channel
            feats.append(f)

    # Emit the text format.
    lines = [f"{num_mols} {num_nodes} {len(edges)}"]
    for f in feats:
        lines.append(" ".join(f"{v:g}" for v in f))
    for c in atom_counts:
        lines.append(str(c))
    for (u, v) in edges:
        lines.append(f"{u} {v}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"({num_mols} molecules, {num_nodes} atoms, {len(edges)} bonds; SYNTHETIC)")


if __name__ == "__main__":
    main()
