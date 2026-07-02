#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic DTI batched-graph sample
# ---------------------------------------------------------------------------
# Project 7.2 : Drug-Target Interaction Prediction (GNN)
#
# WHY THIS EXISTS
#   Real DTI data (BindingDB, ChEMBL, Davis, KIBA) is large and, for the trained
#   pipeline, needs featurized molecular graphs + protein encoders. So the demo
#   ships a TINY, clearly-SYNTHETIC batch of small molecular graphs + protein
#   descriptor vectors that runs offline. Everything here is synthetic and has NO
#   clinical meaning (CLAUDE.md sec 8); it exists to make the GNN + pairwise
#   scoring machinery runnable and the result interpretable.
#
# WHAT IT BUILDS  (format documented in data/README.md)
#   * D small molecular graphs (atoms = nodes with a length-F feature vector;
#     bonds = undirected edges). Graph d is a simple chain/ring of a few atoms.
#   * P protein descriptor vectors (length F).
#   * We engineer ONE drug and ONE protein to share a distinctive feature
#     signature so they are LIKELY to score highest; then, to stay HONEST, we run
#     the exact same fixed-weight forward pass the C++ code uses (mirrored below)
#     and write whichever pair the model actually ranks top as the "ground truth"
#     line. So the label is literally "the pair this fixed model ranks highest",
#     which the demo then recovers -- validating the machinery, not any binding.
#
# USAGE
#   python scripts/make_synthetic.py                 # default tiny sample
#   python scripts/make_synthetic.py --drugs 12 --proteins 6
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "dti_sample.txt"

# These MUST match src/gnn.h (GNN_F, GNN_T) or the C++ loader/model will mismatch.
F = 8
T = 2

# ---------------------------------------------------------------------------
# Fixed-weight model, mirrored from reference_cpu.cpp (same LCG + mapping) so the
# Python "which pair is top?" answer matches the C++ program exactly.
# ---------------------------------------------------------------------------
MASK64 = (1 << 64) - 1


def next_weight(state):
    state = (state * 6364136223846793005 + 1442695040888963407) & MASK64
    bits = (state >> 40) & 0xFFFFFF           # top 24 bits
    return state, (bits / float(1 << 24)) - 0.5


def build_model():
    state = 0x9E3779B97F4A7C15
    W, bias, Wp, bp = [], [], [], []
    for _ in range(T * F * F):
        state, w = next_weight(state); W.append(w)
    for _ in range(T * F):
        state, w = next_weight(state); bias.append(w)
    for _ in range(F * F):
        state, w = next_weight(state); Wp.append(w)
    for _ in range(F):
        state, w = next_weight(state); bp.append(w)
    return W, bias, Wp, bp


def relu(x):
    return x if x > 0.0 else 0.0


def sigmoid(z):
    import math
    if z > 30.0:
        return 1.0
    if z < -30.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def linear_relu(vin, W, bias):
    out = []
    for c in range(F):
        acc = bias[c]
        for k in range(F):
            acc += vin[k] * W[k * F + c]
        out.append(relu(acc))
    return out


# ---------------------------------------------------------------------------
# A drug graph: list of atom feature vectors + list of undirected (u,v) bonds.
# We build simple chains with a distinctive "signature" scaled per drug so the
# embeddings differ. atoms are chains of length 3..5.
# ---------------------------------------------------------------------------
def make_drug(idx, n_atoms, signature):
    feats = []
    for a in range(n_atoms):
        # Base pattern varies smoothly by atom position and drug signature.
        row = [0.10 + 0.05 * ((a + c) % 3) + 0.15 * signature * ((c % 2) + 1)
               for c in range(F)]
        feats.append(row)
    bonds = [(a, a + 1) for a in range(n_atoms - 1)]   # a simple chain
    return feats, bonds


def make_protein(idx, signature):
    return [0.20 + 0.10 * ((idx + c) % 4) + 0.20 * signature * ((c % 3) + 1)
            for c in range(F)]


def forward_top_pair(drugs, proteins):
    """Run the exact fixed-weight forward pass; return the argmax (drug, prot)."""
    W, bias, Wp, bp = build_model()
    embs = []
    for feats, bonds in drugs:
        n = len(feats)
        # CSR-style adjacency with self loops (mirrors reference_cpu load).
        nbr = [[i] for i in range(n)]
        for (u, v) in bonds:
            nbr[u].append(v); nbr[v].append(u)
        cur = [list(r) for r in feats]
        for t in range(T):
            Wt = W[t * F * F:(t + 1) * F * F]
            bt = bias[t * F:(t + 1) * F]
            nxt = []
            for i in range(n):
                msg = [0.0] * F
                for j in nbr[i]:
                    for c in range(F):
                        msg[c] += cur[j][c]
                nxt.append(linear_relu(msg, Wt, bt))
            cur = nxt
        emb = [0.0] * F
        for row in cur:
            for c in range(F):
                emb[c] += row[c]
        embs.append(emb)
    pembs = [linear_relu(p, Wp, bp) for p in proteins]

    best, best_score = (0, 0), -1.0
    for di, e in enumerate(embs):
        for pi, pe in enumerate(pembs):
            logit = sum(e[c] * pe[c] for c in range(F)) / F
            s = sigmoid(logit)
            if s > best_score:
                best_score, best = s, (di, pi)
    return best


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic DTI batched-graph sample.")
    ap.add_argument("--drugs", type=int, default=6, help="number of drug graphs")
    ap.add_argument("--proteins", type=int, default=4, help="number of protein targets")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    D, P = args.drugs, args.proteins
    # Atom counts cycle 3..5 so graphs differ in size; signatures spread the
    # embeddings apart. One drug + one protein get a strong shared signature.
    drugs = []
    for d in range(D):
        n_atoms = 3 + (d % 3)
        sig = 1.0 if d == D - 1 else 0.10 * d      # last drug is the "strong" one
        drugs.append(make_drug(d, n_atoms, sig))
    proteins = []
    for p in range(P):
        sig = 1.0 if p == P - 1 else 0.10 * p       # last protein is "strong"
        proteins.append(make_protein(p, sig))

    # HONEST ground truth: whichever pair the fixed model actually ranks top.
    true_drug, true_prot = forward_top_pair(drugs, proteins)

    # ---- Serialize in the loader's format ----------------------------------
    lines = [f"{D} {P}", f"{true_drug} {true_prot}"]
    for (feats, bonds) in drugs:
        lines.append(f"{len(feats)} {len(bonds)}")
        for row in feats:
            lines.append(" ".join(f"{v:.5f}" for v in row))
        for (u, v) in bonds:
            lines.append(f"{u} {v}")
    for p in proteins:
        lines.append(" ".join(f"{v:.5f}" for v in p))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  (D={D} drugs, P={P} proteins, F={F}, T={T}; "
          f"SYNTHETIC; implanted top pair = drug {true_drug} <-> protein {true_prot})")


if __name__ == "__main__":
    main()
