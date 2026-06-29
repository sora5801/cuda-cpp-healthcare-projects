#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic SMILES training corpus
# ---------------------------------------------------------------------------
# Project 1.10 -- De Novo Generative Molecular Design  (REDUCED-SCOPE teaching
#                 version; see ../THEORY.md "Where this sits in the real world").
#
# WHY THIS EXISTS
#   Real generative-design corpora (ChEMBL, ZINC20, MOSES, GuacaMol) are large
#   and license-restricted (ChEMBL is CC-BY-SA; ZINC20 is free for academic use
#   but huge). For a teaching demo that runs offline we DO NOT need them: a
#   first-order Markov "language model" can be trained on a HANDFUL of SMILES and
#   still demonstrate the whole de-novo pipeline (learn distribution -> sample
#   novel strings -> score -> goal-directed top-K). So this script writes a tiny,
#   clearly-SYNTHETIC corpus of valid-ish SMILES fragments.
#
#   The SMILES below are hand-written toy strings over a deliberately small
#   alphabet so the demo is interpretable and the learned Markov table is small
#   enough to print and reason about. They are NOT real drugs and carry NO
#   clinical meaning -- they exist only to seed a character-transition model.
#
# THE FILE FORMAT (what src/reference_cpu.cpp's loader expects)
#   Line 1 : a header of three integers  "n_train n_generate seed"
#              n_train     = number of training SMILES lines that follow
#              n_generate  = how many novel molecules to sample on CPU+GPU
#              seed        = base RNG seed (so the run is reproducible)
#   Next n_train lines : one training SMILES string each (no spaces).
#   Everything is ASCII; '#' lines and blank lines are ignored by the loader.
#
# USAGE
#   python scripts/make_synthetic.py                 # writes the committed sample
#   python scripts/make_synthetic.py --n-generate 65536 --seed 7   # bigger run
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "smiles_corpus_sample.txt"

# ---------------------------------------------------------------------------
# A tiny SYNTHETIC training corpus of toy SMILES strings.
#   These use only the alphabet the model understands (see generator.h:
#   the symbol table C O N c n ( ) = 1 2 and the special END token). They are
#   intentionally simple chains/rings so a FIRST-ORDER Markov model produces a
#   healthy mix of valid-looking molecules. They are SYNTHETIC and illustrative;
#   do not read chemistry into the exact strings.
# ---------------------------------------------------------------------------
TRAIN_SMILES = [
    "CCO",            # ethanol-like chain
    "CCCC",           # butane-like chain
    "CCN",            # amine chain
    "CCOC",           # ether
    "CC(C)C",         # branched
    "c1ccccc1",       # benzene ring
    "CCc1ccccc1",     # ethylbenzene-like
    "CC(=O)O",        # carboxylic-acid-like
    "CCOCC",          # diethyl-ether-like
    "CN(C)C",         # trimethylamine-like
    "C1CCCCC1",       # cyclohexane-like
    "CCc1ccncc1",     # pyridine-substituted
    "CCC(=O)N",       # amide-like
    "OCC(O)CO",       # polyol-like
    "CCOc1ccccc1",    # phenetole-like
    "CC(C)(C)O",      # tert-butanol-like
]


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic SMILES corpus.")
    ap.add_argument("--n-generate", type=int, default=4096,
                    help="how many novel molecules to sample at run time")
    ap.add_argument("--seed", type=int, default=12345,
                    help="base RNG seed (reproducible generation)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    n_train = len(TRAIN_SMILES)
    lines = []
    lines.append("# SYNTHETIC SMILES training corpus -- project 1.10 (teaching only).")
    lines.append("# Format: header 'n_train n_generate seed', then n_train SMILES lines.")
    lines.append(f"{n_train} {args.n_generate} {args.seed}")
    lines.extend(TRAIN_SMILES)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}")
    print(f"[make_synthetic]   n_train={n_train}  n_generate={args.n_generate}  "
          f"seed={args.seed}  (SYNTHETIC, teaching only)")


if __name__ == "__main__":
    main()
