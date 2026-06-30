#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic variant sample
# ---------------------------------------------------------------------------
# Project 3.19 : Variant Effect / Pathogenicity Prediction
#
# WHY THIS EXISTS
#   Real variant-effect benchmarks (ClinVar labels, gnomAD constraint, DMS
#   atlases) are large and partly access-controlled (see data/README.md). For an
#   OFFLINE, reproducible demo we generate a tiny, clearly-SYNTHETIC variant
#   batch that the loader (reference_cpu.cpp::load_variants) understands and that
#   makes the toy model's ranking interpretable.
#
#   THE TRICK (PATTERNS.md sec 6: engineer a known answer into the sample)
#   The shipped fixed model (reference_cpu.cpp::init_model) plants two motifs:
#       * DELETERIOUS 5-mer  C A G C T  -> strong POSITIVE delta when CREATED
#       * PROTECTIVE  5-mer  T A T A T  -> strong NEGATIVE delta when CREATED
#   We craft variants whose ALTERNATE allele, at the centre base, COMPLETES one
#   of these motifs in the context window -- so the demo's "most pathogenic"
#   ranking recovers exactly the variants that build the deleterious motif, and
#   the protective ones sink to the bottom. Everything else is neutral filler.
#
#   This is SYNTHETIC teaching data: the model is untrained and the motifs are
#   invented. Nothing here is biologically real or clinically meaningful (sec 8).
#
# FILE FORMAT written (matches load_variants):
#   line 1 : "<n> <window_width>"
#   next n : "<pos> <REF> <ALT> <WINDOW>"   ; WINDOW = window_width A/C/G/T letters,
#                                             the REFERENCE context; its centre
#                                             base equals <REF>.
#
# USAGE
#   python scripts/make_synthetic.py                 # writes the committed sample
#   python scripts/make_synthetic.py --n 64 --seed 7 # a bigger synthetic batch
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "variants_sample.txt"

# These constants MUST match vep_model.h.
WINDOW = 21            # VEP_WINDOW : context length (odd; centre at index 10)
CENTER = WINDOW // 2   # VEP_CENTER : the variant locus inside the window
KWIDTH = 5             # VEP_KWIDTH : motif / filter width

BASES = "ACGT"
DEL_MOTIF = "CAGCT"    # planted deleterious 5-mer (matches init_model filter 0)
BEN_MOTIF = "TATAT"    # planted protective 5-mer (matches init_model filter 1)


def random_window(rng):
    """A WINDOW-length string of random A/C/G/T (neutral background context)."""
    return [rng.choice(BASES) for _ in range(WINDOW)]


def place_motif_centered(win, motif, offset_in_motif):
    """Write `motif` into `win` so that motif[offset_in_motif] lands on CENTER.
    Returns the window list mutated in place. The motif occupies positions
    [CENTER - offset_in_motif .. CENTER - offset_in_motif + len(motif) - 1]."""
    start = CENTER - offset_in_motif
    assert 0 <= start and start + len(motif) <= WINDOW
    for j, ch in enumerate(motif):
        win[start + j] = ch
    return win


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic variant sample.")
    ap.add_argument("--n", type=int, default=12, help="number of variants")
    ap.add_argument("--seed", type=int, default=20260628, help="RNG seed (reproducible)")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    rng = random.Random(args.seed)        # seeded -> identical file every run
    # The motif G sits at index 2 of "CAGCT"; the protective middle T at index 2
    # of "TATAT". Centring those means the CENTER base is the one the variant flips.
    del_center_base = DEL_MOTIF[2]        # 'G'
    ben_center_base = BEN_MOTIF[2]        # 'T'

    rows = []                              # each: (pos, ref, alt, window-string)

    # --- A few DELETERIOUS variants: ALT completes C A G C T at the centre. ----
    # Reference context already has C A _ C T around the centre; the centre REF is
    # some non-G base, and flipping it to G builds the motif -> big positive delta.
    n_del = max(2, args.n // 4)
    for _ in range(n_del):
        win = random_window(rng)
        place_motif_centered(win, DEL_MOTIF, 2)     # put the motif G on CENTER
        ref_base = rng.choice([b for b in BASES if b != del_center_base])
        win[CENTER] = ref_base                       # REF breaks the motif...
        rows.append((ref_base, del_center_base, win))  # ...ALT=G completes it

    # --- A few PROTECTIVE variants: ALT completes T A T A T at the centre. ------
    n_ben = max(2, args.n // 4)
    for _ in range(n_ben):
        win = random_window(rng)
        place_motif_centered(win, BEN_MOTIF, 2)     # put the motif T on CENTER
        ref_base = rng.choice([b for b in BASES if b != ben_center_base])
        win[CENTER] = ref_base
        rows.append((ref_base, ben_center_base, win))

    # --- Neutral filler: random context, random single-base substitution. ------
    while len(rows) < args.n:
        win = random_window(rng)
        ref_base = win[CENTER]
        alt_base = rng.choice([b for b in BASES if b != ref_base])
        rows.append((ref_base, alt_base, win))

    # Shuffle so the planted variants are not trivially first (the ranking, not
    # the file order, must surface them) -- a small but honest touch.
    rng.shuffle(rows)

    # Assign synthetic, strictly-increasing genomic coordinates for stable labels.
    lines = [f"{len(rows)} {WINDOW}"]
    for i, (ref_base, alt_base, win) in enumerate(rows):
        pos = 100000 + 137 * i              # arbitrary spaced-out coordinates
        lines.append(f"{pos} {ref_base} {alt_base} {''.join(win)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(n={len(rows)}, window={WINDOW}; SYNTHETIC -- not real genomics)")


if __name__ == "__main__":
    main()
