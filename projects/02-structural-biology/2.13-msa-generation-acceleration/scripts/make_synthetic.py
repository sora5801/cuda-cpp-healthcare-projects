#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic profile-HMM + database
# ---------------------------------------------------------------------------
# Project 2.13 : MSA Generation Acceleration
#
# WHY THIS EXISTS
#   The real databases (UniRef90 ~210 GB, BFD, MGnify) are far too large to
#   commit and are governed by their own licenses. So the offline demo runs on a
#   TINY, clearly-SYNTHETIC stand-in that this script builds deterministically.
#   Everything here is fabricated for teaching -- it carries NO biological meaning
#   and must never be presented as a real search result (CLAUDE.md §8).
#
# WHAT IT BUILDS (engineered so the answer is KNOWN and verifiable -- PATTERNS.md §6)
#   * A query "motif": a short amino-acid pattern (a made-up family signature).
#   * A profile HMM whose emission log-odds reward residues matching the motif
#     column by column (positive log-odds for the consensus residue, mild
#     negatives for others), plus simple gap/transition penalties.
#   * A database of N sequences. A handful of PLANTED sequences contain the motif
#     (optionally with a point mutation or a short insertion) buried in random
#     filler -> they should score HIGH. The rest are pure random noise -> LOW.
#   The demo then reports the top-K hits; the planted indices should dominate,
#   which is the "did we recover the embedded answer?" sanity check.
#
# THE OUTPUT FORMAT (consumed by src/reference_cpu.cpp load_problem):
#   line 1            : "L N"                         (profile length, #sequences)
#   line 2            : 7 transition log-odds (scaled ints): t_mm t_mi t_im t_ii t_md t_dm t_dd
#   next L lines      : 21 emission log-odds (scaled ints) per match column
#   next N lines      : one database sequence each, as amino-acid LETTERS
#   '#'-prefixed lines are comments (skipped by the loader).
#
# Scores are stored as SCALED INTEGERS (log-odds * SCORE_SCALE, SCORE_SCALE=1000)
# so the C++ Viterbi DP is pure integer arithmetic -> CPU and GPU agree exactly.
#
# USAGE
#   python scripts/make_synthetic.py                 # default tiny sample
#   python scripts/make_synthetic.py --n 64 --seed 7 # bigger / different draw
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT  = ROOT / "data" / "sample" / "profile_db_sample.txt"

# The 20 standard amino acids in the SAME order as aa_to_index() in
# src/reference_cpu.cpp (index 20 is the "unknown" catch-all, never emitted here).
AA = "ARNDCQEGHILKMFPSTWYV"
SCORE_SCALE = 1000           # must match hmm_core.h SCORE_SCALE

# A made-up "family signature" motif. Each character is the consensus residue of
# one profile column. (Purely synthetic -- not a real motif.)
MOTIF = "WYGGFPKDEC"


def emission_row(consensus_aa, match_logodds=2.0, other_logodds=-0.5):
    """Build one match column's 21 emission log-odds (scaled ints).
    The consensus residue gets a strong positive log-odds (it is much more likely
    here than by chance); every other residue gets a mild negative; the unknown
    slot (index 20) gets a stronger negative so 'X' never helps a match."""
    row = []
    for a in AA:
        v = match_logodds if a == consensus_aa else other_logodds
        row.append(int(round(v * SCORE_SCALE)))
    row.append(int(round(-2.0 * SCORE_SCALE)))     # index 20: unknown/X
    return row


def random_seq(rng, length):
    """A random protein-like string over the 20 amino acids (uniform)."""
    return "".join(rng.choice(AA) for _ in range(length))


def plant_motif(rng, filler_len, mutate=0, insert=0):
    """Embed MOTIF inside random filler, optionally with `mutate` point mutations
    and a short `insert` of random residues in the middle (to exercise the HMM's
    insert states). Returns the full sequence string."""
    motif = list(MOTIF)
    for _ in range(mutate):
        p = rng.randrange(len(motif))
        motif[p] = rng.choice(AA)
    if insert > 0:
        mid = len(motif) // 2
        motif[mid:mid] = [rng.choice(AA) for _ in range(insert)]
    core = "".join(motif)
    left  = random_seq(rng, filler_len)
    right = random_seq(rng, filler_len)
    return left + core + right


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic profile-HMM + database sample.")
    ap.add_argument("--n", type=int, default=24, help="number of database sequences")
    ap.add_argument("--seed", type=int, default=2025, help="PRNG seed (determinism)")
    ap.add_argument("--filler", type=int, default=12, help="random residues flanking a planted motif")
    ap.add_argument("--noise-len", type=int, default=40, help="length of pure-noise sequences")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    rng = random.Random(args.seed)          # seeded -> deterministic output
    L = len(MOTIF)
    N = args.n

    # --- Transition log-odds (scaled ints). Match->match is the cheap default;
    #     opening/extending gaps and inserts cost progressively more. These are
    #     simple teaching values, not estimated from data. ---
    t = {
        "mm": int(round( 0.0  * SCORE_SCALE)),   # match -> match  (no penalty)
        "mi": int(round(-3.0  * SCORE_SCALE)),   # open an insertion
        "im": int(round(-1.0  * SCORE_SCALE)),   # close an insertion
        "ii": int(round(-0.5  * SCORE_SCALE)),   # extend an insertion
        "md": int(round(-3.0  * SCORE_SCALE)),   # open a deletion
        "dm": int(round(-1.0  * SCORE_SCALE)),   # close a deletion
        "dd": int(round(-0.5  * SCORE_SCALE)),   # extend a deletion
    }

    # --- Emission table: one row per motif column. ---
    emit_rows = [emission_row(c) for c in MOTIF]

    # --- Database: plant the motif in a few sequences, fill the rest with noise.
    #     We record which indices are planted so the data/README can document the
    #     expected hits (the "known answer"). ---
    seqs = []
    planted = []
    # index 0: clean motif (should be the top hit)
    seqs.append(plant_motif(rng, args.filler));                 planted.append(0)
    # index 1: motif with 1 point mutation (slightly lower score)
    seqs.append(plant_motif(rng, args.filler, mutate=1));       planted.append(1)
    # index 2: motif with a 2-residue insertion (tests insert states)
    seqs.append(plant_motif(rng, args.filler, insert=2));       planted.append(2)
    # the rest: pure noise
    while len(seqs) < N:
        seqs.append(random_seq(rng, args.noise_len))

    # --- Emit the file. ---
    lines = []
    lines.append("# SYNTHETIC profile-HMM + sequence database for project 2.13.")
    lines.append("# Fabricated for teaching -- NOT real biological data. See data/README.md.")
    lines.append(f"# motif (consensus): {MOTIF}   planted hits at indices: {planted}")
    lines.append(f"{L} {N}")
    lines.append("# transitions: t_mm t_mi t_im t_ii t_md t_dm t_dd  (scaled log-odds, *1000)")
    lines.append(f"{t['mm']} {t['mi']} {t['im']} {t['ii']} {t['md']} {t['dm']} {t['dd']}")
    lines.append(f"# {L} emission rows, each 21 scaled log-odds (20 amino acids + X)")
    for k, row in enumerate(emit_rows):
        lines.append(" ".join(str(v) for v in row) + f"   # column {k+1}: consensus {MOTIF[k]}")
    lines.append(f"# {N} database sequences (amino-acid letters); planted hits first")
    for s in seqs:
        lines.append(s)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}")
    print(f"[make_synthetic] L={L} profile columns, N={N} sequences, "
          f"planted hits at {planted}  (SYNTHETIC -- not real data)")


if __name__ == "__main__":
    main()
