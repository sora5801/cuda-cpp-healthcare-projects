#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic equivalence-class sample
# ---------------------------------------------------------------------------
# Project 3.22 : RNA-seq Quantification / Pseudo-alignment
#
# WHY THIS EXISTS
#   Real RNA-seq FASTQs (ENCODE, GTEx, SRA) are large and/or need registration,
#   and running a true pseudo-aligner (kallisto/Salmon) to PRODUCE equivalence
#   classes is out of scope for a teaching demo. So we synthesize the *output of
#   the pseudo-alignment step* directly: a small transcriptome with KNOWN
#   abundances and a known isoform / equivalence-class structure, with read
#   counts generated from that truth by the SAME length-normalised model the EM
#   inverts. The EM in this project then RECOVERS the known abundances almost
#   exactly -- which makes the demo interpretable and verifiable.
#
#   The data is SYNTHETIC and labeled so everywhere (data/README.md). It implies
#   nothing clinical.
#
# THE TOY BIOLOGY WE MODEL
#   3 genes, each with isoforms that SHARE sequence, so reads from a shared region
#   are compatible with several transcripts -> they land in a multi-transcript
#   equivalence class (ec). The transcripts:
#     gene A: t0, t1            (two isoforms; one shared exon -> ec {0,1})
#     gene B: t2, t3, t4        (nested sharing -> ecs {2,3} and {2,3,4})
#     gene C: t5                (single isoform; all its reads are unambiguous)
#   Each transcript also has a UNIQUE region producing a singleton ec {t}; those
#   unambiguous reads are exactly the information the EM uses to break the
#   ambiguity in the shared ecs.
#
# THE GENERATIVE MODEL (this is the part worth understanding)
#   Truth: an abundance rho[t] (fraction of reads from transcript t) and an
#   effective length eff[t]. The per-length read-generation "weight" is
#       w[t] = rho[t] / eff[t].
#   Of all reads, transcript t produces reads_t[t] = TOTAL * rho[t]. We send a
#   chosen TOTAL number of reads C_e into each SHARED ec e, and split them among
#   the ec's members in proportion to their weights:
#       reads contributed by member t to ec e  =  C_e * w[t] / sum_{s in e} w[s].
#   Each transcript's leftover reads (its total minus what it sent to shared ecs)
#   go to its UNIQUE singleton ec. Because the shared-ec split is exactly the
#   length-normalised proportion the EM assumes, the true rho is (up to integer
#   rounding) the EM's fixed point: EM recovers it to ~1e-5. We assert the unique
#   counts come out non-negative (the chosen C_e are small enough).
#
#   Everything is integer arithmetic with no RNG -> DETERMINISTIC, so
#   expected_output.txt is stable run to run.
#
# OUTPUT FORMAT (see data/README.md): "T M", eff lengths, then M ec lines
#   "count k m0..m_{k-1}", then a "TRUTH" block of T fractions.
#
# USAGE
#   python scripts/make_synthetic.py                 # writes the committed sample
#   python scripts/make_synthetic.py --reads 200000  # scale the total read budget
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "rnaseq_ec_sample.txt"

# Effective lengths (length units). Longer transcripts emit more reads at the same
# molar abundance; the EM divides this back out via rho/eff. Chosen distinct so the
# length-normalisation is visibly exercised (t4 is longest at 2000).
EFF_LEN = [1000.0, 1500.0, 1200.0, 900.0, 2000.0, 1100.0]

# Ground-truth ABUNDANCE rho[t] = fraction of all reads from transcript t. Chosen
# as round, well-separated values so the recovered numbers are easy to eyeball.
TRUTH_RHO = [0.30, 0.10, 0.05, 0.20, 0.15, 0.20]

# SHARED equivalence classes: (member transcript ids, total reads C_e in that ec).
# Singleton (unique) ecs are generated automatically as the per-transcript
# remainder. Keep the C_e modest so every remainder stays non-negative.
SHARED_ECS = [
    ([0, 1],    18000),   # gene A shared exon
    ([2, 3],     8000),   # gene B shared exon (t2 & t3)
    ([2, 3, 4], 12000),   # gene B triple-shared exon (t2, t3 & t4)
]


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic RNA-seq ec sample.")
    ap.add_argument("--reads", type=int, default=100000,
                    help="total read budget distributed across transcripts by TRUTH_RHO")
    ap.add_argument("--out", default=str(OUT), help="output path")
    args = ap.parse_args()

    T = len(EFF_LEN)
    assert len(TRUTH_RHO) == T

    # Normalise truth to a proper distribution; compute generative weights and the
    # total reads each transcript produces.
    s = sum(TRUTH_RHO)
    truth = [r / s for r in TRUTH_RHO]
    w = [truth[t] / EFF_LEN[t] for t in range(T)]
    reads_t = [args.reads * truth[t] for t in range(T)]

    # Start every transcript's UNIQUE-ec count at its full read budget, then subtract
    # the reads it contributes to each shared ec (split proportional to weight).
    unique = list(reads_t)
    for members, C in SHARED_ECS:
        Wsum = sum(w[t] for t in members)
        for t in members:
            unique[t] -= C * w[t] / Wsum

    # The chosen shared totals must leave non-negative unique remainders.
    for t in range(T):
        assert unique[t] >= -1e-6, (
            f"transcript {t} unique count went negative ({unique[t]:.1f}); "
            f"lower a SHARED_ECS total.")

    # Assemble the full ec list: T singletons (unique regions) then the shared ecs.
    struct = [[t] for t in range(T)] + [m for m, _ in SHARED_ECS]
    counts = [round(u) for u in unique] + [C for _, C in SHARED_ECS]
    ecs = list(zip(counts, struct))

    # ----- Emit the file -----
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# SYNTHETIC RNA-seq pseudo-alignment equivalence classes (project 3.22).")
    lines.append("# NOT real sequencing data. 3 toy genes / 6 transcripts; read counts")
    lines.append("# generated from the TRUTH block below by the length-normalised model the")
    lines.append("# EM inverts, so the EM recovers the truth to ~1e-5.")
    lines.append("# Format: 'T M' / eff lengths / M ec lines 'count k m..' / 'TRUTH' + T rho.")
    lines.append(f"{T} {len(ecs)}")
    lines.append("# effective lengths (length units), one per transcript t0..t{}:".format(T - 1))
    lines.append(" ".join(f"{v:g}" for v in EFF_LEN))
    lines.append("# equivalence classes: count  k  member transcript ids")
    lines.append("#   (the first 6 are unique-region singletons; the last 3 are shared)")
    for count, members in ecs:
        lines.append(f"{count} {len(members)} " + " ".join(str(m) for m in members))
    lines.append("# ground-truth abundances rho (fraction of reads per transcript):")
    lines.append("TRUTH " + " ".join(f"{r:g}" for r in truth))

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    total = sum(c for c, _ in ecs)
    print(f"[make_synthetic] wrote {out}")
    print(f"[make_synthetic]   T={T} transcripts, M={len(ecs)} ecs, "
          f"{total} reads (SYNTHETIC; truth embedded, recoverable to ~1e-5).")


if __name__ == "__main__":
    main()
