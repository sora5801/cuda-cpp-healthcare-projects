#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic metagenome dataset
# ---------------------------------------------------------------------------
# Project 3.14 : Metagenomic Taxonomic Classification
#
# WHY SYNTHETIC
#   Real metagenomic classification uses a multi-gigabyte Kraken2 database built
#   from NCBI RefSeq genomes plus sequencing reads (see scripts/download_data.*
#   and data/README.md). To keep the demo OFFLINE, tiny, and reproducible we
#   generate a clearly-SYNTHETIC dataset: a handful of short "reference genomes"
#   (random A/C/G/T) standing in for distinct microbial species, and a set of
#   "reads" sampled from those genomes with a few point mutations -- exactly the
#   structure a real sample has, at a scale a learner can inspect by eye.
#
# WHAT MAKES THE RESULT INTERPRETABLE  (docs/PATTERNS.md sec 6)
#   * Each read carries its TRUE taxon id as ground truth, so the demo can report
#     classification accuracy, not just CPU==GPU agreement.
#   * Genomes are random and long enough that they share almost no 15-mers by
#     chance, so a read's k-mers vote overwhelmingly for its source taxon -> the
#     abundance profile cleanly recovers the simulated community composition.
#   * A few reads are pure-random "contaminants" (truth taxon 0) that should come
#     back unclassified -- a teaching example of the unclassified bin.
#   A fixed RNG seed makes the output byte-for-byte reproducible (so the committed
#   sample and demo/expected_output.txt are stable).
#
# OUTPUT FORMAT (also documented in data/README.md):
#   T <num_taxa>
#   REF  <taxon_name> <genome_sequence>          x num_taxa
#   R <num_reads>
#   READ <true_taxon_id> <read_sequence>         x num_reads
#   ('#' comment lines and blank lines are ignored by the loader.)
#
# USAGE
#   python scripts/make_synthetic.py                      # the committed sample
#   python scripts/make_synthetic.py --reads 100000       # a bigger stress set
# ===========================================================================
import argparse
import random
from pathlib import Path

# These "species" names are illustrative labels for random sequences -- they are
# NOT real genomes of these organisms. Synthetic data, labeled synthetic.
TAXA = [
    "Escherichia_coli",
    "Staphylococcus_aureus",
    "Pseudomonas_aeruginosa",
    "Bacteroides_fragilis",
    "Lactobacillus_casei",
]

BASES = "ACGT"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "metagenome_sample.txt"


def random_genome(rng, length):
    """A random A/C/G/T string standing in for one species' reference genome."""
    return "".join(rng.choice(BASES) for _ in range(length))


def mutate(rng, seq, rate):
    """Return `seq` with each base independently substituted with probability
    `rate` -- models sequencing error / strain variation. A low rate keeps most
    of a read's k-mers intact so it still votes correctly for its source taxon."""
    out = []
    for b in seq:
        if rng.random() < rate:
            out.append(rng.choice(BASES))   # substitution (may pick same base)
        else:
            out.append(b)
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic metagenome.")
    ap.add_argument("--genome-len", type=int, default=600, help="length of each reference genome")
    ap.add_argument("--read-len",   type=int, default=80,  help="length of each read")
    ap.add_argument("--reads",      type=int, default=40,  help="number of reads to simulate")
    ap.add_argument("--mut-rate",   type=float, default=0.01, help="per-base mutation rate in reads")
    ap.add_argument("--contam-frac",type=float, default=0.1,  help="fraction of pure-random reads")
    ap.add_argument("--seed",       type=int, default=7,   help="RNG seed (determinism)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # 1) Reference genomes: one random sequence per taxon.
    genomes = [random_genome(rng, args.genome_len) for _ in TAXA]

    # 2) A simulated community: choose how many reads come from each taxon. We use
    #    a deliberately UNEVEN mix so the abundance profile is interesting (taxon 1
    #    dominant, taxon 5 rare) -- like a real microbiome.
    weights = [0.40, 0.25, 0.20, 0.10, 0.05]   # relative abundance per taxon
    n_contam = int(round(args.reads * args.contam_frac))
    n_real = args.reads - n_contam

    reads = []  # list of (truth_taxon_id, sequence)
    for _ in range(n_real):
        # Pick a source taxon by abundance, then a random window of its genome,
        # then mutate it lightly to mimic sequencing.
        t = rng.choices(range(len(TAXA)), weights=weights, k=1)[0]
        g = genomes[t]
        start = rng.randint(0, len(g) - args.read_len)
        frag = mutate(rng, g[start:start + args.read_len], args.mut_rate)
        reads.append((t + 1, frag))            # taxon ids are 1-based

    # 3) Contaminant reads: pure random sequence, should be UNCLASSIFIED (truth 0).
    for _ in range(n_contam):
        reads.append((0, random_genome(rng, args.read_len)))

    rng.shuffle(reads)   # interleave so order does not encode the answer

    # 4) Emit the dataset.
    lines = []
    lines.append("# SYNTHETIC metagenome -- generated by scripts/make_synthetic.py")
    lines.append("# Sequences are RANDOM DNA; the species names are illustrative labels,")
    lines.append("# NOT real genomes. For educational use only -- not for any clinical purpose.")
    lines.append(f"T {len(TAXA)}")
    for name, g in zip(TAXA, genomes):
        lines.append(f"REF {name} {g}")
    lines.append(f"R {len(reads)}")
    for truth, seq in reads:
        lines.append(f"READ {truth} {seq}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"({len(TAXA)} taxa, {len(reads)} reads; SYNTHETIC, seed={args.seed})")


if __name__ == "__main__":
    main()
