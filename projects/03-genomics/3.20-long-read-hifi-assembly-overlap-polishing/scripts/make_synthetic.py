#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic read-sketch dataset
# ---------------------------------------------------------------------------
# Project 3.20 : Long-Read HiFi Assembly Overlap & Polishing
#
# WHY SYNTHETIC
#   Real PacBio HiFi reads come from SRA / GenomeArk (see scripts/download_data.*
#   and data/README.md). To keep the demo OFFLINE, reproducible, and small, we
#   simulate a tiny "genome" and sample overlapping long reads from it, then
#   reduce each read to its MINIMISER SKETCH -- exactly the input the overlap
#   chaining kernel consumes. The data is clearly labelled SYNTHETIC everywhere.
#
# WHAT MAKES THE DEMO INTERPRETABLE (PATTERNS.md sec 6)
#   We lay reads end-to-end along the genome with deliberate OVERLAP between
#   neighbours (read i and read i+1 share a stretch of genome). Reads that came
#   from overlapping genome windows share many minimisers and chain into a HIGH
#   integer score; reads from far-apart windows share almost none and score ~0.
#   So the "top overlaps" the program prints recover the true neighbour structure
#   -- a known answer baked into the sample. A few point mutations per read make
#   the minimisers non-trivially robust (canonical hashing + windowing survive
#   isolated substitutions), which is the whole point of minimiser seeding.
#
# PARITY WITH THE C++ CORE  (critical!)
#   The k-mer length K, window W, canonical rule (min of forward / reverse-
#   complement), and the splitmix32 hash mix below MUST match src/overlap_core.h
#   bit-for-bit, because the C++ side trusts the hashes we emit here. If you change
#   one, change the other.
#
# OUTPUT FORMAT (data/README.md):
#   line 1            : "<n_reads>"
#   then per read r   : "<read_len> <cnt>" then cnt lines "<pos> <hash-8hex>"
#                       (positions ascending).
#
# USAGE
#   python scripts/make_synthetic.py                  # default: 12 reads
#   python scripts/make_synthetic.py --n-reads 2000   # a bigger overlap graph
# ===========================================================================
import argparse
import random
from pathlib import Path

# --- MUST match src/overlap_core.h -----------------------------------------
K = 15            # k-mer length (bases)
W = 5             # minimiser window (k-mers)
MASK = (1 << (2 * K)) - 1   # 2K-bit mask for the rolling k-mer code

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "reads_sample.txt"

BASES = "ACGT"
CODE = {"A": 0, "C": 1, "G": 2, "T": 3}


def splitmix32(x):
    """The exact integer avalanche used in ovl_canonical_kmer_hash (overlap_core.h).
    A bijective 32-bit finalizer so minimiser selection by 'smallest hash' is
    unbiased. All ops are mod 2**32 to mirror uint32_t wraparound on the C++ side."""
    x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7feb352d) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846ca68b) & 0xFFFFFFFF
    x ^= x >> 16
    return x & 0xFFFFFFFF


def revcomp_code(fwd):
    """Reverse-complement the 2-bit packing of a K-mer. Complement of code c is
    (3 - c) == (c ^ 3); reversing the base order turns 'fwd' into its rev-comp.
    Done with integer ops so it matches a rolling rev-comp hash exactly."""
    rev = 0
    x = fwd
    for _ in range(K):
        rev = (rev << 2) | ((x & 3) ^ 3)   # take low base, complement, append
        x >>= 2
    return rev & MASK


def kmer_hash(fwd):
    """Canonical, mixed hash of a K-mer given its forward 2-bit code (overlap_core.h
    ovl_canonical_kmer_hash): take min(forward, reverse-complement), then mix."""
    rev = revcomp_code(fwd)
    canon = fwd if fwd < rev else rev
    return splitmix32(canon)


def minimisers(seq):
    """Extract (pos, hash) minimisers from a DNA string, one per window of W
    consecutive k-mers, keeping the SMALLEST hash in each window. Returns a list
    sorted by position, de-duplicated when consecutive windows pick the same
    minimiser (the standard minimiser-sketch behaviour). Mirrors what a single
    pass of minimap2-style sketching produces."""
    n = len(seq)
    if n < K:
        return []
    # Per-k-mer canonical hashes and their start positions.
    hashes = []
    fwd = 0
    for i, b in enumerate(seq):
        fwd = ((fwd << 2) | CODE[b]) & MASK
        if i >= K - 1:
            hashes.append((i - K + 1, kmer_hash(fwd)))   # (start_pos, hash)
    # Slide a window of W k-mers; emit the min-hash k-mer of each window.
    out = []
    last = None
    for w0 in range(0, len(hashes) - W + 1):
        window = hashes[w0:w0 + W]
        # Pick smallest hash; break ties by earliest position (deterministic).
        best = min(window, key=lambda ph: (ph[1], ph[0]))
        if best != last:           # avoid emitting the same minimiser repeatedly
            out.append(best)
            last = best
    # Ensure strictly ascending positions (a window can re-pick an earlier pos).
    out.sort(key=lambda ph: ph[0])
    dedup = []
    seen_pos = -1
    for pos, h in out:
        if pos > seen_pos:
            dedup.append((pos, h))
            seen_pos = pos
    return dedup


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic HiFi overlap dataset.")
    ap.add_argument("--n-reads", type=int, default=12, help="number of reads")
    ap.add_argument("--read-len", type=int, default=300, help="read length (bases)")
    ap.add_argument("--overlap", type=int, default=150, help="overlap between neighbour reads")
    ap.add_argument("--mut-rate", type=float, default=0.004, help="per-base substitution rate (HiFi ~0.3%)")
    ap.add_argument("--seed", type=int, default=20, help="RNG seed (determinism)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Build a reference genome long enough to tile n_reads with the given overlap.
    step = args.read_len - args.overlap                 # genome advance per read
    genome_len = args.read_len + step * (args.n_reads - 1) + K
    genome = "".join(rng.choice(BASES) for _ in range(genome_len))

    reads = []
    for r in range(args.n_reads):
        start = r * step
        sub = list(genome[start:start + args.read_len])
        # Inject HiFi-rate substitutions so minimisers must be substitution-robust.
        for b in range(len(sub)):
            if rng.random() < args.mut_rate:
                sub[b] = rng.choice([c for c in BASES if c != sub[b]])
        # Half the reads come from the reverse strand (rev-comp) so canonical
        # minimiser hashing -- not naive forward hashing -- is what makes them match.
        seq = "".join(sub)
        if r % 2 == 1:
            comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
            seq = "".join(comp[c] for c in reversed(seq))
        reads.append(seq)

    lines = [str(args.n_reads)]
    total_min = 0
    for seq in reads:
        ms = minimisers(seq)
        total_min += len(ms)
        lines.append(f"{len(seq)} {len(ms)}")
        for pos, h in ms:
            lines.append(f"{pos} {h:08x}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}")
    print(f"[make_synthetic]   {args.n_reads} reads, {total_min} minimisers total "
          f"(K={K}, W={W}; SYNTHETIC, seed={args.seed})")
    print(f"[make_synthetic]   neighbour reads overlap by {args.overlap} bp -> "
          f"the top overlaps should be consecutive read indices.")


if __name__ == "__main__":
    main()
