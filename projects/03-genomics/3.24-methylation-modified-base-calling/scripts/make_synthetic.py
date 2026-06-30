#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate a synthetic methylation-calling problem
# ---------------------------------------------------------------------------
# Project 3.24 : Methylation / Modified-Base Calling
#
# WHAT THIS BUILDS (all SYNTHETIC -- no real patient or sequencing data)
#   A self-contained nanopore methylation-calling instance the demo can verify:
#     * a CANONICAL pore model and a METHYLATED pore model over all 4^K k-mers.
#       The two models are IDENTICAL except for the k-mers that contain a CpG 'C':
#       in the methylated model those k-mers' expected current is shifted, which
#       is exactly how 5mC perturbs the ionic current in a real pore.
#     * a set of CpG sites, each labeled methylated (5mC) or canonical (C) as
#       ground truth.
#     * for each site, `coverage` reads. Each read carries the observed event
#       currents over the site's local reference window, drawn (with realistic
#       jitter) from the model matching the site's true state -- so a correct
#       banded-DP + log-likelihood-ratio caller recovers the labels.
#
#   This makes the result INTERPRETABLE (PATTERNS.md §6): the program reports a
#   per-site call and how many match the planted ground truth.
#
# THE PORE-MODEL TRICK (so two global k-mer tables suffice)
#   The C++ data model keeps two pore models indexed by k-mer CODE (0..4^K-1).
#   We center every site's window so the CpG 'C' sits at a fixed base offset, and
#   we choose the surrounding reference bases so the k-mers spanning the CpG have
#   distinct codes. Only those codes get a methylated shift; all other codes are
#   identical between the two models. Thus "is this read methylated?" reduces to
#   "do the events over the CpG-spanning k-mers fit the shifted means?".
#
# OUTPUT FORMAT (must match load_meth_data in src/reference_cpu.cpp; see data/README.md):
#   num_sites num_reads coverage
#   <4^K lines> canon_mean canon_stdv
#   <4^K lines> meth_mean  meth_stdv
#   <num_sites lines> site_pos truth
#   <num_jobs lines>  read_id site_id  b0..b11  e0..e9      (num_jobs = num_sites*coverage)
#
# USAGE
#   python scripts/make_synthetic.py                 # tiny committed sample
#   python scripts/make_synthetic.py --sites 4096    # a bigger instance for timing
# ===========================================================================
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "methylation_sample.txt"

# These MUST mirror the C++ compile-time constants (meth_core.h / reference_cpu.h).
KMER_K       = 3
NUM_KMERS    = 4 ** KMER_K          # 64
WINDOW_BASES = 12
WINDOW_KMERS = WINDOW_BASES - KMER_K + 1   # 10
EVENTS_PER_JOB = WINDOW_KMERS        # one event per reference k-mer (1:1 layout)

BASES = "ACGT"
CODE = {b: i for i, b in enumerate(BASES)}   # A=0,C=1,G=2,T=3 (matches base_to_code)

# Fixed offset of the CpG 'C' inside the 12-base window (0-based). With KMER_K=3,
# the k-mers (windows of 3 bases starting at j) that CONTAIN base CPG_C_OFFSET are
# j in {CPG_C_OFFSET-2 .. CPG_C_OFFSET} intersected with [0, WINDOW_KMERS-1].
CPG_C_OFFSET = 5

# Current shift (in normalized units) applied to CpG-spanning k-mers in the
# METHYLATED model. Big enough that a handful of events clearly separate the two
# models; small enough to stay in a realistic "a few pA" regime.
METH_SHIFT = 6.0


def kmer_code(codes3):
    """Pack 3 base-codes (MSB first) into [0,64) -- mirrors meth_core.h kmer_code."""
    c = 0
    for x in codes3:
        c = (c << 2) | (x & 0x3)
    return c


def build_window():
    """A fixed 12-base reference window with a CpG at CPG_C_OFFSET. Chosen so the
    CpG-spanning k-mers have distinct, easily-shifted codes. Returns (base string,
    list of base codes, set of window-k-mer indices that span the CpG 'C')."""
    # A readable, fixed window: A T G A C C G T A C G T  with C at index 5 (the
    # methylated cytosine) and G at index 6 (the 'CpG'). The exact flanking bases
    # are arbitrary; we just need them fixed so codes are reproducible.
    win = "ATGACCGTACGT"
    assert len(win) == WINDOW_BASES
    assert win[CPG_C_OFFSET] == "C" and win[CPG_C_OFFSET + 1] == "G", "need a CpG at the offset"
    codes = [CODE[b] for b in win]
    spanning = {j for j in range(WINDOW_KMERS)
                if j <= CPG_C_OFFSET <= j + KMER_K - 1}
    return win, codes, spanning


def build_pore_models(codes, spanning):
    """Construct the canonical + methylated k-mer pore models.
       canonical: a smooth, deterministic mean per k-mer code (so every k-mer has
                  a defined expectation), std = 2.0.
       methylated: identical, EXCEPT the codes of the CpG-spanning k-mers get
                  +METH_SHIFT on their mean. Returns (canon, meth) as lists of
                  (mean, stdv) indexed by k-mer code."""
    # Deterministic per-code mean: spread across a plausible current range.
    canon = [(60.0 + (kc % 32) * 1.7, 2.0) for kc in range(NUM_KMERS)]
    meth = list(canon)
    # Identify the k-mer CODES that span the CpG in our fixed window, and shift
    # ONLY those in the methylated model.
    for j in spanning:
        kc = kmer_code(codes[j:j + KMER_K])
        m, s = canon[kc]
        meth[kc] = (m + METH_SHIFT, s)
    return canon, meth


def draw_events(rng, codes, model, jitter):
    """Generate one read's EVENTS_PER_JOB observed currents: event w is drawn from
       a Gaussian around `model[kmer_code(window k-mer w)]`'s mean with the given
       jitter. This is the synthetic 'signal' the DP will re-align and score."""
    ev = []
    for w in range(WINDOW_KMERS):
        kc = kmer_code(codes[w:w + KMER_K])
        mean, _stdv = model[kc]
        ev.append(rng.gauss(mean, jitter))
    return ev


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic methylation-calling dataset.")
    ap.add_argument("--sites", type=int, default=12, help="number of CpG sites")
    ap.add_argument("--coverage", type=int, default=8, help="reads per site")
    ap.add_argument("--meth-frac", type=float, default=0.5, help="fraction of sites that are 5mC")
    ap.add_argument("--jitter", type=float, default=2.0, help="event current noise (std)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    win, codes, spanning = build_window()
    canon, meth = build_pore_models(codes, spanning)

    num_sites = args.sites
    coverage = args.coverage
    num_reads = num_sites * coverage   # every read covers exactly one site here

    # Ground-truth labels: deterministic from the seeded RNG so ~meth_frac of
    # sites are methylated.
    truth = [1 if rng.random() < args.meth_frac else 0 for _ in range(num_sites)]
    # Spread sites along a synthetic chromosome coordinate (arbitrary but increasing).
    site_pos = [1000 + s * 137 for s in range(num_sites)]

    # Build the jobs: for each site, `coverage` reads drawn from the matching model.
    jobs = []  # each: (read_id, site_id, codes(12), events(10))
    read_id = 0
    for s in range(num_sites):
        model = meth if truth[s] == 1 else canon
        for _ in range(coverage):
            ev = draw_events(rng, codes, model, args.jitter)
            jobs.append((read_id, s, list(codes), ev))
            read_id += 1

    # ---- Serialize in the exact format load_meth_data expects ----------------
    lines = [f"{num_sites} {num_reads} {coverage}"]
    for (m, sdv) in canon:
        lines.append(f"{m:.4f} {sdv:.4f}")
    for (m, sdv) in meth:
        lines.append(f"{m:.4f} {sdv:.4f}")
    for s in range(num_sites):
        lines.append(f"{site_pos[s]} {truth[s]}")
    for (rid, sid, cds, ev) in jobs:
        cds_s = " ".join(str(c) for c in cds)
        ev_s = " ".join(f"{x:.4f}" for x in ev)
        lines.append(f"{rid} {sid} {cds_s} {ev_s}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_meth = sum(truth)
    print(f"[make_synthetic] wrote {args.out}")
    print(f"  SYNTHETIC: {num_sites} CpG sites ({n_meth} methylated / {num_sites - n_meth} canonical), "
          f"coverage {coverage}, {num_reads} reads, {len(jobs)} jobs.")
    print(f"  k={KMER_K} ({NUM_KMERS} k-mers), window {WINDOW_BASES} b, CpG-spanning k-mer codes: "
          f"{sorted(kmer_code(codes[j:j+KMER_K]) for j in spanning)} (methylated shift +{METH_SHIFT}).")


if __name__ == "__main__":
    main()
