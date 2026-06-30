// ===========================================================================
// src/reference_cpu.h  --  Variant-calling data model + CPU PairHMM reference
// ---------------------------------------------------------------------------
// Project 3.3 : Variant Calling Acceleration
//
// WHAT THIS PROJECT COMPUTES
//   The PairHMM forward step of germline variant calling: for every (READ,
//   HAPLOTYPE) pair, compute the likelihood P(read | haplotype) -- the marginal
//   probability that the read was sequenced from that candidate haplotype, under
//   a 3-state (Match/Insert/Delete) pair Hidden Markov Model. With `R` reads and
//   `H` haplotypes this produces an R x H likelihood matrix. GATK's
//   HaplotypeCaller turns that matrix into genotype likelihoods; here we stop at
//   the likelihood matrix (the dominant cost), then pick, for each read, the
//   most-likely haplotype -- a deterministic, checkable result.
//
// WHY A SEPARATE PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the plain host compiler and must not see any
//   CUDA/__global__ syntax. Both main.cu and reference_cpu.cpp include THIS
//   header so they agree on the data layout and the reference signature. The
//   actual per-cell arithmetic lives in pairhmm_core.h (shared with the GPU).
//
//   The CPU reference exists for two reasons (CLAUDE.md §5):
//     (a) it is the readable baseline that makes the GPU speed-up legible, and
//     (b) the demo runs BOTH and asserts they agree within tolerance.
//
// READ THIS AFTER: pairhmm_core.h. Used by: main.cu, reference_cpu.cpp.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pairhmm_core.h"   // PairHmmParams, encode_base (shared HD core)

// ---------------------------------------------------------------------------
// A loaded variant-calling problem: a set of reads and a set of candidate
// haplotypes, plus the pair-HMM transition parameters. Bases are stored encoded
// (0..4 from encode_base) so the kernel reads compact bytes; qualities are
// per-read-base Phred scores.
//
//   Layout note: reads have varying lengths in reality, but for a clean teaching
//   demo (and coalesced GPU access) we store all reads at a fixed `read_len` and
//   all haplotypes at a fixed `hap_len`, row-major. Production callers pack
//   ragged reads with offset arrays; THEORY.md "real world" explains the change.
// ---------------------------------------------------------------------------
struct VariantData {
    int n_reads = 0;   // number of sequencing reads
    int n_haps  = 0;   // number of candidate haplotypes
    int read_len = 0;  // bases per read (fixed for this teaching layout)
    int hap_len  = 0;  // bases per haplotype (fixed)
    int truth   = -1;  // haplotype index the synthetic reads were drawn from (-1 if unknown)

    std::vector<uint8_t> reads;  // [n_reads * read_len] encoded bases (0..4), row-major
    std::vector<uint8_t> quals;  // [n_reads * read_len] Phred base qualities (int 0..60)
    std::vector<uint8_t> haps;   // [n_haps  * hap_len ] encoded bases (0..4), row-major

    PairHmmParams params{};      // gap-open / gap-extend, filled by load_variant_data
};

// Load from the text format documented in data/README.md. Throws
// std::runtime_error if the file cannot be opened or is malformed.
VariantData load_variant_data(const std::string& path);

// ---------------------------------------------------------------------------
// CPU reference: fill the R x H log10-likelihood matrix.
//   loglik[r*n_haps + h] = log10 P(read r | haplotype h), computed with the
//   forward algorithm over the full DP table (pairhmm_core.h's `pairhmm_step`).
//   We return log10 likelihoods (GATK's convention) because the raw probabilities
//   underflow to 0 quickly; the log keeps them representable and comparable.
//   This is the trusted baseline the GPU kernel is checked against.
// ---------------------------------------------------------------------------
void pairhmm_cpu(const VariantData& v, std::vector<double>& loglik);

// For each read, the index of the most-likely haplotype (argmax over a row of
// the log-likelihood matrix; ties -> lowest index). Deterministic; the headline
// result printed to stdout. Shared by the CPU and GPU report paths.
void best_haplotype_per_read(const VariantData& v, const std::vector<double>& loglik,
                             std::vector<int>& best);
