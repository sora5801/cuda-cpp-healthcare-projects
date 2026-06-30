// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for profile-HMM search
// ---------------------------------------------------------------------------
// Project 3.28 : Profile HMM (Viterbi / Forward)
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the plain host C++ compiler and must not see
//   any CUDA syntax, so the shared DATA MODEL (the sequence database container,
//   the file loader, the model builder) and the CPU-reference prototypes live
//   here. kernels.cuh (the GPU side) also includes this header so both sides
//   reuse the same ProfileHMM, the same SeqDB, and the same alphabet coding --
//   nothing CUDA-specific leaks in either direction.
//
// THE PROBLEM (full derivation in ../THEORY.md)
//   A PROFILE HMM is a position-specific model of a protein family. "Searching a
//   database" means: for each of N database sequences, compute how well the
//   profile explains it. Two scores per sequence:
//     * VITERBI  score = log-prob of the single best state path (best alignment)
//     * FORWARD  score = log-prob summed over ALL state paths (total support)
//   The N sequences are INDEPENDENT -> perfect data parallelism: one GPU thread
//   per database sequence (kernels.cu). This header is the CPU twin used to
//   verify the GPU and to make the speed-up legible.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh.  Depends on phmm.h (the
//   shared host/device recurrence core).
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "phmm.h"   // ProfileHMM, ALPHA, MAX_M, MAX_L, the HD recurrence core

// ---------------------------------------------------------------------------
// SeqDB : the loaded "database" of protein sequences to score against the model.
//   Sequences vary in length, so we store all residue codes back-to-back in one
//   flat `res` buffer and remember each sequence's [offset, length) -- the
//   classic CSR-style ragged layout. A flat buffer is exactly what we upload to
//   the GPU (one cudaMemcpy), and `off`/`len` let each thread find its sequence.
//     res    : concatenated residue codes (each 0..ALPHA-1)
//     off[s] : start index of sequence s inside res
//     len[s] : length of sequence s (number of residues)
//     name[s]: a short human label for the report (e.g. "homolog", "decoy3")
//   n == number of sequences == off.size() == len.size().
// ---------------------------------------------------------------------------
struct SeqDB {
    int                       n = 0;     // number of database sequences
    std::vector<int>          off;       // [n] start offset of each sequence in `res`
    std::vector<int>          len;       // [n] length of each sequence
    std::vector<std::uint8_t> res;       // concatenated residue codes (0..ALPHA-1)
    std::vector<std::string>  name;      // [n] short labels for the report
};

// ---------------------------------------------------------------------------
// aa_code(c) : map a one-letter amino-acid character to its code 0..ALPHA-1.
//   Returns -1 for any character that is not one of the 20 standard amino acids
//   (the loader treats that as an error). Exposed so the model builder and the
//   loader use ONE canonical ordering. The ordering is the standard alphabetical
//   one: A C D E F G H I K L M N P Q R S T V W Y.
// ---------------------------------------------------------------------------
int aa_code(char c);

// ---------------------------------------------------------------------------
// load_database(path) : parse the tiny FASTA-like text dataset (format is
//   documented in data/README.md). Throws std::runtime_error on a missing file,
//   an unknown residue, or a sequence longer than MAX_L. Returns a SeqDB.
//   The FIRST record (by convention) is the CONSENSUS used to build the profile;
//   main.cu pulls it out and the remaining records become the search database.
// ---------------------------------------------------------------------------
SeqDB load_database(const std::string& path);

// ---------------------------------------------------------------------------
// build_profile_from_consensus(consensus) : construct a simple but realistic
//   ProfileHMM whose match columns prefer the residues of a given CONSENSUS
//   sequence. This stands in for "estimate a profile from a multiple-sequence
//   alignment": match column k emits consensus residue k with high probability
//   and the other 19 residues with low probability, giving the model a clear
//   "signal" that a homologous sequence will match and a random sequence won't.
//   The transition logs use fixed, documented Plan-7-style defaults. M = length
//   of `consensus` (must be <= MAX_M). This is how the demo plants a known
//   answer (PATTERNS.md §6): the homolog is a mutated copy of the consensus.
//   `consensus` is a string of one-letter amino-acid codes.
// ---------------------------------------------------------------------------
ProfileHMM build_profile_from_consensus(const std::string& consensus);

// ---------------------------------------------------------------------------
// THE TWO CPU REFERENCE SCORERS  (the trusted serial baselines)
// ---------------------------------------------------------------------------
// Each fills out[s] with the score of database sequence s under the profile,
// computed by an obviously-correct serial 2-D DP that loops the SAME per-cell
// recurrences as the GPU (from phmm.h). If GPU and CPU agree, we trust the GPU.
//   * viterbi_cpu : max-sum recurrence -> log-prob of the single best path.
//   * forward_cpu : log-sum-exp recurrence -> log-prob over all paths.
// Both resize `out` to db.n. Results are in NATS (natural-log units), stored as
// float (the verification compares float arrays; see main.cu TOLERANCE).
// ---------------------------------------------------------------------------
void viterbi_cpu(const ProfileHMM& p, const SeqDB& db, std::vector<float>& out);
void forward_cpu(const ProfileHMM& p, const SeqDB& db, std::vector<float>& out);
