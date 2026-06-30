// ===========================================================================
// src/reference_cpu.h  --  Data loading + CPU reference for profile-HMM search
// ---------------------------------------------------------------------------
// Project 2.13 : MSA Generation Acceleration
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the DATA LOADER and the CPU reference prototype live here.
//   The actual MODEL (ProfileHMM, SeqDB) and the shared Viterbi recurrence live
//   in hmm_core.h, which BOTH this file and the GPU side include -- so there is
//   exactly one definition of the math.
//
// THE PROBLEM (see ../THEORY.md for the full derivation)
//   To build a multiple sequence alignment (the input AlphaFold2 needs), we take
//   a query protein, turn it into a profile HMM, and SCORE that profile against
//   every sequence in a huge database (UniRef90 ~210 GB) with the Viterbi
//   dynamic program; the top-scoring hits become the MSA. Each database sequence
//   is scored INDEPENDENTLY -> embarrassingly parallel. This project accelerates
//   that scan: CPU reference here, GPU twin in kernels.cu, one block per sequence.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh. (And hmm_core.h first.)
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "hmm_core.h"   // ProfileHMM, SeqDB, ALPHABET_SIZE, scoring constants

// A loaded search problem: one query profile HMM + a database of N sequences.
struct SearchProblem {
    ProfileHMM hmm;   // the query profile (built from the query sequence)
    SeqDB      db;    // the database to scan
};

// ---------------------------------------------------------------------------
// load_problem : parse the tiny text dataset documented in data/README.md.
//   The file encodes the profile's transition log-odds and per-column emission
//   log-odds, then the database sequences (as amino-acid letters). The loader
//   converts letters to indices and packs the database into SeqDB's CSR layout.
//   Throws std::runtime_error on a missing/malformed file so demos fail loudly.
// ---------------------------------------------------------------------------
SearchProblem load_problem(const std::string& path);

// ---------------------------------------------------------------------------
// aa_to_index : map an amino-acid LETTER to its 0..20 index (20 = unknown/X).
//   A pure function with no state; used by the loader to encode the database.
// ---------------------------------------------------------------------------
int aa_to_index(char c);

// ---------------------------------------------------------------------------
// viterbi_search_cpu : the trusted serial baseline.
//   For each database sequence i, run the full Viterbi sweep (calling the shared
//   viterbi_step / best_in_row from hmm_core.h) and store the best match score in
//   out[i]. This is the obviously-correct reference the GPU result is checked
//   against, and the timing baseline that makes the speed-up legible.
//   `out` is resized to db.N. Scores are scaled integers (see SCORE_SCALE).
// ---------------------------------------------------------------------------
void viterbi_search_cpu(const SearchProblem& prob, std::vector<int>& out);
