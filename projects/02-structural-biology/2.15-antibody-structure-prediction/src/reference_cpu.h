// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for CDR similarity screen
// ---------------------------------------------------------------------------
// Project 2.15 : Antibody Structure Prediction  (reduced-scope: CDR screening)
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the plain host C++ compiler and must never
//   see CUDA syntax, so the shared DATA MODEL (the AntibodyLibrary container, the
//   text loader) and the CPU reference prototype live here. kernels.cuh also
//   includes this header so the GPU side reuses the exact same types -- nothing
//   CUDA-specific leaks across the boundary. The actual per-pair SCORING math is
//   one level deeper, in antibody.h (the __host__ __device__ core), so both the
//   CPU and GPU paths call ONE identical scoring function.
//
// THE PROBLEM (see ../THEORY.md for the full derivation)
//   Score ONE query antibody (its six CDR loops) against N library antibodies by
//   a CDR-weighted BLOSUM62 similarity, then report the most similar library
//   members (the screening hits). Every query-vs-library comparison is
//   independent -> one GPU thread per library antibody (kernels.cu).
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh. Read antibody.h first.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "antibody.h"   // AB_RECORD_LEN, ab_cdr_score, ab_encode_residue, ...

// A loaded screening dataset: one query antibody + n library antibodies.
//   Each antibody is stored as AB_RECORD_LEN (=144) ENCODED residues (0..20),
//   i.e. the six CDR fields concatenated and right-padded with gaps. We keep
//   encoded bytes (not ASCII) so the hot scoring loop indexes the substitution
//   matrix directly with no per-character branching.
//   names : human-readable id per library antibody, for the report (NOT scored).
struct AntibodyLibrary {
    int n = 0;                              // number of library antibodies
    std::vector<uint8_t> query;             // [AB_RECORD_LEN] encoded
    std::vector<uint8_t> lib;               // [n * AB_RECORD_LEN] encoded, row-major
    std::vector<std::string> names;         // [n] library antibody names
    std::string query_name;                 // name of the query antibody
};

// load_library: parse the tiny text dataset documented in data/README.md.
//   Format (whitespace/newline separated; lines starting with '#' are comments):
//     "QUERY <name> <H1> <H2> <H3> <L1> <L2> <L3>"
//     then n lines: "<name> <H1> <H2> <H3> <L1> <L2> <L3>"
//   Each CDR token is an amino-acid string (e.g. "ARDYYGSGS"); it is encoded and
//   right-padded with gaps to AB_CDR_LEN. Tokens longer than AB_CDR_LEN are
//   truncated (and the count of truncations is reported via *truncated). Throws
//   std::runtime_error on a missing/malformed file so demos fail loudly instead
//   of silently scoring empty input.
//     truncated : out-param, number of CDR tokens that exceeded AB_CDR_LEN (may
//                 be nullptr if the caller does not care).
AntibodyLibrary load_library(const std::string& path, int* truncated = nullptr);

// score_cpu: fill out[i] with ab_cdr_score(query, library antibody i) -- the
// trusted, obviously-correct serial baseline the GPU result is verified against
// (and the timing baseline that makes the speed-up legible). Scores are integers
// (int32) so the CPU and GPU results compare EXACTLY (tolerance 0). out -> n.
void score_cpu(const AntibodyLibrary& ab, std::vector<int32_t>& out);
