// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for CRISPR off-target scan
// ---------------------------------------------------------------------------
// Project 3.17 : CRISPR Guide Design & Off-Target Scoring
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the shared DATA MODEL (how a guide + genome are stored, how
//   bases are encoded, the file loader) and the CPU reference prototypes live
//   here. kernels.cuh also includes this header to reuse the same types -- so
//   the two sides cannot drift. The actual per-window math is one level deeper,
//   in cfd_score.h (the __host__ __device__ core).
//
// THE PROBLEM (full derivation in ../THEORY.md)
//   Given ONE 20-nt guide RNA spacer and a reference genome (a long base
//   string), find every site where SpCas9 could cut -- i.e. every 23-base
//   window whose last 3 bases are an "NGG" PAM -- and score how strongly the
//   guide would cut there using the CFD off-target model (cfd_score.h). Sliding
//   the window over an L-base genome yields ~L candidate sites, each scored
//   INDEPENDENTLY -> embarrassingly parallel -> one GPU thread per genome
//   position (kernels.cu). This is the same "1 query vs N items + constant-
//   memory query" pattern as flagship 1.12 (Tanimoto), here over genome windows.
//
// READ THIS BEFORE: reference_cpu.cpp, cfd_score.h, kernels.cuh.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "cfd_score.h"   // GUIDE_LEN, PAM_LEN, WINDOW_LEN, WindowScore, score_window

// ---------------------------------------------------------------------------
// encode_base: map an ASCII nucleotide to its 2-bit code (A/C/G/T -> 0/1/2/3),
// anything else -> BASE_INVALID (255). Uppercasing is handled by the caller's
// loader. This MUST agree with the enum in cfd_score.h. Declared here, defined
// in reference_cpu.cpp, and reused by the loader so the genome/guide encodings
// are guaranteed consistent.
// ---------------------------------------------------------------------------
uint8_t encode_base(char c);

// decode_base: inverse of encode_base, for human-readable reporting of the
// matched protospacer ("ACGT...") in main.cu. 255 -> 'N'.
char decode_base(uint8_t code);

// ---------------------------------------------------------------------------
// CrisprProblem: a loaded scan job.
//   guide  : the 20 spacer bases as 2-bit codes (5'->3').
//   genome : the reference as 2-bit codes, one byte per base (length = genome_len).
//   The number of candidate windows is n_windows = genome_len - WINDOW_LEN + 1
//   (every start position at which a full 23-base window still fits).
// ---------------------------------------------------------------------------
struct CrisprProblem {
    std::vector<uint8_t> guide;    // [GUIDE_LEN]
    std::vector<uint8_t> genome;   // [genome_len], 2-bit codes
    int genome_len = 0;            // bases in the genome
    int n_windows  = 0;            // candidate sites = genome_len - WINDOW_LEN + 1
    std::string guide_name;        // label from the data file (for the report)
};

// ---------------------------------------------------------------------------
// Per-window results, in Structure-of-Arrays form (one entry per genome window,
// indexed by the window's start position). SoA keeps each array contiguous so
// the GPU writes are coalesced and the host can scan them linearly.
//   mismatches[i] : guide/protospacer mismatches at window i, or -1 if no PAM.
//   cfd[i]        : CFD off-target score at window i in [0,1] (0 if no PAM).
// ---------------------------------------------------------------------------
struct ScanResult {
    std::vector<int>    mismatches;   // [n_windows]
    std::vector<double> cfd;          // [n_windows]
};

// ---------------------------------------------------------------------------
// load_problem: parse the tiny text dataset documented in data/README.md:
//     line beginning "guide"  -> "guide <NAME> <20-letter ACGT spacer>"
//     line beginning "genome" -> "genome <ACGT string>" (may be one long line)
//   Lines beginning with '#' are comments; blank lines are ignored. Bases are
//   uppercased then encoded. Throws std::runtime_error on a missing file, a
//   wrong-length guide, an unknown base, or a genome too short for one window.
// ---------------------------------------------------------------------------
CrisprProblem load_problem(const std::string& path);

// ---------------------------------------------------------------------------
// scan_cpu: the trusted serial baseline. For every genome window i it calls the
// shared score_window() (cfd_score.h) and stores the mismatch count and CFD
// score. This is what the GPU kernel is verified against (and the timing
// baseline that makes the speed-up legible). Resizes both output arrays to
// prob.n_windows.
// ---------------------------------------------------------------------------
void scan_cpu(const CrisprProblem& prob, ScanResult& out);

// ---------------------------------------------------------------------------
// specificity_score: the guide-level summary metric. Given the SUMMED CFD of
// every off-target window (a window that has a PAM and >= 1 mismatch), the
// CRISPOR/MIT "specificity" is
//     100 / (100 + 100 * sum_of_offtarget_cfd)
// which is 100 for a perfectly specific guide and falls toward 0 as off-target
// burden grows. Defined once here so CPU and GPU paths report the identical
// number. `sum_offtarget_cfd` is the summed CFD over all off-target windows.
// ---------------------------------------------------------------------------
double specificity_score(double sum_offtarget_cfd);
