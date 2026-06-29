// ===========================================================================
// src/reference_cpu.h  --  Sequence model + amino-acid coding + CPU reference
// ---------------------------------------------------------------------------
// Project 1.34 : Amyloid / Aggregation Propensity Prediction
//
// WHAT THIS PROJECT COMPUTES
//   A SEQUENCE-BASED AGGREGATION-PROPENSITY PROFILE, the tractable, didactic
//   core of the well-known predictors TANGO / AGGRESCAN / Zyggregator. Given a
//   protein's amino-acid sequence we:
//
//     1. Map each residue to an INTRINSIC beta-aggregation propensity (a single
//        number per amino-acid type, from a fixed lookup scale: propensity.h).
//     2. SMOOTH that raw per-residue signal with a centered SLIDING-WINDOW MEAN
//        of width W  ->  s[i] = mean propensity over the window centered on
//        residue i. (Aggregation is driven by short *contiguous* hydrophobic /
//        beta-prone stretches, not by isolated residues, so we average a local
//        window -- exactly what TANGO/AGGRESCAN do.)
//     3. THRESHOLD the smoothed profile: residue i is "aggregation-prone" if
//        s[i] >= THRESH. Contiguous prone residues form an APR (Aggregation-
//        Prone Region / "hot spot"). Per protein we report the peak smoothed
//        score & its position, the count of prone residues, and the longest APR.
//
// WHY A GPU  (the pattern this project teaches: SHARED-MEMORY TILED 1-D CONV)
//   Two nested embarrassing parallelisms:
//     * across PROTEINS  -- each sequence is independent (cf. 1.12 Tanimoto:
//       one query vs N library items): give each protein its own GPU BLOCK;
//     * across RESIDUES  -- each smoothed s[i] is an independent window mean:
//       within a block each THREAD owns one residue.
//   The window mean is a 1-D sliding-window/FIR convolution, so the kernel uses
//   the canonical SHARED-MEMORY TILING + HALO optimization (flagship 7.10): the
//   block stages its protein's per-residue propensities into on-chip shared
//   memory once, then every thread reads its W-wide window from there instead of
//   re-reading global memory W times. Proteome-wide liability screens (antibody
//   developability, whole-proteome aggregation maps) batch tens of thousands of
//   sequences -- that batch is what makes this GPU-bound.
//
// THE CPU/GPU PARITY TRICK (PATTERNS.md §2)
//   The per-residue physics -- the lookup and the windowed mean -- lives in ONE
//   `__host__ __device__` header (propensity.h), so this CPU reference and the
//   GPU kernel run identical arithmetic in identical order. Verification is then
//   exact to ~float epsilon.
//
//   This header is PURE C++ (no CUDA): kernels.cu reuses Protein / Dataset.
//   Read this first in the code tour; see ../THEORY.md for the science & math.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

// One protein: a name (FASTA header) and its residue codes as small integers.
//   We pre-encode each one-letter amino-acid code to an index 0..20 ON LOAD
//   (code 20 = "other"/non-standard), so the hot loops never touch chars and the
//   GPU can read a flat int array. `len` is cached to avoid repeated .size().
struct Protein {
    std::string name;          // FASTA header (without the leading '>')
    std::vector<int> codes;    // [len] amino-acid indices 0..20 (see propensity.h)
    int len = 0;               // == codes.size(); residues in this sequence
};

// PAD_CODE: sentinel index stored in the padded tail of each flat row (below).
//   It maps to 0 propensity (propensity.h), and because the loader also records
//   each row's REAL length, padding never enters any window mean. Declared here
//   (not propensity.h) because the loader needs it too.
static constexpr int PAD_CODE = -1;

// A loaded batch of proteins plus a flat, padded device-friendly layout.
//   To run all proteins in one kernel launch we pack every sequence into a flat
//   `flat_codes` array with a fixed STRIDE (= longest sequence). Row p occupies
//   flat_codes[p*stride .. p*stride + lengths[p]); the tail is PAD_CODE. This
//   "ragged batch -> padded matrix" layout is the standard way to batch
//   variable-length sequences on a GPU (one block per row, coalesced row reads).
struct Dataset {
    std::vector<Protein> proteins;   // parsed sequences (host-side, with names)
    std::vector<int> flat_codes;     // [num * stride] padded codes, row-major
    std::vector<int> lengths;        // [num] real length of each protein
    int num = 0;                     // number of proteins
    int stride = 0;                  // padded row width (>= max length)
    int max_len = 0;                 // longest real sequence (for reporting)
};

// Per-protein result of the aggregation scan (one struct per sequence).
//   The four numbers production tools surface as "aggregation liability": where
//   the worst hot spot is, how hot, how much of the chain is prone, and the
//   longest contiguous APR (long APRs nucleate fibrils fastest).
struct AggResult {
    float peak_score  = 0.0f;  // max smoothed propensity over the chain
    int   peak_pos    = 0;     // residue index (0-based) of that peak
    int   prone_count = 0;     // # residues with smoothed score >= threshold
    int   longest_apr = 0;     // longest run of contiguous prone residues
};

// ---- Loading & batching ---------------------------------------------------

// Map a single one-letter amino-acid code to its index (0..20). Unknown symbols
// (including lowercase, gaps, 'X') return 20, the "other" bucket. Defined in
// reference_cpu.cpp; exposed so tests can check the encoding.
int code_of_char(char c);

// Parse a FASTA-style file (see data/README.md) into a Dataset. Lines starting
// with '>' begin a new protein; following lines are appended residues. Throws
// std::runtime_error on open failure or an empty file so demos fail loudly.
Dataset load_dataset(const std::string& path);

// Build the flat padded layout (flat_codes/lengths/stride/num/max_len) from the
// already-parsed proteins. Called by load_dataset; exposed for clarity/tests.
void build_flat_layout(Dataset& ds);

// ---- The CPU reference (the trusted baseline) -----------------------------

// scan_dataset_cpu: for every protein, compute the smoothed propensity profile
// (windowed mean from propensity.h) and reduce it to an AggResult. This is the
// serial reference the GPU kernel is verified against; it calls the SAME
// per-residue functions the kernel uses, so the two agree to ~float epsilon.
//   ds        : the loaded batch
//   window    : sliding-window width W (odd; half = (W-1)/2)
//   threshold : a residue is "prone" if its smoothed score >= this
//   results   : [num] output, one AggResult per protein
//   smoothed  : OPTIONAL [num*stride] flat smoothed profiles (for verification
//               and for printing one profile in the demo); pass nullptr to skip.
void scan_dataset_cpu(const Dataset& ds, int window, float threshold,
                      std::vector<AggResult>& results,
                      std::vector<float>* smoothed);
