// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for MEME motif finding
// ---------------------------------------------------------------------------
// Project 3.29 : Motif Finding in Genomic Sequences
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the shared DATA MODEL (the sequence set, the PWM, the EM
//   driver, the file loader) lives here. kernels.cuh also includes this header
//   to reuse the SequenceSet type and the constants -- nothing CUDA-specific
//   leaks in either direction. The per-window score formula that BOTH sides must
//   compute identically lives in motif_core.h (the HD-macro header).
//
// THE PROBLEM (see ../THEORY.md for the full derivation)
//   Given N DNA sequences (e.g. ChIP-seq peak regions) that share an unknown,
//   over-represented pattern of width W (a transcription-factor binding motif),
//   recover that motif. We model the motif as a POSITION WEIGHT MATRIX (PWM):
//   a W x 4 table where PWM[p][b] is the probability of base b at motif column
//   p. The rest of each sequence is "background" (i.i.d. base frequencies bg[]).
//
//   MEME's Expectation-Maximisation finds the PWM that maximises the data
//   likelihood under the OOPS model (One Occurrence Per Sequence -- each
//   sequence contains exactly one motif instance at an unknown offset):
//     * E-step: for every sequence, score EVERY length-W window (the expensive,
//       embarrassingly-parallel part -> the GPU kernel), then turn the scores
//       into a probability distribution over offsets (the motif's likely site).
//     * M-step: re-estimate the PWM as the responsibility-weighted base counts
//       over all windows, add pseudocounts, renormalise.
//   Iterate to convergence. This file is the trusted serial implementation; the
//   GPU twin (kernels.cu) replaces ONLY the E-step window scoring.
//
// READ THIS BEFORE: motif_core.h, reference_cpu.cpp, kernels.cuh.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "motif_core.h"   // MOTIF_ALPHABET, MOTIF_BASE_N, window_score (HD)

// ---------------------------------------------------------------------------
// SequenceSet : the loaded problem.
//   All N sequences are stored CONCATENATED into one flat byte array `data`
//   (encoded bases 0..4), with per-sequence offsets/lengths -- a "ragged array"
//   laid out contiguously. This is exactly the layout the GPU wants: one big
//   coalesced buffer plus an index, instead of N separate allocations.
//
//   Why concatenated + offsets (CSR-style)?  Sequences have different lengths.
//   A 2-D padded array would waste memory and bandwidth on the padding; the
//   flat layout lets the kernel address window (seq s, start j) as
//   data[ offset[s] + j ] with no padding (THEORY sec "GPU mapping"). A flat
//   list of ALL windows (seq_of_win / start_of_win) then turns the irregular
//   "N sequences of varying length" workload into ONE 1-D grid of independent
//   window-scoring jobs -- the cleanest possible GPU mapping.
// ---------------------------------------------------------------------------
struct SequenceSet {
    int n = 0;                              // number of sequences
    int w = 0;                              // motif width W (columns of the PWM)
    std::vector<unsigned char> data;        // concatenated encoded bases (0..4)
    std::vector<int> offset;                // [n+1] CSR offsets into `data`
    std::vector<int> length;                // [n] length of each sequence
    std::vector<int> win_off;               // [n+1] CSR offsets into the WINDOW list
    std::vector<int> seq_of_win;            // [total_windows] seq each window belongs to
    std::vector<int> start_of_win;          // [total_windows] window start inside its seq

    int total_windows() const { return win_off.empty() ? 0 : win_off[n]; }
};

// ---------------------------------------------------------------------------
// MotifModel : a PWM plus its derived log-odds table.
//   pwm     : [w * 4] row-major motif probabilities (each row sums to 1).
//   bg      : [4]     background base probabilities (the per-window null model).
//   logodds : [w * 4] log2(pwm/bg) -- the table window_score() reads. Rebuilt
//             from pwm+bg every EM iteration by build_logodds().
// ---------------------------------------------------------------------------
struct MotifModel {
    int w = 0;
    std::vector<float> pwm;       // [w*4]
    std::vector<float> bg;        // [4]
    std::vector<float> logodds;   // [w*4]
};

// ---------------------------------------------------------------------------
// EMResult : everything the demo reports, in a DETERMINISTIC form.
//   consensus    : the recovered motif as an A/C/G/T string (argmax base per
//                  column) -- the headline answer.
//   info_content : total information content of the final PWM in bits
//                  (sum over columns of  2 + sum_b pwm*log2(pwm) ), a standard
//                  motif-quality score; higher = sharper motif.
//   best_site    : [n] the argmax window start per sequence (the predicted
//                  binding-site offset), under the final model.
//   iters        : EM iterations actually run.
//   final_scores : [total_windows] the E-step window log-odds under the FINAL
//                  model. This is the array we VERIFY GPU-vs-CPU on (it is the
//                  exact output of the parallelised step).
// ---------------------------------------------------------------------------
struct EMResult {
    std::string consensus;
    double info_content = 0.0;
    std::vector<int> best_site;
    int iters = 0;
    std::vector<float> final_scores;
};

// Encode one DNA character to {A=0,C=1,G=2,T=3, other=4}. Case-insensitive.
unsigned char encode_base(char c);

// Decode a base index 0..3 to its letter (for the consensus string).
char decode_base(int b);

// Load a sequence set from the FASTA-like text format in data/README.md and
// precompute the window index (offsets, seq_of_win, start_of_win) for width w.
// Throws std::runtime_error on a missing file or if any sequence is shorter
// than w (so it has no valid window).
SequenceSet load_sequences(const std::string& path, int w);

// Build the log-odds table logodds[p*4+b] = log2(pwm[p*4+b] / bg[b]) from the
// model's pwm + bg. Called once per EM iteration before the E-step.
void build_logodds(MotifModel& model);

// CPU E-step: fill scores[win] = window_score(...) for EVERY window, using the
// SAME motif_core.h formula the GPU kernel uses. This is the trusted baseline
// and the timing reference for the GPU twin. `scores` is resized to
// set.total_windows().
void score_windows_cpu(const SequenceSet& set, const MotifModel& model,
                       std::vector<float>& scores);

// Run the full MEME OOPS EM loop on the CPU to convergence, returning the
// recovered motif + per-sequence sites + the final-model window scores. This
// drives the whole demo; the GPU only accelerates score_windows_cpu's job, and
// main.cu verifies the GPU E-step against this CPU E-step on the final model.
//   set       : the sequences + precomputed window index (read-only)
//   model     : in/out -- caller seeds w, bg, and an initial pwm; on return it
//               holds the converged PWM and its log-odds table.
//   max_iters : hard cap on EM iterations (safety net).
//   tol       : convergence threshold on the change in total log-likelihood.
EMResult run_meme_em_cpu(const SequenceSet& set, MotifModel& model,
                         int max_iters, double tol);
