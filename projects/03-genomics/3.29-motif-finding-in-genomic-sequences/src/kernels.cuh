// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for the MEME E-step
// ---------------------------------------------------------------------------
// Project 3.29 : Motif Finding in Genomic Sequences
//
// THE BIG IDEA
//   MEME's Expectation-Maximisation spends ~all its time in the E-step: score
//   EVERY length-W window of EVERY sequence against the current motif model.
//   With N sequences of total length L there are ~L window positions, and each
//   score is an independent W-term dot product against the log-odds table.
//   That is a perfect "many independent jobs" workload (PATTERNS.md sec 1, the
//   same pattern as 1.12 Tanimoto and 12.01 spectral search): we give each
//   window its OWN GPU THREAD.
//
//   Two GPU features carry this project:
//     * the W x 4 LOG-ODDS table lives in CONSTANT memory -- every thread reads
//       the same table, never writes it -> the constant cache broadcasts one
//       address to a whole warp in a single transaction (ideal for a small
//       read-only lookup table), and
//     * a grid-stride loop lets one modest grid cover millions of windows.
//
//   The kernel is the GPU twin of score_windows_cpu() in reference_cpu.cpp;
//   BOTH call the same __host__ __device__ window_score() (motif_core.h), so the
//   results match bit-for-bit and main.cu can verify with an EXACT tolerance.
//   The host (reference_cpu.cpp) still runs the cheap E/M bookkeeping (softmax,
//   count accumulation); only the heavy window scoring is offloaded -- which is
//   exactly how mCUDA-MEME structures the real tool (THEORY sec "real world").
//
// READ THIS AFTER: motif_core.h, reference_cpu.h, util/*. Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // SequenceSet, MotifModel (pure C++, safe in .cu)
#include "motif_core.h"      // MOTIF_ALPHABET, window_score

// Maximum motif width the constant-memory log-odds table can hold. The DNA
// log-odds table is w*4 floats; with MAX_W=64 that is 64*4*4 = 1 KiB, trivially
// inside the 64 KiB constant bank. Picked generously -- real TF motifs are
// 6..30 bp -- so the demo and any reasonable experiment fit. Asserted at upload.
constexpr int MAX_W = 64;

// ---- Device kernel -------------------------------------------------------
// score_windows_kernel: one logical thread per window, via a grid-stride loop.
//   data        : [total bases] concatenated encoded sequences (device copy of
//                 SequenceSet::data); bytes in {0,1,2,3}
//   start_of_win: [num_windows] absolute start index of each window into `data`
//   num_windows : number of windows (guards the ragged last block)
//   w           : motif width W
//   out         : [num_windows] output log-odds scores (one per window)
//   The log-odds table is read from the __constant__ symbol filled by the host
//   wrapper (not a parameter) -- see kernels.cu.
__global__ void score_windows_kernel(const unsigned char* __restrict__ data,
                                     const int* __restrict__ start_of_win,
                                     int num_windows, int w,
                                     float* __restrict__ out);

// ---- Host wrapper --------------------------------------------------------
// score_windows_gpu: do the whole E-step scoring on the GPU.
//   Uploads the sequence bytes + window starts + the current log-odds table
//   (to constant memory), launches the kernel, copies the scores back, and
//   reports the measured KERNEL time (CUDA events) via *kernel_ms. This is the
//   GPU twin of score_windows_cpu(); main.cu runs both on the FINAL model and
//   asserts they agree.
//
//   set       : the loaded sequences + window index (host)
//   model     : supplies the W x 4 log-odds table to broadcast (must be built)
//   out       : host output, resized to set.total_windows() (output parameter)
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
void score_windows_gpu(const SequenceSet& set, const MotifModel& model,
                       std::vector<float>& out, float* kernel_ms);
