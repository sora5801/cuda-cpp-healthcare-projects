// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for the CRISPR off-target scan
// ---------------------------------------------------------------------------
// Project 3.17 : CRISPR Guide Design & Off-Target Scoring
//
// THE BIG IDEA
//   Scoring a guide against a genome means scoring it against EVERY 23-base
//   window independently. That is millions of windows for a real genome, all
//   independent -> the textbook GPU pattern: ONE THREAD PER GENOME POSITION.
//   Two CUDA features make it efficient and are the teaching points here:
//     * the 20-base GUIDE lives in CONSTANT memory -- every thread reads all 20
//       guide bases but none writes them, so the constant cache broadcasts them
//       warp-wide in one transaction (same trick as the query in flagship 1.12);
//     * a GRID-STRIDE loop lets one modest grid cover an arbitrarily long genome.
//   The per-window math itself is NOT re-implemented here: the kernel calls the
//   shared score_window() from cfd_score.h, the exact function the CPU uses, so
//   the two results are bit-identical (PATTERNS.md §2).
//
//   This header is included only by .cu units (it declares a __global__). The
//   pure-C++ data model is in reference_cpu.h; main.cu calls scan_gpu().
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, cfd_score.h,
// reference_cpu.h. Then read kernels.cu.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // CrisprProblem, ScanResult (pure C++, safe in .cu)

// ---- Device kernel -------------------------------------------------------
// scan_kernel: one logical thread per genome window, via a grid-stride loop.
//   genome      : [genome_len] device array of 2-bit base codes
//   n_windows   : number of candidate windows = genome_len - WINDOW_LEN + 1
//   d_mismatch  : [n_windows] output, mismatches per window (-1 if no PAM)
//   d_cfd       : [n_windows] output, CFD off-target score per window (double)
// The guide is NOT a parameter -- it is read from the __constant__ symbol that
// kernels.cu defines and fills via cudaMemcpyToSymbol (see the .cu file).
__global__ void scan_kernel(const uint8_t* __restrict__ genome, int n_windows,
                            int* __restrict__ d_mismatch,
                            double* __restrict__ d_cfd);

// ---- Host wrapper --------------------------------------------------------
// scan_gpu: do the whole GPU computation. Uploads the guide to constant memory
// and the genome to global memory, launches scan_kernel, copies the per-window
// results back, and reports the measured KERNEL time (CUDA events) via
// *kernel_ms. main.cu calls exactly this; all CUDA bookkeeping is hidden here.
//   prob       : the loaded scan job (guide + genome)
//   out        : filled with per-window mismatches + CFD (arrays sized n_windows)
//   kernel_ms  : out-param, milliseconds spent in the kernel itself (not copies)
void scan_gpu(const CrisprProblem& prob, ScanResult& out, float* kernel_ms);
