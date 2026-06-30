// ===========================================================================
// src/kernels.cuh  --  GPU alignment-search interface (cuFFT cross-correlation)
// ---------------------------------------------------------------------------
// Project 2.4 : Cryo-ET Subtomogram Averaging  (reduced-scope teaching version)
//
// THE BIG IDEA  (pattern: USE A CUDA LIBRARY + independent batched jobs)
//   Aligning a candidate to the reference means finding the rotation AND the
//   3-D translation that maximize cross-correlation. Computing correlation by
//   shifting one cube over the other costs O(V) per shift and there are O(V)
//   shifts -> O(V^2) per (candidate, angle). The CROSS-CORRELATION THEOREM
//   collapses that to O(V log V):
//
//       corr(ref, g) = IFFT( conj(FFT(ref)) .* FFT(g) )
//
//   The value at output voxel (0,0,0) is the correlation at ZERO shift; the
//   global PEAK over the whole IFFT output is the best translational alignment.
//   cuFFT computes the forward/inverse 3-D FFTs; tiny custom kernels do the
//   rotation, the per-frequency complex multiply, and the peak reduction.
//
//   We BATCH the work: every (candidate, angle) pair is an independent job, so
//   we rotate all n_sub*n_angles cubes, run ONE batched cuFFT over all of them,
//   multiply each by the (single, shared) reference spectrum, inverse-FFT the
//   batch, and reduce each cube to its (peak, zero-shift) pair. This is the
//   "score one reference vs many candidates, each independent" pattern
//   (PATTERNS.md §1) layered on top of cuFFT (PATTERNS.md §5).
//
//   main.cu calls align_gpu(); kernels.cu implements it.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // SubtomogramSet, trial_angle (pure C++, safe in .cu)

// ---------------------------------------------------------------------------
// align_gpu: GPU alignment search for every candidate over every trial angle.
//
//   Pipeline (all device-side except the final small copies):
//     1. rotate_kernel : produce the rotated candidate for each (cand, angle).
//     2. cuFFT R2C (batched) : FFT every rotated cube + the reference cube.
//     3. xcorr_mul_kernel : per frequency, conj(REF) .* CAND (the theorem).
//     4. cuFFT C2R (batched) : inverse-FFT back to the correlation field.
//     5. reduce_kernel : per cube, find the PEAK correlation and read the
//        ZERO-SHIFT value; both are normalized to NCC by the cube energies.
//
//   Inputs:
//     set : the loaded, zero-meaned subtomograms (host side).
//   Outputs (resized by the wrapper):
//     ncc_zero_shift[s*n_angles + k] : zero-shift NCC (verified against the CPU)
//     ncc_peak[s*n_angles + k]       : best-over-all-shifts NCC (the real metric)
//     best_angle[s]                  : argmax over angles of ncc_peak (ties->low k)
//     kernel_ms : out-param, GPU time of the FFT + kernels (CUDA events).
//
//   Determinism: every reduction here is over floats but uses a fixed,
//   index-ordered scan (not atomics), so the result is bit-stable run to run.
// ---------------------------------------------------------------------------
void align_gpu(const SubtomogramSet& set,
               std::vector<double>& ncc_zero_shift,
               std::vector<double>& ncc_peak,
               std::vector<int>& best_angle,
               float* kernel_ms);

// ---- Device kernels (declared here, defined + documented in kernels.cu) ----

// rotate_kernel: in-plane rotation about z with bilinear interpolation, the GPU
//   twin of rotate_cube_cpu(). One thread per output voxel of one (cand,angle)
//   job; the grid's z-dimension selects the job. Identical arithmetic to the CPU
//   so the rotated cubes match voxel-for-voxel.
__global__ void rotate_kernel(const float* __restrict__ cand, // [n_sub * V]
                              float* __restrict__ out,         // [n_jobs * V]
                              int d, int n_sub, int n_angles);

// xcorr_mul_kernel: per complex frequency bin, multiply each job's spectrum by
//   the conjugate of the reference spectrum (the cross-correlation theorem).
//   One thread per (job, frequency-bin). Writes back in place into the job's
//   spectrum buffer.
__global__ void xcorr_mul_kernel(float2* __restrict__ job_spec,        // [n_jobs * nfreq]
                                 const float2* __restrict__ ref_spec,  // [nfreq]
                                 int nfreq, int n_jobs);

// reduce_kernel: for each job's correlation field, find the peak value and read
//   the zero-shift value, then normalize both to NCC using the precomputed cube
//   energies. One block per job; a deterministic, index-ordered block reduction.
__global__ void reduce_kernel(const float* __restrict__ corr,  // [n_jobs * V]
                              const float* __restrict__ job_energy,  // [n_jobs]
                              float ref_energy, int V, float invV,
                              float* __restrict__ out_zero,    // [n_jobs]
                              float* __restrict__ out_peak);   // [n_jobs]
