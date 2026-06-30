// ===========================================================================
// src/kernels.cuh  --  GPU CTF-estimation interface (cuFFT + radial avg + search)
// ---------------------------------------------------------------------------
// Project 2.11 : Cryo-EM CTF Estimation & Particle Picking
//
// THE BIG IDEA (two GPU patterns in one project)
//   1. USE A LIBRARY KERNEL WITHOUT IT BEING A BLACK BOX. The micrograph power
//      spectrum is a 2-D FFT -- a solved problem with a superb GPU library, cuFFT.
//      kernels.cu documents exactly what cufftExecR2C computes, the batched layout
//      it expects, and what hand-rolling would take (CLAUDE.md §6.1.6). We add only
//      tiny custom kernels for the |X|^2 power and the rotational average.
//
//   2. INDEPENDENT JOBS + CONSTANT MEMORY. The defocus search is one thread per
//      CANDIDATE defocus: every thread reads the same observed radial profile
//      (placed in CONSTANT memory, whose broadcast cache is ideal for read-by-all
//      data) and scores its own dz with the shared ncc_model_vs_profile(). This is
//      the same pattern as the 1.12 Tanimoto flagship (query in constant memory,
//      one thread per library item). See PATTERNS.md §1.
//
//   PIPELINE on the device:
//     image --cuFFT R2C--> half-spectrum --power_kernel--> |X|^2
//           --radial_average_kernel (atomic, FIXED-POINT)--> raw radial profile
//           (host flattens background; copies profile to __constant__)
//           --ctf_search_kernel (1 thread / candidate dz)--> NCC score curve
//           (host argmax) --> best defocus.
//
//   The whole physics (ctf_squared, ncc_model_vs_profile) lives in ctf_model.h and
//   is shared verbatim with the CPU reference, so GPU and CPU agree exactly.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, ctf_model.h, reference_cpu.h
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // Micrograph, CtfFitConfig, CtfFitResult (pure C++)

// ---------------------------------------------------------------------------
// radial_power_profile_gpu: stages 1+2 on the GPU.
//   Computes the cuFFT 2-D power spectrum of the micrograph and rotationally
//   averages it into a raw radial profile of length `nbins` (== n/2). Returns the
//   RAW (pre-background-flattening) profile in `raw` so main.cu can apply the same
//   flatten_background() as the CPU path (guaranteeing identical post-processing).
//   `kernel_ms` receives the GPU time of the FFT + power + radial-average steps.
// ---------------------------------------------------------------------------
void radial_power_profile_gpu(const Micrograph& m, int nbins,
                              std::vector<double>& raw, float* kernel_ms);

// ---------------------------------------------------------------------------
// fit_ctf_gpu: stage 3 on the GPU. Uploads the (already flattened) radial profile
// to constant memory, launches one thread per candidate defocus to fill the NCC
// score curve, copies it back, and takes the argmax on the host (a tiny n_dz-long
// reduction not worth a second kernel). Returns the full score curve + best dz and
// writes the GPU kernel time to `kernel_ms`.
// ---------------------------------------------------------------------------
CtfFitResult fit_ctf_gpu(const std::vector<double>& prof, const CtfParams& optics,
                         const CtfFitConfig& cfg, float* kernel_ms);
