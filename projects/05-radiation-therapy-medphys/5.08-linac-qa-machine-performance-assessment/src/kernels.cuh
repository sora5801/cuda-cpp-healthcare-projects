// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for the 2-D gamma-index map
// ---------------------------------------------------------------------------
// Project 5.8 : Linac QA & Machine Performance Assessment  (catalog ID 5.8)
//
// ROLE IN THE PROJECT
//   Declares the GPU path that main.cu calls. Included only by .cu translation
//   units because it references a __global__ kernel; the plain C++ compiler must
//   never see it (that is why the CPU reference prototypes live in the separate
//   pure-C++ reference_cpu.h).
//
// THE BIG IDEA  (gather pattern -- PATTERNS.md §1, exemplified by 4.01)
//   The gamma index at each MEASURED pixel is INDEPENDENT of every other measured
//   pixel: it only reads a local window of the reference plane and takes a min.
//   That is a textbook "gather": we assign ONE GPU THREAD PER MEASURED PIXEL,
//   lay a 2-D thread grid over the 2-D plane, and every thread runs the exact
//   same gamma_value_at() (from gamma.h) the CPU reference runs -- so the GPU map
//   equals the CPU map bit-for-bit. No atomics, no shared memory, no data races:
//   each thread writes exactly one output pixel it alone owns.
//
//   The heavy inner search (scanning a (2R+1)^2 reference window per pixel) is
//   what the GPU parallelises across thousands of pixels at once. The pass rate
//   and the flatness/symmetry metrics are cheap host reductions done afterwards.
//
// READ THIS AFTER: gamma.h (the shared math), util/cuda_check.cuh, util/timer.cuh.
// Then read kernels.cu for the implementation, and main.cu for the caller.
// ===========================================================================
#pragma once

#include <vector>

#include "gamma.h"          // GammaParams + gamma_value_at (host/device core)
#include "reference_cpu.h"  // QAProblem (the input bundle)

// ---------------------------------------------------------------------------
// gamma_map_gpu: run the whole GPU gamma computation and return the map.
//   Uploads the two dose planes, launches one thread per measured pixel, copies
//   the resulting gamma map back to the host, and reports the pure KERNEL time
//   (via CUDA events) in *kernel_ms. All device bookkeeping is hidden here so
//   main.cu stays about the SCIENCE, not the plumbing.
//
//   Parameters:
//     q         : the loaded QA problem (holds meas/ref planes + geometry)
//     p         : the gamma tolerances (dd/dta/search radius) -- SAME struct the
//                 CPU reference uses, guaranteeing identical arithmetic
//     gamma_out : host output, resized to nx*ny, filled with per-pixel gamma
//     kernel_ms : out-param, milliseconds spent inside the kernel (not copies)
// ---------------------------------------------------------------------------
void gamma_map_gpu(const QAProblem& q, const GammaParams& p,
                   std::vector<float>& gamma_out, float* kernel_ms);
