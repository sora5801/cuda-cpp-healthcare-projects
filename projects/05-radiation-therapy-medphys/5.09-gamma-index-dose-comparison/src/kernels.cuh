// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 5.9 -- Gamma-Index Dose Comparison
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls gamma_map_gpu(); kernels.cu
//   implements both the host wrapper and the device kernel. Included only by
//   .cu translation units (it contains a __global__ declaration, so the plain
//   C++ compiler must never see it -- that is why the CPU reference lives behind
//   the pure-C++ reference_cpu.h instead).
//
// THE BIG IDEA (the core CUDA mapping for this project)
//   The gamma index at each reference voxel is an INDEPENDENT search: voxel i's
//   answer depends only on reading nearby evaluated doses, never on another
//   voxel's answer. That independence is the whole reason the GPU wins -- we
//   assign ONE THREAD PER REFERENCE VOXEL. Each thread scans the evaluated
//   voxels inside a fixed physical window, keeps a running MINIMUM of the
//   squared gamma term (gamma_core.h) in a register, and writes one sqrt at the
//   end. This is the "gather + per-thread min-reduction" pattern (PATTERNS.md
//   §1, closest flagship 4.01 CT backprojection, which also gathers per output
//   pixel over a set of inputs).
//
//   The reference voxel grid is 2-D, so we launch a 2-D grid of 2-D blocks --
//   thread (gx, gy) owns reference voxel (rx=gx, ry=gy). See kernels.cu for the
//   launch-config reasoning and ../THEORY.md §4 for the full mapping.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, gamma_core.h. Then
// read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "dose_problem.h"   // DoseProblem (inputs) -- pure C++, safe here

// ---- Host wrapper --------------------------------------------------------
// gamma_map_gpu: the host-callable "do the whole GPU gamma computation".
//   Allocates device buffers for both dose maps and the output, copies the
//   inputs H2D, launches the gamma kernel over a 2-D grid, copies the gamma map
//   D2H, and reports the measured KERNEL time (CUDA events) via *kernel_ms.
//   main.cu calls exactly this; all CUDA bookkeeping is hidden inside.
//
//   prob      : the two dose maps, grid geometry, and acceptance criteria.
//   gamma_out : host output, resized to prob.size(); gamma_out[i] is the gamma
//               index at reference voxel i (dimensionless; <= 1 == passes).
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies).
//
//   Postcondition: gamma_out is bit-identical to gamma_map_cpu(prob, .) because
//   both call gamma_sq_term() from gamma_core.h over the same fixed candidate
//   set (THEORY §6).
void gamma_map_gpu(const DoseProblem& prob, std::vector<float>& gamma_out,
                   float* kernel_ms);
