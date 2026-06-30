// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 2.15 : Antibody Structure Prediction  (reduced-scope: CDR screening)
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls score_gpu(); kernels.cu
//   implements both the host wrapper and the device kernel. Included only by
//   .cu translation units (it declares a __global__ kernel, so the plain C++
//   compiler must never see it -- that is why the data model + CPU reference
//   live in the pure-C++ reference_cpu.h, and the scoring math in antibody.h).
//
// THE BIG IDEA (PATTERNS.md §1: "score one query vs N items, each independent")
//   We compare ONE query antibody's six CDR loops against N library antibodies.
//   Every comparison is fully independent, so we give each library antibody its
//   own GPU THREAD (a grid-stride loop lets a fixed grid cover any N). This is
//   the same pattern as project 1.12 (Tanimoto search): a small, read-only query
//   broadcast to every thread, a big library streamed from global memory, and an
//   independent score written per item.
//
//   THE QUERY GOES IN CONSTANT MEMORY. All N threads read the same 144-byte query
//   record but never write it; constant memory's broadcast cache serves one
//   address to a whole warp in a single transaction -- ideal for a value every
//   thread reads identically.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, antibody.h. Then kernels.cu.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // AntibodyLibrary (the shared data model)

// ---- Host wrapper --------------------------------------------------------
// score_gpu: the host-callable "do the whole GPU computation" function.
//   Uploads the query to constant memory, allocates + uploads the library,
//   launches the scoring kernel (one thread per library antibody via a
//   grid-stride loop), copies the integer scores back, and reports the measured
//   KERNEL time (CUDA events) via *kernel_ms. main.cu calls exactly this; all
//   CUDA bookkeeping is hidden here.
//
//   ab        : the loaded dataset (query + n library antibodies, encoded)
//   out       : host output scores, resized to ab.n (output parameter, int32)
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
//
//   The scores are int32 and match score_cpu() EXACTLY -- the math is shared
//   integer arithmetic from antibody.h, so verification is bit-for-bit.
void score_gpu(const AntibodyLibrary& ab, std::vector<int32_t>& out, float* kernel_ms);
