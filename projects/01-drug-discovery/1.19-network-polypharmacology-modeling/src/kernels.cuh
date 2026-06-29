// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for TransE link prediction
// ---------------------------------------------------------------------------
// Project 1.19 : Network / Polypharmacology Modeling
//
// THE BIG IDEA (pattern: INDEPENDENT JOBS + CONSTANT-MEMORY QUERY, cf. 1.12)
//   Scoring a query drug against N candidate protein tails is N INDEPENDENT jobs:
//   tail j's TransE score depends only on the (shared) head h, the (shared)
//   relation r, and tail j's own embedding. So we give each candidate tail its
//   OWN GPU THREAD. Two CUDA features make this both fast and a good lesson:
//     * the head and relation vectors live in CONSTANT memory -- every thread
//       reads them, none writes them, and they are identical for the whole launch
//       -> the constant cache broadcasts one address to a whole warp in a single
//       transaction (vs. re-reading them from global memory per thread); and
//     * a grid-stride loop lets one modest grid cover an arbitrarily large
//       candidate set (a real STRING/STITCH graph has tens of thousands of
//       proteins; a trained KG can have millions of entities).
//
//   The per-tail math is the SHARED transe_score() in transe.h, called by both
//   this kernel and the CPU reference -> the results match exactly. kernels.cu
//   defines the kernel and the host wrapper.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, transe.h, reference_cpu.h.
// Then read kernels.cu. The science/GPU-mapping is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // KnowledgeGraph (pure C++, safe to include in .cu)

// Maximum embedding dimension we reserve room for in constant memory. The head
// and relation each occupy `dim` floats; 256 floats = 1 KB each, so both fit
// comfortably in the 64 KB constant bank with room to spare. A real trained
// TransE model uses dim ~ 50..200, well under this cap. (If you ever need a
// larger dim, move head/relation to global memory -- see THEORY "GPU mapping".)
constexpr int MAX_DIM = 256;

// Device kernel: out[j] = transe_score(head, relation, tail_j). The head and
// relation are read from __constant__ symbols defined in kernels.cu (not params).
//   tails : [n * dim] row-major device array of candidate tail embeddings
//   n     : number of candidate tails
//   dim   : embedding dimension
//   out   : [n] device array of plausibility scores (output)
__global__ void transe_kernel(const float* __restrict__ tails, int n, int dim,
                              float* __restrict__ out);

// Host wrapper: upload head+relation to constant memory and the tails to global
// memory, launch the kernel, time it (CUDA events), and return the scores.
//   kg        : the loaded query (head + relation + n tail embeddings)
//   out       : resized to kg.n; filled with per-candidate TransE scores
//   kernel_ms : out-param, GPU-measured kernel time in milliseconds
void transe_score_gpu(const KnowledgeGraph& kg, std::vector<float>& out, float* kernel_ms);
