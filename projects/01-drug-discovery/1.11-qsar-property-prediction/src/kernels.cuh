// ===========================================================================
// src/kernels.cuh  --  GPU GCN inference interface
// ---------------------------------------------------------------------------
// Project 1.11 : QSAR / Property Prediction
//
// THE BIG IDEA  (pattern: PER-OUTPUT GATHER over a CSR graph -- PATTERNS.md §1)
//   A GCN layer recomputes every node's feature row from its neighbors. We give
//   each OUTPUT NODE its own GPU thread; that thread walks the node's neighbor
//   list (CSR slice) and accumulates the normalized, W-projected messages. Two
//   layers = two kernel launches over all nodes. A third kernel gives each
//   MOLECULE a thread to mean-pool its atoms and apply the linear head.
//
//   Because each thread writes ONLY its own node/molecule, there is NO atomic
//   scatter and NO race -- and the neighbor sum runs in the SAME CSR order as the
//   CPU reference, so GPU and CPU agree to fp32 rounding. The per-node math is
//   the shared gcn.h (GCN_HD) code, identical on both sides (PATTERNS.md §2).
//
//   kernels.cu defines the kernels + the host wrapper gcn_predict_gpu(); main.cu
//   calls the wrapper and compares against gcn_predict_cpu().
//
// READ THIS AFTER: gcn.h, reference_cpu.h, util/cuda_check.cuh, util/timer.cuh.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // Graph, Model (plain C++, safe to include in .cu)

// ---------------------------------------------------------------------------
// gcn_predict_gpu: run the full 2-layer GCN + readout on the GPU.
//   Inputs : the batched graph `g` and the fixed weights `m`.
//   Output : `pred` (num_mols predictions), filled to match gcn_predict_cpu().
//   `kernel_ms` receives the GPU time for the three kernels (CUDA events),
//   reported to STDERR as a teaching artifact (never a benchmark claim).
//
//   The wrapper owns all device memory: it uploads the CSR + features + weights
//   once, launches layer-1, layer-2, then the readout kernel, and copies the
//   predictions back. See kernels.cu for the launch configs and the thread->data
//   mapping.
// ---------------------------------------------------------------------------
void gcn_predict_gpu(const Graph& g, const Model& m,
                     std::vector<float>& pred, float* kernel_ms);
