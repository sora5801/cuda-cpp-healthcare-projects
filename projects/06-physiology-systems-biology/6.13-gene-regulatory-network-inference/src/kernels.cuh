// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for GRN inference (MI + DPI)
// ---------------------------------------------------------------------------
// Project 6.13 : Gene Regulatory Network Inference (ARACNE)
//
// THE BIG IDEA
//   Scoring the mutual information (MI) of every gene pair is an O(G^2) set of
//   INDEPENDENT jobs -- exactly the "many independent items" pattern from the
//   Tanimoto flagship (1.12), but the "items" are the G*(G-1)/2 unordered pairs
//   of a G x S expression matrix. We map ONE PAIR -> ONE THREAD:
//     * each thread builds its pair's B x B joint histogram in a private
//       register/local array (JOINT_CELLS = 64 ints for B=8), then calls the
//       SHARED core mi_from_joint() (grn.h) -- so the number it produces is the
//       same one the CPU produces from the same integer counts;
//     * the discretized matrix (small, read by every thread) is uploaded once;
//     * a grid-stride loop lets a modest grid cover any number of pairs.
//   Then a second O(G^3) kernel applies the Data Processing Inequality (DPI),
//   one thread per candidate edge, testing every mediator k.
//
//   Determinism: the joint histogram is INTEGER counting (order-independent) and
//   MI is evaluated from those exact counts, so stdout is byte-identical every
//   run and matches the CPU to ~1e-12 (PATTERNS.md sec 3-4). No atomics needed:
//   each thread owns a disjoint output cell.
//
//   This header is included only by .cu units. main.cu calls grn_infer_gpu().
//
// READ THIS AFTER: grn.h, reference_cpu.h, util/cuda_check.cuh, util/timer.cuh.
// Then read kernels.cu. The science / GPU-mapping is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // GrnData, N_BINS, JOINT_CELLS (pure C++, safe in .cu)

// ---------------------------------------------------------------------------
// grn_infer_gpu : run the whole GPU pipeline and return results to verify.
//   Steps (mirrored from the CPU reference, all on-device):
//     1. upload raw expression; discretize it in a kernel (per-gene binning);
//     2. one-thread-per-pair MI kernel -> dense symmetric G*G MI matrix;
//     3. one-thread-per-edge DPI kernel -> G*G keep mask.
//   Outputs (host-side, resized here):
//     mi   : [G*G] mutual information in nats (symmetric, zero diagonal)
//     keep : [G*G] 1 = direct edge kept, 0 = pruned/below-threshold
//   Params:
//     data         : loaded dataset (expr used; disc recomputed on device)
//     mi_threshold : minimum MI (nats) for an edge to be considered at all
//     tolerance    : DPI slack (see dpi_prune_cpu)
//     mi_ms/dpi_ms : out-params, GPU-measured kernel times (ms), for teaching
// ---------------------------------------------------------------------------
void grn_infer_gpu(const GrnData& data,
                   double mi_threshold, double tolerance,
                   std::vector<double>& mi, std::vector<uint8_t>& keep,
                   float* mi_ms, float* dpi_ms);
