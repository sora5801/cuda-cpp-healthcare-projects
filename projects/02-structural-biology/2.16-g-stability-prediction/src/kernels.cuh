// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for the ΔΔG saturation scan
// ---------------------------------------------------------------------------
// Project 2.16 : ΔΔG Stability Prediction (reduced-scope teaching version)
//
// THE BIG IDEA  (PATTERNS.md §1: "score one query vs N items, each independent")
//   A saturation-mutagenesis scan is an L × 20 grid of INDEPENDENT ΔΔG scores:
//   cell (p,a) = predicted ΔΔG of mutating residue p to amino acid a. No cell
//   depends on any other, so we give each of the L*20 cells its own GPU thread.
//   This is the "batched masked prediction" the catalog describes for real
//   ΔΔG models (ThermoMPNN / ProteinMPNN-ddG): the structure is fixed and every
//   (position, mutant-AA) query is evaluated in parallel.
//
//   Two small CUDA ideas carry the lesson:
//     * the per-residue wild-type codes and burial fractions are tiny and read
//       by many threads but never written during the launch, so they go in
//       CONSTANT memory (broadcast cache) -- see kernels.cu;
//     * a 2-D thread block maps naturally onto the (position, amino-acid) grid,
//       and a grid-stride over positions lets one modest grid cover any length L.
//
//   This header is included only by .cu units. It re-uses the Protein type and
//   NUM_AA from reference_cpu.h / ddg_model.h so the host and device share one
//   data model and one scoring function. main.cu calls ddg_scan_gpu().
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h,
//   ddg_model.h. Then read kernels.cu. The GPU mapping is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // Protein, NUM_AA (pure C++, safe to include in .cu)

// The maximum residue length the constant-memory feature buffers can hold. Our
// committed sample is tiny (a few dozen residues); this cap keeps the constant
// arrays small (well within the 64 KB constant bank) while comfortably covering
// any teaching-sized protein. Larger proteins would move these to global memory
// (an exercise in the README). Declared here so main.cu can validate against it.
constexpr int MAX_RESIDUES = 4096;

// Device kernel: out[p*NUM_AA + a] = ddg_predict(wt_p, a, buried_p) for the cell
// (position p, mutant amino acid a). The wild-type codes and burial fractions
// are read from __constant__ symbols defined in kernels.cu (not parameters).
//   L   : number of residues (scan has L*NUM_AA cells)
//   out : [L * NUM_AA] device array of ΔΔG scores, row-major (output)
__global__ void ddg_scan_kernel(int L, float* __restrict__ out);

// Host wrapper: uploads the per-residue features to constant memory, launches
// the kernel over the L × NUM_AA grid, times ONLY the kernel (CUDA events), and
// copies the scores back.
//   prot       : the loaded protein (provides L, wt_code, buried)
//   out        : resized to L*NUM_AA; filled with per-cell ΔΔG (kcal/mol)
//   kernel_ms  : out-param, GPU-measured kernel time in milliseconds
void ddg_scan_gpu(const Protein& prot, std::vector<float>& out, float* kernel_ms);
