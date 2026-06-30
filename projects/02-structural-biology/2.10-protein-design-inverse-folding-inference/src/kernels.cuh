// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for inverse-folding design
// ---------------------------------------------------------------------------
// Project 2.10 : Protein Design / Inverse Folding Inference
//
// THE BIG IDEA
//   Inverse folding here is TWO independent-per-residue passes, both of which
//   are embarrassingly parallel -- the classic "score N items independently"
//   GPU pattern (PATTERNS.md sec 1, exemplar 1.12 Tanimoto):
//
//     KERNEL 1  neighbor_kernel : one thread per residue i counts how many
//               other residues' Calpha atoms lie within CONTACT_RADIUS. This is
//               the all-pairs O(L^2) BURIAL computation -- thread i loops over
//               all L residues. It is the analog of message-passing over the
//               protein graph: every node gathers information from its spatial
//               neighbors. The backbone coordinates are read by EVERY thread and
//               never change, so we cache them in shared memory per block.
//
//     KERNEL 2  design_kernel   : one thread per residue i scores all 20 amino
//               acids with the SHARED score_aa_at_residue() (from
//               inverse_folding.h, identical to the CPU) and writes the argmax.
//               This is the per-position "decode" step at temperature 0.
//
//   Because both kernels call the exact same integer scoring core the CPU uses,
//   the GPU result is bit-for-bit identical -> we verify with EXACT equality.
//
//   This header is included only by .cu units (it declares __global__ kernels).
//   main.cu calls design_gpu(); reference_cpu.h carries the pure-C++ data model.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h,
// inverse_folding.h. Then read kernels.cu. Science/GPU-mapping in ../THEORY.md.
// ===========================================================================
#pragma once

#include "reference_cpu.h"   // Backbone, DesignResult (pure C++, safe in .cu)

// ---- Device kernels (documented in detail at their definitions) -----------

// neighbor_kernel: thread i -> residue i; writes neighbors[i] = contact count.
//   res       : [L] device array of Calpha coordinates (BackboneResidue)
//   L         : residue count
//   neighbors : [L] device output, neighbor counts
__global__ void neighbor_kernel(const BackboneResidue* __restrict__ res, int L,
                                int* __restrict__ neighbors);

// design_kernel: thread i -> residue i; writes designed[i] (argmax aa) and
//   score[i] (its score), using neighbors[i] computed by neighbor_kernel.
//   neighbors : [L] device input, per-residue burial counts
//   L         : residue count
//   designed  : [L] device output, chosen amino-acid index 0..NUM_AA-1
//   score     : [L] device output, the chosen amino acid's integer score
__global__ void design_kernel(const int* __restrict__ neighbors, int L,
                              int* __restrict__ designed, int* __restrict__ score);

// ---- Host wrapper ----------------------------------------------------------
// design_gpu: run the whole GPU pipeline (upload backbone, launch both kernels,
//   download results) and report the combined KERNEL time (CUDA events) via
//   *kernel_ms. main.cu calls exactly this; all CUDA bookkeeping is hidden here.
//
//   bb        : the loaded backbone problem (host)
//   out       : filled with neighbors/designed/score (resized to L)
//   kernel_ms : out-param, milliseconds spent in the two kernels (not copies)
void design_gpu(const Backbone& bb, DesignResult& out, float* kernel_ms);
