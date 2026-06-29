// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 1.27 : MM-GBSA / MM-PBSA Rescoring
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls rescore_gpu(); kernels.cu
//   implements both the host wrapper and the device kernel. Included only by
//   .cu translation units (it declares a __global__ kernel, so the plain C++
//   compiler must never see it -- the CPU reference lives in reference_cpu.h).
//
// THE BIG IDEA -- "embarrassingly parallel over snapshots" (PATTERNS.md §1)
//   MM-GBSA rescoring evaluates the SAME energy function on EACH of S MD
//   snapshots, and the snapshots are completely INDEPENDENT (no snapshot needs
//   any other). That is the textbook GPU job: assign ONE THREAD PER SNAPSHOT.
//   Thread t computes dg[t] = snapshot_dg(receptor, ligand_snapshot_t) -- the
//   exact same shared function the CPU reference calls, so verification is
//   bit-near exact. A grid-stride loop lets a modest grid cover thousands of
//   frames. No atomics, no shared memory, no inter-thread communication: each
//   thread owns one output and writes it once.
//
//   The ensemble average (mean over snapshots) is done on the host AFTER the
//   per-snapshot energies come back -- a tiny O(S) reduction not worth a GPU
//   pass, and doing it on the host keeps the summation order identical to the
//   CPU reference (determinism, PATTERNS.md §3).
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h.
// Then read kernels.cu. The science/GPU-mapping is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // Atom, Complex, snapshot_dg (the shared HD core)

// ---- Device kernel -------------------------------------------------------
// rescore_kernel: one thread evaluates one snapshot's binding-energy estimate.
//   receptor   : [R] receptor atoms in device global memory (read by every
//                thread, never written -- a natural constant-memory candidate;
//                see kernels.cu for why we keep it in global memory here).
//   R          : receptor atom count.
//   ligand     : [S*L] ligand atoms, row-major by snapshot, in device memory.
//   L          : ligand atom count per snapshot.
//   S          : number of snapshots (= number of logical threads / outputs).
//   minus_TdS  : constant entropy term passed by value into every thread.
//   dg         : [S] output per-snapshot binding-energy estimates (kcal/mol).
// The kernel body is a grid-stride loop that calls the shared snapshot_dg().
__global__ void rescore_kernel(const Atom* __restrict__ receptor, int R,
                               const Atom* __restrict__ ligand,   int L,
                               int S, double minus_TdS,
                               double* __restrict__ dg);

// ---- Host wrapper --------------------------------------------------------
// rescore_gpu: the host-callable "do the whole GPU rescoring" function.
//   Uploads the receptor and all ligand snapshots, launches rescore_kernel,
//   times ONLY the kernel (CUDA events), copies the per-snapshot energies back.
//   main.cu calls exactly this; all CUDA bookkeeping is hidden here.
//
//   cx        : the loaded problem (receptor + S ligand snapshots + entropy).
//   dg        : host output, resized to cx.S, filled with per-snapshot dG.
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies).
void rescore_gpu(const Complex& cx, std::vector<double>& dg, float* kernel_ms);
