// ===========================================================================
// src/kernels.cuh  --  GPU Debye-scattering interface (declarations + the idea)
// ---------------------------------------------------------------------------
// Project 2.24 : SAXS / SANS Data-Driven Structure Modeling
//
// THE BIG IDEA (pattern: INDEPENDENT JOBS + per-thread O(N^2) reduction)
//   The forward SAXS curve is I(q) for many q values, and each q is computed by
//   an independent double sum over all atom pairs (the Debye formula in
//   saxs_core.h). So we map ONE GPU THREAD PER q VALUE: thread k computes
//   I(q[k]) by looping over all n_atoms^2 pairs in its registers, then writes a
//   single output. There is no cross-thread communication and no atomics -- the
//   whole reduction for a given q lives inside one thread -- which makes the
//   result deterministic and trivially correct to verify against the CPU.
//
//   The atom arrays (x,y,z,f) are read by EVERY thread but never modified during
//   the launch. They are large (one per atom) so they live in global memory; we
//   stage them into shared memory in tiles so the inner loop reads fast on-chip
//   memory instead of hammering global memory n_q times (see kernels.cu). The
//   query-in-constant-memory trick used by 1.12/12.01 does not fit here because
//   the atom set is the *shared* operand, not a small per-thread query.
//
//   This is the GPU twin of debye_profile_cpu(): same math (shared saxs_core.h),
//   same outputs, just every q in parallel.
//
// READ THIS AFTER: saxs_core.h, util/cuda_check.cuh, util/timer.cuh,
//   reference_cpu.h.  Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // SaxsModel (pure C++, safe to include from a .cu)

// ---------------------------------------------------------------------------
// debye_gpu: host wrapper that runs the whole forward-model on the GPU.
//   Allocates device buffers for the atom arrays and the q grid, copies them up,
//   launches the kernel (one thread per q), copies the n_q intensities back, and
//   reports the measured KERNEL time (CUDA events) via *kernel_ms.
//
//     m         : the loaded model (atoms + q grid); only its arrays are read
//     I_model   : output, resized to m.n_q -- the forward-modeled intensities
//     kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
//
//   main.cu calls exactly this; all CUDA bookkeeping is hidden inside kernels.cu.
//   The result must match debye_profile_cpu(m, ...) within the documented
//   tolerance (main.cu).
// ---------------------------------------------------------------------------
void debye_gpu(const SaxsModel& m, std::vector<double>& I_model, float* kernel_ms);
