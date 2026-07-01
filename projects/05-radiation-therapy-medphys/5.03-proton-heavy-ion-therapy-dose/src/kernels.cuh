// ===========================================================================
// src/kernels.cuh  --  GPU pencil-beam dose interface (declaration + big idea)
// ---------------------------------------------------------------------------
// Project 5.3 : Proton & Heavy-Ion Therapy Dose
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls dose_gpu(); kernels.cu
//   implements the host wrapper and the device kernel. Included only by .cu
//   translation units (it declares a __global__ kernel, so the plain host C++
//   compiler must never see it -- that is why the CPU reference lives in a
//   separate pure-C++ header, reference_cpu.h).
//
// THE BIG IDEA -- "one thread per voxel, gather over spots"
//   The dose at a voxel is an INDEPENDENT sum over all spots:
//       dose(voxel) = sum_over_spots  dose_from_spot(beam, spot, voxel)
//   Different voxels never write the same memory, so there is no contention and
//   NO atomics are needed: we give each voxel its own thread, and that thread
//   loops over every spot, accumulating into a private register, then writes its
//   voxel once. This is the classic GATHER pattern (each output pulls from many
//   inputs), the same shape as CT backprojection (4.01). Because each thread
//   reuses the whole spot list, we stage the spots in CONSTANT memory: it is
//   read by every thread but never changes during the launch, so its broadcast
//   cache is ideal (docs/PATTERNS.md §1, catalog CUDA pattern for 5.3).
//
//   Thread -> voxel mapping: we flatten the 3-D grid to a 1-D index
//       idx = (k*ny + j)*nx + i        (x fastest -> coalesced writes)
//   and launch enough 1-D blocks to cover nx*ny*nz voxels; a grid-stride loop
//   lets a fixed grid cover any voxel count. See kernels.cu for the details.
//
//   WHY NOT atomics like the Monte-Carlo flagship (5.01)? MC SCATTERS random
//   deposits into shared bins (collisions -> atomics). This analytic engine
//   GATHERS a deterministic sum per voxel (no collisions -> plain register add).
//   Contrasting the two is a core teaching point of the radiation-physics domain.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, proton_physics.h,
//                  reference_cpu.h. Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // Plan, Grid, BeamModel, Spot (pure C++, safe in a .cu)

// ---- Host wrapper --------------------------------------------------------
// dose_gpu: run the WHOLE GPU dose computation for a plan.
//   Copies the spot list to constant memory, launches the per-voxel kernel,
//   copies the dose volume back, and reports the measured KERNEL time (CUDA
//   events) via *kernel_ms. main.cu calls exactly this; all CUDA bookkeeping is
//   hidden here so the entry point stays about the science, not the plumbing.
//
//   plan      : the treatment plan (grid + beam + spots); read-only
//   dose      : host output, resized to nx*ny*nz voxels (output parameter)
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
void dose_gpu(const Plan& plan, std::vector<float>& dose, float* kernel_ms);
