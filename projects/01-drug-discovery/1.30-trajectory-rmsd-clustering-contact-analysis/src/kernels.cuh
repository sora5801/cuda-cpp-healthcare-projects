// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for trajectory RMSD + contacts
// ---------------------------------------------------------------------------
// Project 1.30 : Trajectory RMSD, Clustering & Contact Analysis
//
// THE BIG IDEA
//   Analyzing F frames of a trajectory is F INDEPENDENT jobs: frame f's RMSD to
//   the reference and its native-contact fraction Q depend only on frame f and
//   the (shared) reference frame -- never on any other frame. So we give each
//   frame its OWN GPU thread, exactly the "independent jobs" pattern from the
//   1.12 fingerprint search (PATTERNS.md sec 1). The per-thread work here is a
//   small dense computation -- build a 3x3 covariance, solve a 4x4 eigenvalue by
//   QCP, sweep N^2 atom pairs -- all living in rmsd_core.h so the device thread
//   and the CPU reference run byte-identical math.
//
//   The reference frame is read by EVERY thread but never changes during the
//   launch, so it is the textbook candidate for CONSTANT memory (its broadcast
//   cache hands one address to a whole warp in a single transaction). That is
//   the teaching point this header sets up; the kernel in kernels.cu reads the
//   reference from a __constant__ buffer rather than a kernel parameter.
//
//   This header is included only by .cu units; it pulls in reference_cpu.h for
//   the Trajectory type and rmsd_core.h (transitively) for N_ATOMS.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h,
//   rmsd_core.h. Then read kernels.cu. The GPU mapping is in ../THEORY.md.
// ===========================================================================
#pragma once

#include "reference_cpu.h"   // Trajectory, FrameMetrics, N_ATOMS (pure C++)

// Device kernel: one thread per frame.
//   d_coords  : [n_frames * N_ATOMS * 3] device array of all frames (frame-major)
//   n_frames  : number of frames
//   native_total : native-contact count of the reference frame (precomputed on
//                  the host so every thread divides Q by the same integer)
//   d_rmsd    : [n_frames] output, per-frame optimal-superposition RMSD
//   d_qnc     : [n_frames] output, per-frame native-contact fraction (0..1)
// The reference frame is NOT a parameter -- it is read from the __constant__
// symbol uploaded by analyze_trajectory_gpu().
__global__ void analyze_frames_kernel(const double* __restrict__ d_coords,
                                      int n_frames, int native_total,
                                      double* __restrict__ d_rmsd,
                                      double* __restrict__ d_qnc);

// Host wrapper: uploads the reference frame to constant memory and all frames to
// global memory, launches the kernel (one thread per frame), times ONLY the
// kernel with CUDA events, and copies the per-frame metrics back.
//   traj       : the loaded trajectory (frames + which one is the reference)
//   out        : out.rmsd / out.qnc each resized to traj.n_frames and filled
//   kernel_ms  : out-param, GPU-measured kernel time in milliseconds
void analyze_trajectory_gpu(const Trajectory& traj, FrameMetrics& out, float* kernel_ms);
