// ===========================================================================
// src/kernels.cu  --  Per-frame trajectory analysis kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 1.30 : Trajectory RMSD, Clustering & Contact Analysis
//
// WHAT THIS FILE DOES
//   Implements analyze_frames_kernel (one thread per frame) and the host glue
//   analyze_trajectory_gpu (upload reference -> constant memory, upload frames
//   -> global memory, launch, time, copy back). It is the GPU twin of
//   analyze_trajectory_cpu() in reference_cpu.cpp; both call the SAME per-frame
//   math from rmsd_core.h, so main.cu's CPU-vs-GPU check is exact to ~1e-9.
//
//   Teaching points (see ../THEORY.md "GPU mapping"):
//     * one thread per frame -- independent jobs, the 1.12 pattern;
//     * the shared reference frame lives in __constant__ memory (broadcast);
//     * a grid-stride loop lets a modest grid cover an arbitrarily long traj;
//     * double precision (FP64) throughout, so CPU and GPU agree to ~eps.
//
// READ THIS AFTER: kernels.cuh (the interface + the constant-memory idea) and
//   rmsd_core.h (the math every thread runs).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// ---------------------------------------------------------------------------
// The REFERENCE frame in CONSTANT memory.
//   * Every thread reads all N_ATOMS*3 reference coordinates but NONE writes
//     them, and they are identical for the whole launch -> constant memory is
//     the ideal home: its hardware cache broadcasts one address to an entire
//     warp in a single transaction, instead of a global load per thread.
//   * Size is fixed at compile time (N_ATOMS*3*8 = 384 bytes for N_ATOMS=16),
//     trivially within the 64 KB constant bank. Filled by cudaMemcpyToSymbol()
//     in analyze_trajectory_gpu().
//   * rmsd_core.h's kabsch_rmsd/frac_native_contacts take a `const double*` to
//     the reference, so we simply pass &c_ref[0] -- the same code runs on the
//     CPU (where the pointer is into a std::vector) and the GPU (constant mem).
// ---------------------------------------------------------------------------
__constant__ double c_ref[N_ATOMS * 3];

// 128 threads/block: a multiple of the 32-lane warp with good occupancy on
// sm_75..sm_89. Each thread does a fair amount of FP64 work (a 4x4 eigenvalue
// plus an N^2 contact sweep), so we keep the block modest; the grid-stride loop
// below handles any number of frames. (See THEORY "GPU mapping" for the
// occupancy reasoning.)
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// analyze_frames_kernel: one logical thread per frame, via a grid-stride loop so
// a fixed-size grid still covers an arbitrarily long trajectory.
//   Thread (blockIdx.x, threadIdx.x) starts at f = block*blockDim + thread and
//   strides by the total thread count until f >= n_frames.
//   Memory: c_ref from the constant cache (broadcast); frame f's coordinates
//   from global memory; two scalar writes (d_rmsd[f], d_qnc[f]). No shared
//   memory or atomics -- outputs are fully independent.
// ---------------------------------------------------------------------------
__global__ void analyze_frames_kernel(const double* __restrict__ d_coords,
                                      int n_frames, int native_total,
                                      double* __restrict__ d_rmsd,
                                      double* __restrict__ d_qnc) {
    const int stride = blockDim.x * gridDim.x;             // total threads in grid
    for (int f = blockIdx.x * blockDim.x + threadIdx.x; f < n_frames; f += stride) {
        // Pointer to this frame's first coordinate (frame-major layout). The
        // same frame_ptr() helper is used by the CPU reference.
        const double* fr = frame_ptr(d_coords, f);
        // Both calls are the shared __host__ __device__ math from rmsd_core.h --
        // IDENTICAL arithmetic to the CPU path. The reference comes from constant
        // memory (c_ref); kabsch_rmsd/frac_native_contacts don't care where the
        // pointer points, so no special device variant is needed.
        d_rmsd[f] = kabsch_rmsd(fr, c_ref);
        d_qnc[f]  = frac_native_contacts(fr, c_ref, native_total);
    }
}

// ---------------------------------------------------------------------------
// analyze_trajectory_gpu: the canonical CUDA steps, with the reference frame
// going to constant memory instead of a global buffer. We time ONLY the kernel
// (CUDA events), not the H2D/D2H copies (those are discussed in THEORY).
//
//   IMPORTANT (determinism): native_total is computed ONCE on the host from the
//   reference frame and passed in, so every thread divides Q by the identical
//   integer -- matching the CPU reference exactly. We deliberately do not
//   recompute it per thread (wasteful and a chance for divergence).
// ---------------------------------------------------------------------------
void analyze_trajectory_gpu(const Trajectory& traj, FrameMetrics& out, float* kernel_ms) {
    const int n = traj.n_frames;
    out.rmsd.assign(static_cast<std::size_t>(n), 0.0);
    out.qnc.assign(static_cast<std::size_t>(n), 0.0);

    const std::size_t coords_bytes = traj.coords.size() * sizeof(double);
    const std::size_t out_bytes    = static_cast<std::size_t>(n) * sizeof(double);

    // (a) Upload the reference frame to the __constant__ symbol. This is a
    //     special copy that targets the constant bank rather than ordinary
    //     global memory. The host computes native_total from the same data.
    const double* h_ref = frame_ptr(traj.coords.data(), traj.ref);
    CUDA_CHECK(cudaMemcpyToSymbol(c_ref, h_ref, N_ATOMS * 3 * sizeof(double)));
    const int native_total = count_native_contacts(h_ref);  // once, on the host

    // (b) Allocate + upload all frames, and allocate the two output arrays.
    double* d_coords = nullptr;   // [n*N_ATOMS*3] device, frame-major
    double* d_rmsd   = nullptr;   // [n] device output
    double* d_qnc    = nullptr;   // [n] device output
    CUDA_CHECK(cudaMalloc(&d_coords, coords_bytes));
    CUDA_CHECK(cudaMalloc(&d_rmsd, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_qnc, out_bytes));
    CUDA_CHECK(cudaMemcpy(d_coords, traj.coords.data(), coords_bytes, cudaMemcpyHostToDevice));

    // (c) Launch. Enough blocks to cover n frames one-thread-each, but capped so
    //     the grid stays modest; the grid-stride loop covers any larger n.
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 1024) blocks = 1024;   // cap: grid-stride handles the remainder
    GpuTimer timer;
    timer.start();
    analyze_frames_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_coords, n, native_total,
                                                         d_rmsd, d_qnc);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("analyze_frames_kernel");

    // (d) Copy the per-frame metrics back, then (e) free device memory.
    CUDA_CHECK(cudaMemcpy(out.rmsd.data(), d_rmsd, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.qnc.data(),  d_qnc,  out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_coords));
    CUDA_CHECK(cudaFree(d_rmsd));
    CUDA_CHECK(cudaFree(d_qnc));
}
