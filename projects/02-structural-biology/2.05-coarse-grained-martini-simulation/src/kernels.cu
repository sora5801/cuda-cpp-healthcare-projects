// ===========================================================================
// src/kernels.cu  --  GPU velocity-Verlet MD kernels + host time-loop wrapper
// ---------------------------------------------------------------------------
// Project 2.5 : Coarse-Grained / MARTINI Simulation
//
// GPU twin of simulate_cpu(). Identical per-bead physics (martini.h), ONE
// THREAD PER BEAD, two kernels per step around the force recompute. main.cu
// compares the final positions/velocities. See ../THEORY.md "GPU mapping".
//
// WHY THREE KERNELS AND NOT ONE
//   A velocity-Verlet step has a data dependency: every bead must finish its
//   drift (move to the new position) BEFORE any force is recomputed, because a
//   force reads ALL beads' positions. A single kernel cannot synchronise across
//   the whole grid mid-flight, but a KERNEL BOUNDARY is a global barrier. So we
//   split the step into separate launches: kick_drift -> force -> kick. The
//   launches are cheap relative to the O(N^2) force kernel that dominates.
// ===========================================================================
#include "kernels.cuh"
#include "martini.h"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, 8 warps to hide the long latency of the global-memory loads in
// the inner pair loop, and enough resident blocks for good occupancy.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// force_kernel: force[i] = total Lennard-Jones force on bead i.
//   Launch: grid = ceil(n / 256), block = 256.
//   Thread (blockIdx.x, threadIdx.x) -> bead i = blockIdx.x*blockDim.x + tid.
//   Memory: reads ALL of pos[] and type[] from global memory (the O(N) inner
//   loop per thread => O(N^2) global reads total); writes one force[i]. No
//   shared memory or atomics in this teaching version -- each thread owns a
//   distinct output, so there are no races. (THEORY section 4 sketches the
//   shared-memory "tiled" optimisation that production codes use.)
// ---------------------------------------------------------------------------
__global__ void force_kernel(MdParams P, const Vec3* __restrict__ pos,
                             const int* __restrict__ type,
                             Vec3* __restrict__ force) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P.n) return;                          // guard the ragged last block
    // compute_force_on() lives in martini.h and is the SAME function the CPU
    // calls -- summing over j in identical index order -> identical result.
    force[i] = compute_force_on(i, pos, type, P);
}

// ---------------------------------------------------------------------------
// kick_drift_kernel: first half-kick + drift for bead i (uses the OLD force).
//   v[i] += 0.5*(f/m)*dt ; x[i] += v[i]*dt ; wrap x[i] into the box.
//   One thread per bead; positions/velocities are independent here.
// ---------------------------------------------------------------------------
__global__ void kick_drift_kernel(MdParams P, Vec3* __restrict__ pos,
                                  Vec3* __restrict__ vel,
                                  const Vec3* __restrict__ force) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P.n) return;
    verlet_kick_drift(pos[i], vel[i], force[i], P);   // shared physics (martini.h)
}

// ---------------------------------------------------------------------------
// kick_kernel: second half-kick for bead i (uses the freshly recomputed force).
//   v[i] += 0.5*(f_new/m)*dt. Splitting the kicks around the drift is what makes
//   velocity-Verlet symplectic (energy-stable) -- see martini.h.
// ---------------------------------------------------------------------------
__global__ void kick_kernel(MdParams P, Vec3* __restrict__ vel,
                            const Vec3* __restrict__ force) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P.n) return;
    verlet_kick(vel[i], force[i], P);                 // shared physics (martini.h)
}

// ---------------------------------------------------------------------------
// simulate_gpu: run the entire velocity-Verlet loop on the GPU.
//   The five canonical CUDA steps, with the time loop living in step (3):
//     (1) allocate device buffers for pos, vel, type, force
//     (2) copy inputs host -> device
//     (3) loop: initial force, then per step {kick_drift, force, kick}
//     (4) copy final pos, vel device -> host
//     (5) free device memory
//   Only the kernel loop is timed (CUDA events) so the reported figure is the
//   compute cost, not the one-time PCIe transfer (discussed in THEORY).
// ---------------------------------------------------------------------------
void simulate_gpu(System& sys, float* kernel_ms) {
    const MdParams& P = sys.P;
    const int n = P.n;
    const std::size_t vbytes = static_cast<std::size_t>(n) * sizeof(Vec3);
    const std::size_t ibytes = static_cast<std::size_t>(n) * sizeof(int);

    // (1) Device buffers. The d_ prefix marks DEVICE pointers (CLAUDE.md 12):
    //     dereferencing one on the host would crash, so the naming matters.
    Vec3 *d_pos = nullptr, *d_vel = nullptr, *d_force = nullptr;
    int*  d_type = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pos,   vbytes));      // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_vel,   vbytes));
    CUDA_CHECK(cudaMalloc(&d_force, vbytes));
    CUDA_CHECK(cudaMalloc(&d_type,  ibytes));

    // (2) Copy the initial configuration H2D. .data() is the vector's backing array.
    CUDA_CHECK(cudaMemcpy(d_pos,  sys.pos.data(),  vbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel,  sys.vel.data(),  vbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_type, sys.type.data(), ibytes, cudaMemcpyHostToDevice));

    // (3) The time loop. Blocks must cover all n beads: ceil(n / B).
    const int grid = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    // Initial force at the starting positions (needed before the first kick).
    force_kernel<<<grid, THREADS_PER_BLOCK>>>(P, d_pos, d_type, d_force);
    for (int step = 0; step < P.steps; ++step) {
        // First half-kick + drift with the current force.
        kick_drift_kernel<<<grid, THREADS_PER_BLOCK>>>(P, d_pos, d_vel, d_force);
        // Recompute all forces at the new positions (the kernel boundary above
        // is the global barrier that guarantees every drift finished first).
        force_kernel<<<grid, THREADS_PER_BLOCK>>>(P, d_pos, d_type, d_force);
        // Second half-kick with the new force -> completes the Verlet step.
        kick_kernel<<<grid, THREADS_PER_BLOCK>>>(P, d_vel, d_force);
    }
    *kernel_ms = timer.stop_ms();                  // GPU-measured loop time
    CUDA_CHECK_LAST("md kernels");                 // catch launch + execution errors

    // (4) Bring the final state back to the host.
    CUDA_CHECK(cudaMemcpy(sys.pos.data(), d_pos, vbytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sys.vel.data(), d_vel, vbytes, cudaMemcpyDeviceToHost));

    // (5) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_vel));
    CUDA_CHECK(cudaFree(d_force));
    CUDA_CHECK(cudaFree(d_type));
}
