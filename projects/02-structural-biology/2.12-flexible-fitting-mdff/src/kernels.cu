// ===========================================================================
// src/kernels.cu  --  MDFF GPU kernel (one thread per atom) + host wrapper
// ---------------------------------------------------------------------------
// Project 2.12 : Flexible Fitting / MDFF
//
// WHAT THIS FILE DOES
//   The GPU twin of fit_cpu(): identical per-atom physics (mdff.h), one thread
//   per atom, double-buffered Jacobi iteration over the (read-only) density map.
//   main.cu runs both fit_cpu() and fit_gpu() and compares the final atom
//   positions. See ../THEORY.md "GPU mapping".
//
// READ THIS AFTER: kernels.cuh (declarations + the thread-mapping idea), mdff.h.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, gives the scheduler 8 warps to hide the density-map memory
// latency, and leaves plenty of blocks resident for occupancy. Atom counts here
// are tiny (tens), but the mapping is written to scale to large complexes.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// mdff_step_kernel : one thread advances one atom by one fitting iteration.
//   Launch config (set in fit_gpu):
//     grid  = ceil(natoms / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Memory: reads the shared density map `rho` (global, read-only) and this
//   atom's x_old[i] / x_ref[i]; writes x_new[i]. No shared memory and no atomics
//   are needed because every thread reads only from x_old and writes a distinct
//   x_new[i] -> the atoms are independent within an iteration (Jacobi update).
//
//   The actual force + integration is mdff_step_atom() from mdff.h -- the SAME
//   function the CPU reference calls -- so the GPU and CPU compute identical math.
// ---------------------------------------------------------------------------
__global__ void mdff_step_kernel(const double* __restrict__ rho,
                                 const Vec3* __restrict__ x_old,
                                 const Vec3* __restrict__ x_ref,
                                 Vec3* __restrict__ x_new,
                                 MdffParams P) {
    // Global atom index this thread is responsible for.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard the ragged last block: natoms is rarely a multiple of the block
    // size, so the final block has threads with i >= natoms. They must do
    // nothing, or they would read/write out of bounds (an illegal address).
    if (i >= P.natoms) return;

    // One steepest-descent step: density gradient (uphill) minus restraint.
    x_new[i] = mdff_step_atom(rho, x_old[i], x_ref[i], P);
}

// ---------------------------------------------------------------------------
// fit_gpu : host wrapper -- the five canonical steps of a CUDA computation,
//   wrapped around an iteration loop.
//     (1) allocate device memory        (density + two position buffers + ref)
//     (2) copy inputs host->device      (density + x0 + x_ref, ONCE)
//     (3) launch the kernel per iter     (ping-pong x_old/x_new each step)
//     (4) copy the final result D->H
//     (5) free device memory
//   We time ONLY the kernel launches (step 3) with CUDA events so the reported
//   figure is on-device compute, not the one-time PCIe transfer (THEORY timing).
// ---------------------------------------------------------------------------
void fit_gpu(const MdffParams& P, const std::vector<double>& rho,
             const std::vector<Vec3>& x0, const std::vector<Vec3>& x_ref,
             std::vector<Vec3>& out, float* kernel_ms) {
    const int   N        = P.natoms;
    const std::size_t rho_bytes = rho.size() * sizeof(double);
    const std::size_t pos_bytes = static_cast<std::size_t>(N) * sizeof(Vec3);

    // (1) Device buffers. The d_ prefix marks DEVICE pointers (CLAUDE.md 12):
    //     dereferencing one on the host would crash. We keep TWO position
    //     buffers (d_a, d_b) so we can double-buffer the Jacobi iteration.
    double* d_rho = nullptr;
    Vec3 *d_a = nullptr, *d_b = nullptr, *d_ref = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rho, rho_bytes));   // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_a, pos_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, pos_bytes));
    CUDA_CHECK(cudaMalloc(&d_ref, pos_bytes));

    // (2) Upload the density map and the atom arrays ONCE. The density never
    //     changes during fitting, so it is copied a single time and then read by
    //     every thread on every iteration.
    CUDA_CHECK(cudaMemcpy(d_rho, rho.data(), rho_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a, x0.data(), pos_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ref, x_ref.data(), pos_bytes, cudaMemcpyHostToDevice));

    // (3) Iterate. Blocks must cover all N atoms -> ceiling division "round up".
    const int grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    Vec3* src = d_a;   // read from here this iteration
    Vec3* dst = d_b;   // write to here this iteration

    GpuTimer timer;
    timer.start();
    for (int it = 0; it < P.iters; ++it) {
        mdff_step_kernel<<<grid, THREADS_PER_BLOCK>>>(d_rho, src, d_ref, dst, P);
        // Ping-pong: this iteration's output becomes next iteration's input.
        Vec3* tmp = src; src = dst; dst = tmp;
    }
    *kernel_ms = timer.stop_ms();          // GPU-measured time over all launches
    CUDA_CHECK_LAST("mdff_step_kernel");   // catch launch + execution errors

    // After the final swap, `src` points at the buffer holding the last result.
    // (4) Bring the fitted positions back to the host output vector.
    out.assign(static_cast<std::size_t>(N), Vec3{0.0, 0.0, 0.0});
    CUDA_CHECK(cudaMemcpy(out.data(), src, pos_bytes, cudaMemcpyDeviceToHost));

    // (5) Always free what we allocated (there is no GPU garbage collector).
    CUDA_CHECK(cudaFree(d_rho));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_ref));
}
