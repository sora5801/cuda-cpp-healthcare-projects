// ===========================================================================
// src/kernels.cu  --  TG-43 dose kernel: per-voxel threads + constant-memory tables
// ---------------------------------------------------------------------------
// Project 5.7 : Brachytherapy Dose & Source Modeling
//
// WHAT THIS FILE DOES
//   Implements the device kernel (dose_kernel) and the host-side glue
//   (dose_gpu) that uploads the TG-43 tables + dwell list into __constant__
//   memory, allocates the device dose buffer, launches one thread per voxel,
//   times the kernel with CUDA events, and copies the dose back. It is the GPU
//   twin of dose_cpu() in reference_cpu.cpp; main.cu runs both and compares.
//
//   The per-(voxel,dwell) physics is NOT duplicated here -- the kernel calls
//   dose_rate_one_dwell() from tg43_physics.h, the exact same inline function
//   the CPU reference uses. That shared __host__ __device__ core is what makes
//   the GPU and CPU results match to floating-point precision.
//
// READ THIS AFTER: kernels.cuh (declarations + thread-mapping idea), tg43_physics.h.
// ===========================================================================
#include "kernels.cuh"
#include "tg43_physics.h"        // SourceModel, Dwell, dose_rate_one_dwell
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, gives the scheduler 8 warps to hide the (double-precision)
// arithmetic latency, and leaves many blocks resident for occupancy.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// CONSTANT MEMORY: the TG-43 source dataset and the dwell list.
//   __constant__ is a small (64 KB total) read-only region with a dedicated
//   broadcast cache: when every thread in a warp reads the SAME address (which
//   is exactly our access pattern -- all threads share one source model and
//   iterate the same dwell list), the read is serviced in a single transaction.
//   That is far cheaper than each thread fetching the tables from global memory.
//
//   Budget check (well under 64 KB):
//     SourceModel  ~ (32 + 32 + 32 + 24 + 32*24) doubles ~ 7.4 KB
//     Dwell[64]    = 64 * 4 doubles                       = 2.0 KB
//   We copy the WHOLE SourceModel by value (it is POD with fixed arrays), so
//   there are no device pointers to chase inside the kernel.
// ---------------------------------------------------------------------------
__constant__ SourceModel c_source;                 // the source TG-43 tables
__constant__ Dwell       c_dwells[TG43_MAX_DWELLS]; // the implant dwell positions
__constant__ int         c_n_dwells;                // how many dwells are valid

// ---------------------------------------------------------------------------
// dose_kernel: one thread computes the total dose at ONE voxel.
//   Launch config (set in dose_gpu):
//     grid  = ceil(N / THREADS_PER_BLOCK) blocks,  N = nx*ny*nz voxels
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: flat voxel index i = blockIdx.x*blockDim.x + threadIdx.x,
//   decoded to (ix,iy,iz) with the SAME x-fastest layout as DoseGrid so the GPU
//   and CPU write voxel i identically.
//
//   Memory traffic: reads the source + dwells from CONSTANT memory (broadcast),
//   writes exactly one float to global memory (dose[i]). No shared memory and no
//   atomics are needed -- each voxel is owned by exactly one thread, so there is
//   never a write conflict. This is the "independent jobs + gather from constant
//   tables" pattern (PATTERNS.md section 1).
//
//   Accumulate in DOUBLE (acc): the same choice as the CPU reference, so the
//   per-voxel summation order and precision match -> exact agreement up to FMA.
// ---------------------------------------------------------------------------
__global__ void dose_kernel(int nx, int ny, int nz,
                            double ox, double oy, double oz, double spacing,
                            float* __restrict__ dose) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // flat voxel index
    const int N = nx * ny * nz;
    if (i >= N) return;                                   // guard ragged last block

    // Decode the flat index back to (ix,iy,iz). x is fastest, then y, then z --
    // must match DoseGrid's (iz*ny + iy)*nx + ix ordering in reference_cpu.
    const int ix = i % nx;
    const int iy = (i / nx) % ny;
    const int iz = i / (nx * ny);

    // World-space center of this voxel [cm].
    const double px = ox + ix * spacing;
    const double py = oy + iy * spacing;
    const double pz = oz + iz * spacing;

    // Superpose the TG-43 dose rate from every dwell position. Every thread in
    // the warp walks the same c_dwells[] entries in lockstep -> constant-cache
    // broadcast, no divergence. c_source is likewise shared by all threads.
    double acc = 0.0;
    for (int k = 0; k < c_n_dwells; ++k)
        acc += dose_rate_one_dwell(c_source, c_dwells[k], px, py, pz);

    dose[i] = static_cast<float>(acc);   // one global write per thread
}

// ---------------------------------------------------------------------------
// dose_gpu: host wrapper. Steps:
//   (1) upload the source model + dwells into constant memory (cudaMemcpyToSymbol)
//   (2) allocate the device dose buffer
//   (3) launch dose_kernel (one thread per voxel), timed with CUDA events
//   (4) copy the dose back to the host
//   (5) free device memory
// Only step (3) is timed, so the reported figure is the kernel cost, not the
// (tiny) constant-memory upload or the result copy (discussed in THEORY).
// ---------------------------------------------------------------------------
void dose_gpu(const Plan& plan, std::vector<float>& dose, float* kernel_ms) {
    const DoseGrid& g = plan.grid;
    const int N = g.size();
    dose.assign(static_cast<std::size_t>(N), 0.0f);

    // (1) Upload the read-only tables. cudaMemcpyToSymbol copies host bytes into
    //     the named __constant__ symbol. We copy the SourceModel by value and
    //     the dwell array (only the valid prefix) plus its count.
    const int n_dwells = static_cast<int>(plan.dwells.size());
    CUDA_CHECK(cudaMemcpyToSymbol(c_source, &plan.source, sizeof(SourceModel)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_dwells, plan.dwells.data(),
                                  sizeof(Dwell) * static_cast<std::size_t>(n_dwells)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_n_dwells, &n_dwells, sizeof(int)));

    // (2) Device dose buffer (one float per voxel).
    float* d_dose = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(N) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_dose, bytes));

    // (3) Launch. Blocks cover all N voxels via ceiling division.
    const int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    dose_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        g.nx, g.ny, g.nz, g.ox, g.oy, g.oz, g.spacing, d_dose);
    *kernel_ms = timer.stop_ms();       // GPU-measured kernel time
    CUDA_CHECK_LAST("dose_kernel");     // catch launch + execution errors

    // (4) Bring the dose back to the host vector.
    CUDA_CHECK(cudaMemcpy(dose.data(), d_dose, bytes, cudaMemcpyDeviceToHost));

    // (5) Free (there is no GPU garbage collector).
    CUDA_CHECK(cudaFree(d_dose));
}
