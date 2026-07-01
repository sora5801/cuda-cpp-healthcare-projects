// ===========================================================================
// src/kernels.cu  --  GPU per-voxel ASL Buxton fit (one thread per voxel)
// ---------------------------------------------------------------------------
// Project 4.23 : Arterial Spin Labeling & Perfusion Imaging
//
// WHAT THIS FILE DOES
//   The GPU twin of fit_cpu(): thread `v` fits voxel `v` by calling the shared
//   asl_fit_voxel() (asl.h) -- the same Gauss-Newton solve the CPU reference
//   runs -- and writes one AslFit. main.cu runs both and checks agreement.
//
//   Two teaching points live here:
//     (1) CONSTANT MEMORY for the PLD schedule: it is read by every thread but
//         is identical across the launch, so the constant cache broadcasts one
//         value to a whole warp in a single transaction (cf. project 1.12).
//     (2) The canonical five-step host wrapper (alloc / H2D / launch / D2H /
//         free), with the kernel timed by CUDA events (util/timer.cuh).
//
// READ THIS AFTER: kernels.cuh (the pattern + declarations) and asl.h (the math).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 128 is a solid default here: the fit is register- and
// compute-heavy (a few Gauss-Newton iterations, each a loop over PLDs), so we do
// not need a huge block to hide memory latency; 128 keeps register pressure low
// enough for good occupancy on sm_75..sm_89 while still giving 4 warps/block.
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// CONSTANT MEMORY: the shared PLD schedule.
//   __constant__ memory is a small (64 KB) read-only space cached by a dedicated
//   broadcast cache. Every thread reads the SAME pld[j] at step j, so a warp's 32
//   reads collapse to one -- the perfect use case. We copy the host PLDs here
//   once per launch with cudaMemcpyToSymbol.
// ---------------------------------------------------------------------------
__constant__ double c_pld[ASL_MAX_PLDS];   // [n_plds] delay schedule (s)

// ---------------------------------------------------------------------------
// asl_fit_kernel: thread v owns voxel v.
//   Launch config (set in fit_gpu):
//     grid  = ceil(n_voxels / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: v = blockIdx.x * blockDim.x + threadIdx.x.
//   Memory: reads this voxel's signal row from global memory and the PLDs from
//   constant memory; the whole Gauss-Newton state lives in registers; writes one
//   AslFit to global memory. No shared memory or atomics -- voxels are independent.
//
//   We rebuild a local AslDataset `view` that points the model at the CONSTANT-
//   memory PLDs (c_pld) instead of the global copy, so the hot inner loop reads
//   delays from the broadcast cache. Everything else (signal pointer, constants,
//   fit controls) is copied by value from the passed `meta`.
// ---------------------------------------------------------------------------
__global__ void asl_fit_kernel(AslDataset meta,
                               const double* __restrict__ d_signal,
                               AslFit* __restrict__ d_fits) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= meta.n_voxels) return;         // guard the ragged last block

    // Point the model at constant-memory PLDs and the device signal buffer.
    AslDataset view = meta;                 // value copy (counts, constants, controls)
    view.pld    = c_pld;                    // broadcast-cached delay schedule
    view.signal = d_signal;                 // device-resident measured curves

    // The identical shared solver the CPU reference calls -> matching result.
    d_fits[v] = asl_fit_voxel(view, v);
}

// ---------------------------------------------------------------------------
// fit_gpu: host wrapper. The five canonical steps of a CUDA computation:
//   (0) upload the PLD schedule to constant memory
//   (1) allocate device memory for the signal + the fit outputs
//   (2) copy the signal host->device
//   (3) launch the kernel (one thread per voxel), timed with CUDA events
//   (4) copy the AslFit results device->host
//   (5) free device memory
// We time ONLY step (3) so the reported figure is the fit cost, not the PCIe
// transfer cost (discussed separately in THEORY §"honest timing").
// ---------------------------------------------------------------------------
void fit_gpu(const HostDataset& ds, std::vector<AslFit>& fits, float* kernel_ms) {
    const int   nV = ds.n_voxels;
    const int   nP = ds.n_plds;
    const size_t sig_bytes = (size_t)nV * nP * sizeof(double);
    const size_t fit_bytes = (size_t)nV * sizeof(AslFit);

    fits.assign(nV, AslFit{});

    // Guard: constant-memory PLD buffer is fixed-size; refuse an oversized schedule.
    if (nP > ASL_MAX_PLDS) {
        std::fprintf(stderr, "[fit_gpu] n_plds=%d exceeds ASL_MAX_PLDS=%d\n", nP, ASL_MAX_PLDS);
        std::exit(EXIT_FAILURE);
    }

    // (0) Upload the delay schedule into constant memory (broadcast cache).
    CUDA_CHECK(cudaMemcpyToSymbol(c_pld, ds.pld.data(), (size_t)nP * sizeof(double)));

    // (1) Device buffers. The d_ prefix marks DEVICE pointers (CLAUDE.md §12):
    //     dereferencing one on the host would crash, so the naming matters.
    double* d_signal = nullptr;
    AslFit* d_fits   = nullptr;
    CUDA_CHECK(cudaMalloc(&d_signal, sig_bytes));   // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_fits,   fit_bytes));

    // (2) Copy the measured signal curves H2D. .data() is vector's contiguous store.
    CUDA_CHECK(cudaMemcpy(d_signal, ds.signal.data(), sig_bytes, cudaMemcpyHostToDevice));

    // A metadata view WITHOUT valid pointers: the kernel overwrites pld/signal with
    // the constant-memory and device pointers. We pass it by value (small struct).
    AslDataset meta = ds.view();
    meta.pld    = nullptr;   // replaced by c_pld inside the kernel
    meta.signal = nullptr;   // replaced by d_signal inside the kernel

    // (3) Launch. Blocks must cover all voxels -> ceiling division (round up).
    const int blocks = (nV + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    asl_fit_kernel<<<blocks, THREADS_PER_BLOCK>>>(meta, d_signal, d_fits);
    *kernel_ms = timer.stop_ms();           // GPU-measured kernel time
    CUDA_CHECK_LAST("asl_fit_kernel");      // catch launch + execution errors

    // (4) Bring the fitted physiology back to the host.
    CUDA_CHECK(cudaMemcpy(fits.data(), d_fits, fit_bytes, cudaMemcpyDeviceToHost));

    // (5) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_signal));
    CUDA_CHECK(cudaFree(d_fits));
}
