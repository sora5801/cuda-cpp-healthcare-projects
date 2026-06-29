// ===========================================================================
// src/kernels.cu  --  Multi-walker metadynamics kernel (one thread per walker)
// ---------------------------------------------------------------------------
// Project 1.6 : Enhanced Sampling -- Metadynamics & Replica Exchange
//
// WHAT THIS FILE DOES
//   The GPU twin of integrate_cpu(): each thread runs the SAME metad::run_walker()
//   loop (defined in metad.h) for one walker, into its own slice of a big device
//   bias buffer. The host wrapper allocates that buffer, launches the kernel,
//   copies results back, and forms the ensemble-average bias grid. main.cu then
//   compares the per-walker GPU results against the CPU reference (machine
//   precision, since both sides run byte-identical double-precision math).
//
//   Comment density is high here on purpose (CLAUDE.md §6.2): kernels are where
//   the GPU reasoning lives.
//
// READ THIS AFTER: kernels.cuh (declarations + the ensemble idea), metad.h.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 128 is a solid default for register-heavy, long-running
// per-thread loops on sm_75..sm_89: it is a multiple of the 32-lane warp and
// keeps occupancy reasonable even when each thread holds many doubles in
// registers (the Langevin state + loop counters). Tune per GPU.
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// metad_kernel: thread `id` owns walker `id`.
//   Launch config (set in integrate_gpu):
//     grid  = ceil(n_walkers / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: id = blockIdx.x * blockDim.x + threadIdx.x  -> walker id.
//
//   Memory: the thread's bias grid is the global slice d_bias[id*nbins ...]. The
//   walker reads/writes it through metad.h helpers. There is NO inter-thread
//   communication and NO atomics: each walker is fully independent (this is the
//   teaching version; production multi-walker MetaD shares one hill list and DOES
//   need synchronization -- see THEORY.md "GPU mapping" and the Exercises).
//
//   Divergence is mild: all walkers run the same number of steps; only the
//   crossing-count and deposit branches differ, and the deposit branch is taken
//   in lock-step (every deposit_every steps) by every thread.
// ---------------------------------------------------------------------------
__global__ void metad_kernel(MetadConfig c, double* __restrict__ d_bias,
                             metad::WalkerResult* __restrict__ d_out) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ensemble_size(c)) return;          // guard the ragged last block

    // This walker's private bias grid: a contiguous slice of the big buffer.
    // (size_t cast avoids 32-bit overflow when n_walkers*nbins is large.)
    double* my_bias = d_bias + static_cast<std::size_t>(id) * c.model.nbins;

    // Run the entire trajectory (Langevin + tempered hill deposition). The exact
    // same call the CPU reference makes -> identical results. walker_start()
    // alternates the start well by parity so the ensemble seeds both basins.
    d_out[id] = metad::run_walker(c.model, my_bias, c.seed, id, walker_start(c, id));
}

// ---------------------------------------------------------------------------
// integrate_gpu: host wrapper. The canonical CUDA steps, plus a reduction.
//   (1) allocate the big per-walker bias buffer + the results buffer
//   (2) launch one thread per walker (timed with CUDA events -> kernel-only ms)
//   (3) copy the per-walker summaries AND the full bias buffer back
//   (4) average the per-walker bias grids on the host -> ensemble-mean bias
//   (5) free device memory
//   We average on the host (step 4) for clarity; an atomic/segmented GPU
//   reduction would work too but is unnecessary at this teaching scale and would
//   reintroduce the float-summation-order caveat (PATTERNS.md §3). The per-bin
//   sum here is over a FIXED walker order, so it is deterministic.
// ---------------------------------------------------------------------------
void integrate_gpu(const MetadConfig& c,
                  std::vector<metad::WalkerResult>& results,
                  std::vector<double>& mean_bias,
                  float* kernel_ms) {
    const int M  = ensemble_size(c);
    const int nb = c.model.nbins;
    results.assign(static_cast<std::size_t>(M), metad::WalkerResult{});
    mean_bias.assign(static_cast<std::size_t>(nb), 0.0);

    // (1) Device buffers. d_bias holds M independent grids back to back.
    const std::size_t bias_count = static_cast<std::size_t>(M) * nb;
    double* d_bias = nullptr;
    metad::WalkerResult* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bias, bias_count * sizeof(double)));         // OOM-able
    CUDA_CHECK(cudaMalloc(&d_out,  static_cast<std::size_t>(M) * sizeof(metad::WalkerResult)));

    // (2) Launch one thread per walker; time only the kernel (not the copies).
    const int blocks = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    metad_kernel<<<blocks, THREADS_PER_BLOCK>>>(c, d_bias, d_out);
    *kernel_ms = timer.stop_ms();           // GPU-measured kernel time
    CUDA_CHECK_LAST("metad_kernel");        // catch launch + execution errors

    // (3) Copy the summaries and the full bias buffer back to the host.
    CUDA_CHECK(cudaMemcpy(results.data(), d_out,
                          static_cast<std::size_t>(M) * sizeof(metad::WalkerResult),
                          cudaMemcpyDeviceToHost));
    std::vector<double> all_bias(bias_count, 0.0);
    CUDA_CHECK(cudaMemcpy(all_bias.data(), d_bias,
                          bias_count * sizeof(double), cudaMemcpyDeviceToHost));

    // (4) Ensemble-average the per-walker bias grids (same fixed order as the
    //     CPU reference -> identical floating-point sum -> exact agreement).
    for (int id = 0; id < M; ++id) {
        const double* g = all_bias.data() + static_cast<std::size_t>(id) * nb;
        for (int j = 0; j < nb; ++j) mean_bias[static_cast<std::size_t>(j)] += g[j];
    }
    const double inv_M = 1.0 / static_cast<double>(M);
    for (int j = 0; j < nb; ++j) mean_bias[static_cast<std::size_t>(j)] *= inv_M;

    // (5) Free device memory (no GPU garbage collector).
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_out));
}
