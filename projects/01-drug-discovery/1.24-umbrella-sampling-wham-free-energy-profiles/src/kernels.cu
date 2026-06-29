// ===========================================================================
// src/kernels.cu  --  Umbrella-window kernel (one thread per window) + wrapper
// ---------------------------------------------------------------------------
// Project 1.24 : Umbrella Sampling / WHAM Free Energy Profiles
//
// WHAT THIS FILE DOES
//   Implements window_kernel (thread k runs window k's full biased Langevin
//   trajectory and fills its histogram slice) and sample_windows_gpu (the host
//   glue that allocates+zeroes the device histogram, launches the kernel, times
//   it, and copies the counts back). This is the GPU twin of sample_windows_cpu()
//   in reference_cpu.cpp; main.cu runs both and asserts the histograms are equal.
//
//   The actual per-step math lives in umbrella.h (shared host+device), so the
//   kernel and the CPU loop run byte-identical simulations -> identical integer
//   histograms (CLAUDE.md section 6.2 density target; PATTERNS.md sections 2-3).
//
// READ THIS AFTER: kernels.cuh (the thread->window idea), umbrella.h (the physics).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 128 is a good default for an ensemble kernel: each thread is
// register-heavy (it carries a full trajectory's working state), so a smaller
// block keeps register pressure manageable while still giving the scheduler 4
// warps per block to hide latency. The grid is sized to cover all windows.
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// window_kernel: one thread owns one umbrella window.
//   Launch config (set in sample_windows_gpu):
//     grid  = ceil(n_windows / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: k = blockIdx.x * blockDim.x + threadIdx.x  is the window.
//   Memory: the thread reads the small POD config from registers (passed by
//   value), runs the trajectory entirely in registers/local memory, and writes
//   ONLY into hist[k*nbins .. k*nbins+nbins-1] -- a slice no other thread touches,
//   so there is no contention and no atomics are required (contrast 5.01's shared
//   tally, which must use atomicAdd). Counts are integers -> deterministic.
//
//   Divergence note: every window runs the SAME number of steps, so warps stay in
//   lock-step through the time loop; the only per-thread branch is the in-grid
//   test inside simulate_window(). Mild, healthy divergence.
// ---------------------------------------------------------------------------
__global__ void window_kernel(UmbrellaConfig c, unsigned int* __restrict__ hist) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's window
    if (k >= c.n_windows) return;                         // guard the ragged last block

    // This window's private output slice.
    unsigned int* h = hist + static_cast<std::size_t>(k) * c.grid.nbins;

    // Run the full equilibration + sampling trajectory and fill h. Identical call
    // to the one the CPU reference makes -> identical counts.
    simulate_window(c.pot, c.grid, window_spec(c, k),
                    c.D, c.dt, c.n_equil, c.n_sample,
                    c.seed, k, h);
}

// ---------------------------------------------------------------------------
// sample_windows_gpu: host wrapper. The canonical steps of a CUDA computation,
// minus input H2D copies (the only "input" is the small config, passed by value):
//   (1) allocate + ZERO the device histogram   (2) launch one thread per window
//   (3) copy the counts device->host           (4) free device memory
// We time ONLY the kernel with CUDA events so the figure is compute, not transfer.
// ---------------------------------------------------------------------------
void sample_windows_gpu(const UmbrellaConfig& c,
                        std::vector<unsigned int>& hist_out,
                        float* kernel_ms) {
    const int total = total_hist_size(c);          // n_windows * nbins counts
    hist_out.assign(static_cast<std::size_t>(total), 0u);
    const std::size_t bytes = static_cast<std::size_t>(total) * sizeof(unsigned int);

    // (1) Device histogram. The d_ prefix marks a DEVICE pointer (CLAUDE.md 12).
    //     We must zero it: the kernel only INCREMENTS counts, it does not clear.
    unsigned int* d_hist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hist, bytes));         // can fail: out of device memory
    CUDA_CHECK(cudaMemset(d_hist, 0, bytes));       // all counts start at 0

    // (2) Launch: enough blocks to give every window a thread (ceiling division).
    const int blocks = (c.n_windows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    window_kernel<<<blocks, THREADS_PER_BLOCK>>>(c, d_hist);
    *kernel_ms = timer.stop_ms();                   // GPU-measured kernel time
    CUDA_CHECK_LAST("window_kernel");               // catch launch + execution errors

    // (3) Bring the histograms back to the host.
    CUDA_CHECK(cudaMemcpy(hist_out.data(), d_hist, bytes, cudaMemcpyDeviceToHost));

    // (4) Free device memory (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_hist));
}
