// ===========================================================================
// src/kernels.cu  --  GPU track-structure kernel (per-thread tracks + atomics)
// ---------------------------------------------------------------------------
// Project 5.11 : Microdosimetry & Track-Structure Simulation
//
// GPU twin of track_cpu(): identical tracks (shared ts_physics.h), but run in
// parallel and scored with atomicAdd into shared integer tallies. main.cu runs
// both and asserts every tally matches EXACTLY. See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "ts_physics.h"          // TrackParams, Rng, simulate_track (host+device)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, gives the scheduler 8 warps per block to hide latency, and leaves
// plenty of blocks resident for occupancy. (Track simulation is compute-bound and
// branchy; 256 keeps register pressure and occupancy well balanced here.)
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// track_kernel: grid-stride over primary tracks. Each iteration simulates ONE
// primary with its own reproducible RNG stream and atomically adds its integer
// results into the shared tallies.
//   * No shared memory for the tallies: they are tiny (three scalars + a small
//     histogram) but written by every thread, so they live in global memory and
//     are updated with atomicAdd.
//   * Integer quanta / counts => the atomics COMMUTE => deterministic, exactly
//     CPU-matching sums (PATTERNS.md §3). A float energy tally would drift.
//   * WARP DIVERGENCE is the classic track-structure challenge: different
//     primaries take different numbers of steps and branches, so lanes in a warp
//     finish at different times. Production codes (Geant4-DNA/MPEXS-DNA) sort
//     tracks by interaction type before each step to keep a warp coherent; we
//     keep the readable one-thread-per-track version and explain the fix in
//     THEORY.md rather than obscuring the teaching kernel with it.
// ---------------------------------------------------------------------------
__global__ void track_kernel(TrackParams tp, unsigned long long n_tracks,
                             unsigned long long seed,
                             unsigned long long* __restrict__ d_quanta,
                             unsigned long long* __restrict__ d_ssb,
                             unsigned long long* __restrict__ d_dsb,
                             unsigned long long* __restrict__ d_yhist) {
    // Grid-stride bounds: `start` is this thread's first track, `stride` is the
    // total number of threads. The loop lets a fixed-size grid cover any number
    // of tracks -- the standard idiom for "more work items than threads".
    const unsigned long long stride =
        static_cast<unsigned long long>(blockDim.x) * gridDim.x;
    const unsigned long long start =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;

    for (unsigned long long i = start; i < n_tracks; i += stride) {
        Rng rng = rng_seed(seed, i);                 // this track's private stream
        TrackResult r = simulate_track(tp, rng);     // shared physics (ts_physics.h)

        // Fold the integer results into the shared tallies. Many threads hit the
        // same addresses => atomicAdd. Integer adds are order-independent, so the
        // final sums are deterministic and identical to the serial CPU tally.
        atomicAdd(d_quanta, r.energy_quanta);
        if (r.ssb) atomicAdd(d_ssb, static_cast<unsigned long long>(r.ssb));
        if (r.dsb) atomicAdd(d_dsb, static_cast<unsigned long long>(r.dsb));
        atomicAdd(&d_yhist[r.y_bin], 1ULL);          // one increment into y-spectrum
    }
}

// ---------------------------------------------------------------------------
// track_gpu: host wrapper. Allocate + zero the device tallies, launch all
// tracks, copy the results back, free. We time ONLY the kernel with CUDA events
// so the reported figure is the compute cost (the tally copy-back is a few tens
// of bytes, negligible; there are no large H2D transfers because the "input" is
// just parameters passed by value).
// ---------------------------------------------------------------------------
void track_gpu(const TrackProblem& prob, TrackTally& tally, float* kernel_ms) {
    const int n_bins = prob.tp.n_y_bins;

    // Device accumulators. Three scalar counters + the y-histogram, all u64 so
    // the sums cannot overflow across many tracks and so integer atomics apply.
    unsigned long long *d_quanta = nullptr, *d_ssb = nullptr, *d_dsb = nullptr;
    unsigned long long *d_yhist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_quanta, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_ssb,    sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_dsb,    sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_yhist,  n_bins * sizeof(unsigned long long)));

    // Zero everything before scoring (atomicAdd accumulates onto these).
    CUDA_CHECK(cudaMemset(d_quanta, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_ssb,    0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_dsb,    0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_yhist,  0, n_bins * sizeof(unsigned long long)));

    // A fixed, generous grid: enough resident warps to hide the branchy per-track
    // work; the grid-stride loop then covers any number of tracks.
    const int blocks = 1024;
    GpuTimer timer;
    timer.start();
    track_kernel<<<blocks, THREADS_PER_BLOCK>>>(prob.tp, prob.n_tracks, prob.seed,
                                                d_quanta, d_ssb, d_dsb, d_yhist);
    *kernel_ms = timer.stop_ms();          // GPU-measured kernel time
    CUDA_CHECK_LAST("track_kernel");       // catch launch + execution errors

    // Copy the small tallies back into the host TrackTally.
    tally.y_hist.assign(static_cast<std::size_t>(n_bins), 0ULL);
    CUDA_CHECK(cudaMemcpy(&tally.total_quanta, d_quanta, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&tally.total_ssb, d_ssb, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&tally.total_dsb, d_dsb, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(tally.y_hist.data(), d_yhist,
                          n_bins * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));

    // Free device memory (no GPU garbage collector).
    CUDA_CHECK(cudaFree(d_quanta));
    CUDA_CHECK(cudaFree(d_ssb));
    CUDA_CHECK(cudaFree(d_dsb));
    CUDA_CHECK(cudaFree(d_yhist));
}
