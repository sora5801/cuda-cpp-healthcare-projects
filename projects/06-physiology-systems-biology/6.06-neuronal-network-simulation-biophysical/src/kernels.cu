// ===========================================================================
// src/kernels.cu  --  GPU network kernel (one thread per neuron, per-step launch)
// ---------------------------------------------------------------------------
// Project 6.6 : Neuronal Network Simulation (Biophysical)
//
// GPU twin of integrate_cpu(): each thread owns one neuron and advances it by
// ONE timestep per launch, reading its presynaptic partner's spike from the
// previous-step buffer and writing its own spike into the current buffer. The
// host loop launches this kernel `steps` times, swapping the two spike buffers
// between launches (ping-pong). Because it calls the SAME neuron.h step_neuron()
// the CPU uses, per-cell spike counts match the reference exactly. See
// ../THEORY.md (GPU mapping) and kernels.cuh (the big idea).
// ===========================================================================
#include "kernels.cuh"
#include "neuron.h"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

// 128 threads/block is a solid occupancy default across sm_75..sm_89. Each
// thread holds one NeuronState (a few hundred bytes) in local memory; that is
// the register/local-memory pressure, not shared memory, so block size is free.
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// step_kernel: advance EVERY neuron by one timestep.
//   grid  : ceil(ncell / 128) blocks
//   block : 128 threads
//   thread (blockIdx.x, threadIdx.x) -> neuron index i = bx*blockDim.x + tx
//
//   Each thread:
//     1. reads its presynaptic partner's spike from spike_prev[] (last step),
//     2. calls step_neuron() to integrate its own neuron one dt in place
//        (state persists in global memory `net[]` between launches),
//     3. writes its own spike (0/1) into spike_now[],
//     4. bumps its CellResult (spike_count, first_spike) on a spike.
//
//   `t` is the current step index (for recording first_spike). There is NO
//   inter-thread communication within the kernel -- coupling flows only through
//   the previous step's spike buffer -- so the result is independent of thread
//   scheduling and hence deterministic.
// ---------------------------------------------------------------------------
__global__ void step_kernel(NeuronState* __restrict__ net,
                            const int* __restrict__ spike_prev,
                            int* __restrict__ spike_now,
                            CellResult* __restrict__ results,
                            NetworkConfig c, int t) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.ncell) return;                       // guard the ragged last block

    // Synaptic drive: did this neuron's ring predecessor fire on the LAST step?
    const int pre      = presynaptic_of(i, c.ncell);
    const int spike_in = spike_prev[pre];

    // Integrate one dt in place (state lives in global memory across launches).
    const int fired = step_neuron(net[i], c.hh, c.dt, spike_in);

    spike_now[i] = fired;                            // publish for next step's read
    if (fired) {
        results[i].spike_count += 1;
        if (results[i].first_spike < 0) results[i].first_spike = t;
    }
}

// ---------------------------------------------------------------------------
// seed_kernel: one-time initialisation on the device.
//   Puts every neuron at rest, then depolarises the first n_stim somata to start
//   the travelling wave -- exactly mirroring integrate_cpu()'s seeding so the two
//   runs begin from byte-identical state. results[] is zeroed here too.
// ---------------------------------------------------------------------------
__global__ void seed_kernel(NeuronState* __restrict__ net,
                            CellResult* __restrict__ results,
                            NetworkConfig c) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.ncell) return;
    init_neuron(net[i], c.ncomp, c.v_rest);
    if (i < c.n_stim) net[i].V[0] = c.v_rest + c.i_stim;   // kick the leading cells
    results[i].spike_count = 0;
    results[i].first_spike = -1;
}

// ---------------------------------------------------------------------------
// integrate_gpu: allocate device buffers, seed, then run the per-step launch
// loop with ping-pong spike buffers, and copy the per-cell results back.
// ---------------------------------------------------------------------------
void integrate_gpu(const NetworkConfig& c,
                   std::vector<CellResult>& results,
                   float* kernel_ms) {
    const int NC = c.ncell;
    results.assign(NC, CellResult{0, -1});

    // --- Device allocations -------------------------------------------------
    NeuronState* d_net = nullptr;      // per-neuron dynamical state
    CellResult*  d_res = nullptr;      // per-neuron spike summary
    int* d_spike_a = nullptr;          // ping-pong spike buffer A
    int* d_spike_b = nullptr;          // ping-pong spike buffer B
    CUDA_CHECK(cudaMalloc(&d_net, sizeof(NeuronState) * NC));
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(CellResult)  * NC));
    CUDA_CHECK(cudaMalloc(&d_spike_a, sizeof(int) * NC));
    CUDA_CHECK(cudaMalloc(&d_spike_b, sizeof(int) * NC));
    // Both spike buffers start at 0 (no one has fired before step 0).
    CUDA_CHECK(cudaMemset(d_spike_a, 0, sizeof(int) * NC));
    CUDA_CHECK(cudaMemset(d_spike_b, 0, sizeof(int) * NC));

    const int blocks = (NC + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // --- Seed on device (matches CPU initial condition) ---------------------
    seed_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_net, d_res, c);
    CUDA_CHECK_LAST("seed_kernel");

    // --- Timed per-step launch loop -----------------------------------------
    // spike_prev = buffer holding LAST step's spikes (read); spike_now = buffer
    // we write THIS step. After each launch we swap the two pointers so this
    // step's output becomes next step's input -- the ping-pong. The kernel
    // boundary is the grid-wide barrier that makes the coupling well-defined.
    int* spike_prev = d_spike_a;
    int* spike_now  = d_spike_b;

    GpuTimer timer;
    timer.start();
    for (int t = 0; t < c.steps; ++t) {
        step_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_net, spike_prev, spike_now, d_res, c, t);
        int* tmp = spike_prev; spike_prev = spike_now; spike_now = tmp;  // ping-pong swap
    }
    *kernel_ms = timer.stop_ms();          // total GPU time across all launches
    CUDA_CHECK_LAST("step_kernel");        // catches any launch/execution error

    // --- Copy per-cell summaries back ---------------------------------------
    CUDA_CHECK(cudaMemcpy(results.data(), d_res, sizeof(CellResult) * NC,
                          cudaMemcpyDeviceToHost));

    // --- Free ---------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_net));
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_spike_a));
    CUDA_CHECK(cudaFree(d_spike_b));
}
