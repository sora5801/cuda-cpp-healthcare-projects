// ===========================================================================
// src/kernels.cu  --  GPU LIF-network kernels + host driver
// ---------------------------------------------------------------------------
// Project 6.7 : Spiking Neural Network (Point-Neuron) Simulation
//
// GPU twin of simulate_cpu(). Same physics (lif.h), same fixed-point synaptic
// accumulation, same one-step synaptic delay -> the spike counts match the CPU
// EXACTLY (verification tolerance == 0). main.cu runs both and compares.
//
// The per-step work is split into three kernels (see ../THEORY.md "GPU mapping"):
//   1. deliver_kernel  -- one thread per neuron; if it spiked LAST step, scatter
//                         its weight into all its targets with atomicAdd on an
//                         INTEGER fixed-point buffer (order-independent -> exact).
//   2. external_kernel -- one thread per neuron; add this step's deterministic
//                         background drive into the same fixed-point buffer.
//   3. update_kernel   -- one thread per neuron; convert accumulated input to a
//                         double, call lif_step(), record spikes, roll buffers.
// We keep the whole time loop on the device (no per-step host sync) so the timed
// region measures real kernel work, not launch/copy overhead.
//
// READ THIS AFTER: lif.h, reference_cpu.cpp, kernels.cuh.
// ===========================================================================
#include "kernels.cuh"
#include "lif.h"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <vector>

// 256 threads/block is a solid occupancy default on sm_75..sm_89 and divides most
// neuron counts cleanly; the ragged last block is guarded in every kernel.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// atomic_add_fixed: atomically add a SIGNED int64 increment into a fixed-point
//   accumulator. CUDA's atomicAdd overload is for `unsigned long long`, but
//   integer addition is the same bit operation for signed and unsigned two's-
//   complement values, so we reinterpret the signed increment as unsigned, add,
//   and the stored bits are the correct signed sum. This is what makes the
//   synaptic scatter both parallel AND exactly equal to the serial CPU sum
//   (integer adds commute -- PATTERNS.md sec.3).
// ---------------------------------------------------------------------------
__device__ inline void atomic_add_fixed(long long* addr, long long inc) {
    atomicAdd(reinterpret_cast<unsigned long long*>(addr),
              static_cast<unsigned long long>(inc));
}

// ---------------------------------------------------------------------------
// init_kernel: set each neuron's starting state from the shared init_state().
//   grid : ceil(n / block) ; block : 256 ; thread i owns neuron i.
//   Writing the same deterministic ICs the CPU uses is what lets the two
//   trajectories line up from step 0.
// ---------------------------------------------------------------------------
__global__ void init_kernel(NetworkConfig c, NeuronState* __restrict__ state) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= net_n(c)) return;                 // guard the ragged last block
    state[i] = init_state(c, i);
}

// ---------------------------------------------------------------------------
// external_kernel: add the deterministic background drive for step `t`.
//   thread i -> neuron i. external_drive_fixed() is shared with the CPU, so the
//   same neurons get kicked on the same steps.
// ---------------------------------------------------------------------------
__global__ void external_kernel(NetworkConfig c, int t, long long* __restrict__ input_fixed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= net_n(c)) return;
    input_fixed[i] += external_drive_fixed(c, i, t);   // private slot: no atomic needed
}

// ---------------------------------------------------------------------------
// deliver_kernel: scatter last step's spikes along the sparse graph.
//   thread s -> source neuron s. If s spiked last step, it walks its out_degree
//   synapses and atomicAdds its (fixed-point) weight into each target's slot.
//   MANY source threads can target the SAME neuron in the same step, which is why
//   the accumulation must be atomic. Using fixed-point keeps it deterministic.
//   (This is the "atomic scatter" that dominates SNN runtime; production codes
//   sort spikes by target or use per-block staging to cut atomic contention --
//   see THEORY "Where this sits in the real world".)
// ---------------------------------------------------------------------------
__global__ void deliver_kernel(NetworkConfig c,
                               const unsigned char* __restrict__ spiked_prev,
                               long long* __restrict__ input_fixed) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= net_n(c)) return;
    if (!spiked_prev[s]) return;               // only spikers deliver -> divergence,
                                               // but most neurons skip cheaply.
    const long long w_fixed = syn_to_fixed(src_weight(c, s));
    for (int k = 0; k < c.out_degree; ++k) {
        const int tgt = synapse_target(c, s, k);
        atomic_add_fixed(&input_fixed[tgt], w_fixed);
    }
}

// ---------------------------------------------------------------------------
// update_kernel: advance every neuron one step with the shared lif_step().
//   thread i -> neuron i. Reads its accumulated fixed-point input, converts to a
//   double drive, calls lif_step (identical math to the CPU), and records whether
//   it spiked into spiked_now[i]. Spiking neurons atomically bump a single global
//   counter for this step's population spike count (an integer atomic -> exact).
//   We do NOT zero input_fixed here; the host memset-zeros it before the next
//   step's external+deliver kernels (clearer and avoids an extra write here).
// ---------------------------------------------------------------------------
__global__ void update_kernel(NetworkConfig c,
                              NeuronState* __restrict__ state,
                              const long long* __restrict__ input_fixed,
                              unsigned char* __restrict__ spiked_now,
                              int* __restrict__ spikes_per_neuron,
                              unsigned long long* __restrict__ step_spike_count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= net_n(c)) return;

    const double drive = syn_from_fixed(input_fixed[i]);
    NeuronState s = state[i];                  // pull to registers, update, write back
    const int fired = lif_step(c.np, s, drive);
    state[i] = s;

    spiked_now[i] = (unsigned char)fired;
    if (fired) {
        spikes_per_neuron[i] += 1;                       // private slot: no atomic
        atomicAdd(step_spike_count, 1ULL);               // shared counter: atomic (int)
    }
}

// ---------------------------------------------------------------------------
// simulate_gpu: the host driver. Allocates device buffers, runs the time loop
//   entirely on the GPU (three kernels/step), and copies the deterministic
//   summary back once at the end.
//
//   Device buffers:
//     d_state           [n]      NeuronState per neuron
//     d_spiked_a/_b     [n]      double-buffered "did neuron spike?" flags
//                                (ping-pong: read prev, write now, then swap)
//     d_input_fixed     [n]      int64 fixed-point synaptic accumulator (zeroed/step)
//     d_spn             [n]      per-neuron spike counts
//     d_step_counts     [steps]  population spike count per step (one atomic target
//                                per step; we point the update kernel at slot t)
// ---------------------------------------------------------------------------
void simulate_gpu(const NetworkConfig& c, SimResult& out, float* kernel_ms) {
    const int n = net_n(c);
    const int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    NeuronState*        d_state = nullptr;
    unsigned char*      d_spiked_a = nullptr;   // ping
    unsigned char*      d_spiked_b = nullptr;   // pong
    long long*          d_input_fixed = nullptr;
    int*                d_spn = nullptr;              // per-neuron spike counts
    unsigned long long* d_step_counts = nullptr;      // per-step population counts

    CUDA_CHECK(cudaMalloc(&d_state,       (size_t)n * sizeof(NeuronState)));
    CUDA_CHECK(cudaMalloc(&d_spiked_a,    (size_t)n * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_spiked_b,    (size_t)n * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_input_fixed, (size_t)n * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_spn,         (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_step_counts, (size_t)c.steps * sizeof(unsigned long long)));

    // Zero the persistent accumulators; init the neuron states on-device.
    CUDA_CHECK(cudaMemset(d_spiked_a,    0, (size_t)n * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemset(d_spiked_b,    0, (size_t)n * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemset(d_spn,         0, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_step_counts, 0, (size_t)c.steps * sizeof(unsigned long long)));
    init_kernel<<<blocks, THREADS_PER_BLOCK>>>(c, d_state);
    CUDA_CHECK_LAST("init_kernel");

    // Ping-pong pointers: prev = spikes we read this step, now = spikes we write.
    unsigned char* d_prev = d_spiked_a;
    unsigned char* d_now  = d_spiked_b;

    GpuTimer timer;
    timer.start();
    for (int t = 0; t < c.steps; ++t) {
        // (0) clear this step's synaptic accumulator.
        CUDA_CHECK(cudaMemsetAsync(d_input_fixed, 0, (size_t)n * sizeof(long long)));
        // (1) deterministic external drive.
        external_kernel<<<blocks, THREADS_PER_BLOCK>>>(c, t, d_input_fixed);
        // (2) deliver last step's spikes (atomic scatter).
        deliver_kernel<<<blocks, THREADS_PER_BLOCK>>>(c, d_prev, d_input_fixed);
        // (3) update neurons; write this step's spikes into d_now and slot t count.
        update_kernel<<<blocks, THREADS_PER_BLOCK>>>(c, d_state, d_input_fixed, d_now,
                                                     d_spn, d_step_counts + t);
        // ping-pong: this step's spikes become next step's "prev".
        unsigned char* tmp = d_prev; d_prev = d_now; d_now = tmp;
    }
    *kernel_ms = timer.stop_ms();     // one sync at the end -> times the whole loop
    CUDA_CHECK_LAST("simulate_gpu time loop");

    // ---- Copy the deterministic summary back to the host -------------------
    out.spikes_per_neuron.assign(n, 0);
    out.spikes_per_step.assign(c.steps, 0);
    out.final_v.assign(n, 0.0);

    CUDA_CHECK(cudaMemcpy(out.spikes_per_neuron.data(), d_spn,
                          (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));

    // Per-step counts come back as unsigned long long; narrow into ints for the
    // report (counts are tiny). Also accumulate the grand total here.
    std::vector<unsigned long long> step_counts(c.steps);
    CUDA_CHECK(cudaMemcpy(step_counts.data(), d_step_counts,
                          (size_t)c.steps * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    out.total_spikes = 0;
    for (int t = 0; t < c.steps; ++t) {
        out.spikes_per_step[t] = (int)step_counts[t];
        out.total_spikes += (long long)step_counts[t];
    }

    // Final membrane potentials: copy the NeuronState array and pull out v.
    std::vector<NeuronState> final_states(n);
    CUDA_CHECK(cudaMemcpy(final_states.data(), d_state,
                          (size_t)n * sizeof(NeuronState), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) out.final_v[i] = final_states[i].v;

    const double duration_s = c.steps * c.np.dt * 1.0e-3;
    out.mean_rate_hz = (n > 0 && duration_s > 0.0)
                     ? (double)out.total_spikes / ((double)n * duration_s)
                     : 0.0;

    CUDA_CHECK(cudaFree(d_state));
    CUDA_CHECK(cudaFree(d_spiked_a));
    CUDA_CHECK(cudaFree(d_spiked_b));
    CUDA_CHECK(cudaFree(d_input_fixed));
    CUDA_CHECK(cudaFree(d_spn));
    CUDA_CHECK(cudaFree(d_step_counts));
}
