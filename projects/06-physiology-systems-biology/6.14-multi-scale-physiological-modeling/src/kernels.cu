// ===========================================================================
// src/kernels.cu  --  Monodomain split-step kernels (one thread per node)
// ---------------------------------------------------------------------------
// Project 6.14 : Multi-Scale Physiological Modeling
//
// GPU twin of simulate_cpu(): the same operator-split reaction-diffusion loop,
// but every node is advanced by its own thread each global step. All per-node
// arithmetic comes from the shared multiscale.h routines, so the GPU field
// matches the CPU field to a documented tolerance. See ../THEORY.md.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// 128 threads/block: a multiple of the 32-lane warp, giving the scheduler four
// warps per block to hide the global-memory latency of the stencil neighbour
// loads. Fine for the small cables here; tune per GPU/problem.
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// react_kernel : REACTION sub-step (the FINE, sub-grid scale).
//   grid   : ceil(n / THREADS_PER_BLOCK) blocks
//   block  : THREADS_PER_BLOCK threads
//   thread : node  i = blockIdx.x*blockDim.x + threadIdx.x  (one node per thread)
//   Advances that node's cell ODE (FHN) by one RK4 step in registers, touching
//   ONLY its own (v[i], w[i]) -- no neighbour reads, no shared memory, no
//   atomics. This is embarrassingly parallel: the whole point of putting the
//   millions of sub-grid cell ODEs on the GPU.
// ---------------------------------------------------------------------------
__global__ void react_kernel(int n, FhnParams p, double dt,
                             double* __restrict__ v, double* __restrict__ w) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;                       // guard the ragged last block
    double vi = v[i], wi = w[i];              // pull state into registers
    react_rk4_step(vi, wi, p, dt);            // shared HD routine (multiscale.h)
    v[i] = vi;                                // write the reacted state back
    w[i] = wi;
}

// ---------------------------------------------------------------------------
// diffuse_kernel : DIFFUSION sub-step (the COARSE, tissue scale) via ping-pong.
//   Reads the OLD voltage field v_in (including neighbours) and writes the new
//   field v_out, so no thread ever reads a value another thread has already
//   overwritten -- the Jacobi/ping-pong pattern (flagship 6.04 / 14.02). Uses
//   the shared stencil + mirrored (zero-flux) boundaries so it indexes the cable
//   IDENTICALLY to the CPU reference.
//     v_out[i] = v_in[i] + dt * D * laplacian(v_in)
//   thread : node i owns output v_out[i].
// ---------------------------------------------------------------------------
__global__ void diffuse_kernel(int n, FhnParams p, double dt, double dx,
                              const double* __restrict__ v_in,
                              double* __restrict__ v_out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double left  = mirror_left(v_in, n, i);      // reflecting boundary
    const double right = mirror_right(v_in, n, i);
    const double lap   = diffusion_laplacian(left, v_in[i], right, dx);
    v_out[i] = v_in[i] + dt * p.D * lap;
}

// ---------------------------------------------------------------------------
// activation_kernel : record the FIRST time each node crosses v >= 0.5.
//   act[i] holds -1 until node i activates, then the step time t. We only write
//   when the node is still unactivated AND is now above threshold, so the first
//   crossing wins. Each thread touches only act[i] -> no races. `t` is passed in
//   (the current step time) so the recorded value is deterministic.
// ---------------------------------------------------------------------------
__global__ void activation_kernel(int n, double t, double thresh,
                                  const double* __restrict__ v,
                                  double* __restrict__ act) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (act[i] < 0.0 && v[i] >= thresh) act[i] = t;
}

// ---------------------------------------------------------------------------
// simulate_gpu : orchestrate the whole run on the device.
//   Buffers: two voltage arrays (d_v, d_v2) for ping-pong, one recovery array
//   (d_w), one activation-time array (d_act). Per global step we launch three
//   tiny kernels (react -> diffuse -> record). We time the SUM of on-device
//   step time with CUDA events (a teaching artifact -- these many tiny launches
//   are launch-bound on small cables, see THEORY / PATTERNS.md section 7).
// ---------------------------------------------------------------------------
void simulate_gpu(const CableConfig& c, CableResult& out, float* kernel_ms) {
    const int n = c.n;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(double);
    const double THRESH = 0.5;   // activation threshold (matches the CPU reference)

    // Device buffers. d_v / d_v2 are the ping-pong pair for the diffusion sweep.
    double *d_v = nullptr, *d_v2 = nullptr, *d_w = nullptr, *d_act = nullptr;
    CUDA_CHECK(cudaMalloc(&d_v,   bytes));
    CUDA_CHECK(cudaMalloc(&d_v2,  bytes));
    CUDA_CHECK(cudaMalloc(&d_w,   bytes));
    CUDA_CHECK(cudaMalloc(&d_act, bytes));

    // Host-side initial condition (mirror of simulate_cpu): zero fields, then
    // stimulate the first stim_nodes nodes to v=1, activation time 0 there.
    std::vector<double> v0(n, 0.0), w0(n, 0.0), act0(n, -1.0);
    for (int i = 0; i < c.stim_nodes; ++i) { v0[i] = 1.0; act0[i] = 0.0; }

    CUDA_CHECK(cudaMemcpy(d_v,   v0.data(),   bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w,   w0.data(),   bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_act, act0.data(), bytes, cudaMemcpyHostToDevice));

    const int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Time the whole stepping loop with one pair of CUDA events.
    GpuTimer timer;
    timer.start();
    for (int s = 1; s <= c.steps; ++s) {
        const double t = s * c.dt;

        // (A) reaction: independent cell ODE at every node (in place on d_v/d_w).
        react_kernel<<<blocks, THREADS_PER_BLOCK>>>(n, c.p, c.dt, d_v, d_w);

        // (B) diffusion: read d_v (old), write d_v2 (new) -> ping-pong, then swap
        //     the pointers so d_v again holds the current field for the next step.
        diffuse_kernel<<<blocks, THREADS_PER_BLOCK>>>(n, c.p, c.dt, c.dx, d_v, d_v2);
        double* tmp = d_v; d_v = d_v2; d_v2 = tmp;

        // (C) record first-crossing activation times.
        activation_kernel<<<blocks, THREADS_PER_BLOCK>>>(n, t, THRESH, d_v, d_act);
    }
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("monodomain step kernels");   // catch launch/exec errors

    // Copy the final field + activation map back to the host result.
    out.v_final.assign(n, 0.0);
    out.w_final.assign(n, 0.0);
    out.activation_time.assign(n, -1.0);
    CUDA_CHECK(cudaMemcpy(out.v_final.data(),        d_v,   bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.w_final.data(),        d_w,   bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.activation_time.data(), d_act, bytes, cudaMemcpyDeviceToHost));

    // Derive the summary metrics with the SAME routine the CPU uses (defined in
    // reference_cpu.cpp) so the reported CV/count are computed identically.
    summarize_activation(c, out.activation_time, out.n_activated, out.conduction_velocity);

    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_v2));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_act));
}
