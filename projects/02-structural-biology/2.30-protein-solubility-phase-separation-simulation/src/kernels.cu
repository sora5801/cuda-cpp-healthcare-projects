// ===========================================================================
// src/kernels.cu  --  GPU velocity-Verlet HPS molecular dynamics
// ---------------------------------------------------------------------------
// Project 2.30 : Protein Solubility & Phase Separation Simulation
//
// GPU twin of run_cpu(): the SAME shared physics (bead_force in hps_model.h) and
// the SAME fixed pair order, run in parallel (one thread per bead) instead of
// serially. main.cu runs both and asserts their final-state summaries agree.
// See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "hps_model.h"           // the shared __host__ __device__ force/energy core
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <vector>

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, 8 warps to hide global-memory latency, and many blocks resident
// for occupancy. The force kernel is the heavy one (each thread scans all beads).
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// force_kernel: one thread computes the force on one bead (the "gather" pattern).
//   Launch config (set in run_gpu):
//     grid  = ceil(N / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: i = blockIdx.x*blockDim.x + threadIdx.x owns bead i.
//   Memory: READS all of x/y/z/lam/chain from global memory (every thread scans
//   the whole arrays -- the O(N^2) cost), WRITES only fx[i],fy[i],fz[i],u_half[i].
//   Because each thread writes a distinct slot there is no contention, so we
//   need neither shared memory nor atomics. The actual physics is delegated to
//   the shared bead_force() so it is identical to the CPU path.
//   (A production code would tile the j-loop positions into shared memory to
//   reuse them across the block; we keep the straight global-memory version for
//   clarity -- see THEORY.md "GPU mapping" for the tiling optimization.)
// ---------------------------------------------------------------------------
__global__ void force_kernel(int N,
                             const double* __restrict__ x,
                             const double* __restrict__ y,
                             const double* __restrict__ z,
                             const double* __restrict__ lam,
                             const int* __restrict__ chain,
                             SimParams p,
                             double* __restrict__ fx,
                             double* __restrict__ fy,
                             double* __restrict__ fz,
                             double* __restrict__ u_half) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's bead
    if (i >= N) return;                              // guard the ragged last block

    double Fx, Fy, Fz, U;
    // The one true force routine -- byte-identical to what the CPU reference
    // calls, including the fixed j = 0..N-1 summation order.
    bead_force(i, N, x, y, z, lam, chain, p, &Fx, &Fy, &Fz, &U);
    fx[i] = Fx; fy[i] = Fy; fz[i] = Fz;
    u_half[i] = U;
}

// ---------------------------------------------------------------------------
// integrate_kernel: one thread advances one bead by half of a velocity-Verlet
//   step. Splitting the step into two phases (with a fresh force evaluation in
//   between) is what makes the GPU integration identical to the CPU's:
//     phase 0 : v += (dt/2/m) f ;  r += dt v ;  wrap r into [0, box)
//     phase 1 : v += (dt/2/m) f                  (forces are the NEW ones)
//   The host calls force_kernel between phase 0 and phase 1 so that phase 1 sees
//   the updated forces -- exactly the (1,2) then (3) then (4) ordering of the
//   serial reference in reference_cpu.cpp.
// ---------------------------------------------------------------------------
__global__ void integrate_kernel(int N, double dt, double mass, double box,
                                 int phase,
                                 double* __restrict__ x,
                                 double* __restrict__ y,
                                 double* __restrict__ z,
                                 double* __restrict__ vx,
                                 double* __restrict__ vy,
                                 double* __restrict__ vz,
                                 const double* __restrict__ fx,
                                 const double* __restrict__ fy,
                                 const double* __restrict__ fz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const double half_dt_over_m = 0.5 * dt / mass;   // force -> half-step velocity change

    // Both phases do the half-kick; only phase 0 also drifts + wraps positions.
    vx[i] += half_dt_over_m * fx[i];
    vy[i] += half_dt_over_m * fy[i];
    vz[i] += half_dt_over_m * fz[i];

    if (phase == 0) {
        x[i] += dt * vx[i];
        y[i] += dt * vy[i];
        z[i] += dt * vz[i];
        // Periodic wrap back into [0, box) -- matches the CPU reference exactly
        // (floor-based wrap), so the position checksum agrees.
        x[i] -= box * floor(x[i] / box);
        y[i] -= box * floor(y[i] / box);
        z[i] -= box * floor(z[i] / box);
    }
}

// ---------------------------------------------------------------------------
// run_gpu: host wrapper. Allocate device state, run n_steps Verlet steps as
//   kernel-launch pairs, copy the final state back, and build the SimSummary.
//   The summary's reductions (potential, kinetic, checksum) are done on the host
//   in index order so they are deterministic and match the CPU bit-for-bit; the
//   order parameters reuse the host order_params() on the returned positions.
// ---------------------------------------------------------------------------
void run_gpu(System sys, SimSummary& out, float* kernel_ms) {
    const int N = sys.n();
    const SimParams& p = sys.p;
    const std::size_t db = static_cast<std::size_t>(N) * sizeof(double);
    const std::size_t ib = static_cast<std::size_t>(N) * sizeof(int);

    // (1) Device buffers (d_ prefix = DEVICE pointer; CLAUDE.md §12).
    double *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_lam;
    double *d_fx, *d_fy, *d_fz, *d_u;
    int    *d_chain;
    CUDA_CHECK(cudaMalloc(&d_x, db));   CUDA_CHECK(cudaMalloc(&d_y, db));   CUDA_CHECK(cudaMalloc(&d_z, db));
    CUDA_CHECK(cudaMalloc(&d_vx, db));  CUDA_CHECK(cudaMalloc(&d_vy, db));  CUDA_CHECK(cudaMalloc(&d_vz, db));
    CUDA_CHECK(cudaMalloc(&d_lam, db)); CUDA_CHECK(cudaMalloc(&d_chain, ib));
    CUDA_CHECK(cudaMalloc(&d_fx, db));  CUDA_CHECK(cudaMalloc(&d_fy, db));  CUDA_CHECK(cudaMalloc(&d_fz, db));
    CUDA_CHECK(cudaMalloc(&d_u, db));

    // (2) Copy the initial state host->device.
    CUDA_CHECK(cudaMemcpy(d_x, sys.x.data(), db, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, sys.y.data(), db, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, sys.z.data(), db, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, sys.vx.data(), db, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, sys.vy.data(), db, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vz, sys.vz.data(), db, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lam, sys.lambda.data(), db, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chain, sys.chain_id.data(), ib, cudaMemcpyHostToDevice));

    const int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // (3) Time the whole integration loop with CUDA events (kernel time only).
    GpuTimer timer;
    timer.start();

    // Initial forces a(0).
    force_kernel<<<blocks, THREADS_PER_BLOCK>>>(N, d_x, d_y, d_z, d_lam, d_chain, p,
                                                d_fx, d_fy, d_fz, d_u);
    CUDA_CHECK_LAST("force_kernel(initial)");

    for (int step = 0; step < p.n_steps; ++step) {
        // phase 0: half-kick + drift + wrap (uses forces from the previous step)
        integrate_kernel<<<blocks, THREADS_PER_BLOCK>>>(N, p.dt, p.mass, p.box, 0,
            d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz);
        CUDA_CHECK_LAST("integrate_kernel(phase0)");
        // recompute forces at the new positions
        force_kernel<<<blocks, THREADS_PER_BLOCK>>>(N, d_x, d_y, d_z, d_lam, d_chain, p,
            d_fx, d_fy, d_fz, d_u);
        CUDA_CHECK_LAST("force_kernel(step)");
        // phase 1: second half-kick with the NEW forces
        integrate_kernel<<<blocks, THREADS_PER_BLOCK>>>(N, p.dt, p.mass, p.box, 1,
            d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz);
        CUDA_CHECK_LAST("integrate_kernel(phase1)");
    }

    // d_u currently holds the half-pair energies at the FINAL positions (the last
    // force_kernel ran at the final geometry, and phase-1 only touched velocities)
    // -- exactly the energy the CPU reports. No extra force pass needed.
    *kernel_ms = timer.stop_ms();

    // (4) Copy the final state back to host for the deterministic reductions.
    CUDA_CHECK(cudaMemcpy(sys.x.data(), d_x, db, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sys.y.data(), d_y, db, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sys.z.data(), d_z, db, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sys.vx.data(), d_vx, db, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sys.vy.data(), d_vy, db, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sys.vz.data(), d_vz, db, cudaMemcpyDeviceToHost));
    std::vector<double> u_half(N);
    CUDA_CHECK(cudaMemcpy(u_half.data(), d_u, db, cudaMemcpyDeviceToHost));

    // (5) Free device memory (no GPU garbage collector exists).
    cudaFree(d_x);  cudaFree(d_y);  cudaFree(d_z);
    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
    cudaFree(d_lam); cudaFree(d_chain);
    cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz); cudaFree(d_u);

    // ---- deterministic host-side reductions (index order = matches CPU) -----
    double pe = 0.0, ke = 0.0, checksum = 0.0;
    for (int i = 0; i < N; ++i) {
        pe += u_half[i];                 // sum of half-pair energies = total PE
        ke += 0.5 * p.mass * (sys.vx[i]*sys.vx[i] + sys.vy[i]*sys.vy[i] + sys.vz[i]*sys.vz[i]);
        checksum += sys.x[i] + sys.y[i] + sys.z[i];
    }
    out.potential = pe;
    out.kinetic   = ke;
    out.pos_checksum = checksum;
    order_params(p, sys.x, sys.y, sys.z,
                 out.max_local_density, out.mean_local_density, out.n_condensed);
}
