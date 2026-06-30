// ===========================================================================
// src/kernels.cu  --  GPU MD kernels (forces / Verlet half-kicks) + time loop
// ---------------------------------------------------------------------------
// Project 2.19 : Membrane Protein Simulation   (reduced-scope teaching version)
//
// GPU twin of simulate_cpu(): one thread per bead, identical per-bead/per-pair
// physics (membrane.h), same loop order so the floating-point sums match the
// CPU. main.cu compares the final positions/velocities. See ../THEORY.md
// "GPU mapping".
//
// Three kernels drive each step (mirroring the CPU's A/B/C phases):
//   compute_forces_kernel  -- (B) conservative forces at current x
//   kick_drift_kernel      -- (A) v += f/m*dt/2 ; x += v*dt   (f includes Langevin)
//   kick_kernel            -- (C) v += f/m*dt/2               (f includes Langevin)
// The host orders them f0 -> [A, recompute(B), C] per step, exactly like the CPU.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

// 256 threads/block is a solid occupancy default across sm_75..sm_89: it keeps
// enough warps resident to hide global-memory latency while staying within the
// register/shared-memory budget. (THEORY "GPU mapping" discusses occupancy.)
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// compute_forces_kernel: thread `i` computes the total conservative force on
// bead i and writes f[i]. This is the O(N) inner work that, summed over all
// threads, is the O(N^2) all-pairs evaluation -- but spread across N threads.
//
// We deliberately walk j in ascending index order and the bond list in stored
// order, EXACTLY as reference_cpu.cpp's compute_forces() does, so the partial
// sums round identically -> CPU/GPU agreement is near-exact, not approximate.
//
// Bonds: the CPU adds bond_force(pos[bi]-pos[bj]) to endpoint bi and its
// NEGATION to endpoint bj. Here each thread owns ONE bead, so it scans the bond
// list and, if it is an endpoint, adds the matching contribution -- computing
// the SAME bond_force(pos[bi]-pos[bj]) value and negating it for the bj side,
// so the rounding matches the CPU.
// ---------------------------------------------------------------------------
__global__ void compute_forces_kernel(SimParams P,
                                       const Vec3* __restrict__ pos,
                                       const int*  __restrict__ type,
                                       const int*  __restrict__ bond_i,
                                       const int*  __restrict__ bond_j,
                                       int n_bonds,
                                       Vec3* __restrict__ f) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's bead
    if (i >= P.n_beads) return;                           // guard ragged block

    const Vec3 ri = pos[i];
    const int  ti = type[i];
    Vec3 fi = {0.0, 0.0, 0.0};
    double u_unused;   // lj_force/bond_force also return energy; unused here.

    // --- non-bonded LJ over every other bead within the cutoff (same order) ---
    for (int j = 0; j < P.n_beads; ++j) {
        if (j == i) continue;
        const Vec3 dij = min_image_delta(ri, pos[j], P.box_x, P.box_y);
        const double e = eps_of(P, ti, type[j]);
        fi = fi + lj_force(dij, e, P.sigma, P.rcut, &u_unused);
    }

    // --- bonded springs: add the contribution wherever bead i is an endpoint ---
    for (int b = 0; b < n_bonds; ++b) {
        const int bi = bond_i[b], bj = bond_j[b];
        if (bi != i && bj != i) continue;                 // not our bond
        // Compute the canonical bond_force(pos[bi]-pos[bj]) so the value matches
        // the CPU's; add it on the bi side, subtract it on the bj side.
        const Vec3 dij = min_image_delta(pos[bi], pos[bj], P.box_x, P.box_y);
        const Vec3 fij = bond_force(dij, P.k_bond, P.r_bond, &u_unused);
        if (i == bi) fi = fi + fij;                       // endpoint bi
        else         fi = fi - fij;                       // endpoint bj (opposite)
    }

    f[i] = fi;
}

// ---------------------------------------------------------------------------
// kick_drift_kernel: Verlet phase (A) for bead i.
//   total force = conservative f[i] + Langevin(friction + deterministic kick),
//   then v += f/m*dt/2 and x += v*dt. The Langevin random force uses the shared
//   normal01() keyed by (step, bead, axis), so it is IDENTICAL to the CPU's.
// ---------------------------------------------------------------------------
__global__ void kick_drift_kernel(SimParams P, int step,
                                   const Vec3* __restrict__ f,
                                   const double* __restrict__ mass,
                                   const double* __restrict__ inv_mass,
                                   Vec3* __restrict__ pos, Vec3* __restrict__ vel) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P.n_beads) return;
    Vec3 x = pos[i], v = vel[i];
    const Vec3 fl = langevin_force(v, mass[i], P.gamma, P.temperature, P.dt,
                                   P.seed, step, i);
    const Vec3 ftot = f[i] + fl;
    verlet_kick_drift(x, v, ftot, inv_mass[i], P.dt);     // shared HD integrator
    pos[i] = x; vel[i] = v;
}

// ---------------------------------------------------------------------------
// kick_kernel: Verlet phase (C) for bead i. Re-evaluates the Langevin force at
// the (drifted) velocity and the NEW conservative force, then the final
// half-kick v += f/m*dt/2. Only velocities change here.
// ---------------------------------------------------------------------------
__global__ void kick_kernel(SimParams P, int step,
                            const Vec3* __restrict__ f,
                            const double* __restrict__ mass,
                            const double* __restrict__ inv_mass,
                            Vec3* __restrict__ vel) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P.n_beads) return;
    Vec3 v = vel[i];
    const Vec3 fl = langevin_force(v, mass[i], P.gamma, P.temperature, P.dt,
                                   P.seed, step, i);
    const Vec3 ftot = f[i] + fl;
    verlet_kick(v, ftot, inv_mass[i], P.dt);
    vel[i] = v;
}

// ---------------------------------------------------------------------------
// simulate_gpu: orchestrate the device-side MD loop.
//   1. Allocate device buffers and copy the System over (H2D).
//   2. Compute initial forces, then run `steps` of [A, recompute B, C].
//   3. Copy final pos/vel back (D2H). Time the loop with CUDA events.
//   The kernel sequence mirrors reference_cpu.cpp::simulate_cpu() one-to-one.
// ---------------------------------------------------------------------------
void simulate_gpu(const SimParams& P, System& sys, float* kernel_ms) {
    const int N = P.n_beads;
    const int n_bonds = static_cast<int>(sys.bond_i.size());
    const std::size_t vbytes = static_cast<std::size_t>(N) * sizeof(Vec3);
    const std::size_t dbytes = static_cast<std::size_t>(N) * sizeof(double);
    const std::size_t ibytes = static_cast<std::size_t>(N) * sizeof(int);
    const std::size_t bbytes = static_cast<std::size_t>(n_bonds) * sizeof(int);

    // Device pointers (d_ prefix marks DEVICE pointers per the repo style).
    Vec3   *d_pos = nullptr, *d_vel = nullptr, *d_f = nullptr;
    double *d_mass = nullptr, *d_inv = nullptr;
    int    *d_type = nullptr, *d_bi = nullptr, *d_bj = nullptr;

    CUDA_CHECK(cudaMalloc(&d_pos, vbytes));
    CUDA_CHECK(cudaMalloc(&d_vel, vbytes));
    CUDA_CHECK(cudaMalloc(&d_f,   vbytes));
    CUDA_CHECK(cudaMalloc(&d_mass, dbytes));
    CUDA_CHECK(cudaMalloc(&d_inv,  dbytes));
    CUDA_CHECK(cudaMalloc(&d_type, ibytes));
    // n_bonds can be 0 in pathological inputs; guard the malloc/copy.
    if (n_bonds > 0) {
        CUDA_CHECK(cudaMalloc(&d_bi, bbytes));
        CUDA_CHECK(cudaMalloc(&d_bj, bbytes));
    }

    CUDA_CHECK(cudaMemcpy(d_pos, sys.pos.data(), vbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, sys.vel.data(), vbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mass, sys.mass.data(), dbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inv,  sys.inv_mass.data(), dbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_type, sys.type.data(), ibytes, cudaMemcpyHostToDevice));
    if (n_bonds > 0) {
        CUDA_CHECK(cudaMemcpy(d_bi, sys.bond_i.data(), bbytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bj, sys.bond_j.data(), bbytes, cudaMemcpyHostToDevice));
    }

    const int grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    GpuTimer timer;
    timer.start();
    // Initial conservative forces (matches the CPU's pre-loop compute_forces).
    compute_forces_kernel<<<grid, THREADS_PER_BLOCK>>>(P, d_pos, d_type, d_bi, d_bj, n_bonds, d_f);
    for (int step = 0; step < P.steps; ++step) {
        // (A) half-kick + drift  (f includes Langevin inside the kernel)
        kick_drift_kernel<<<grid, THREADS_PER_BLOCK>>>(P, step, d_f, d_mass, d_inv, d_pos, d_vel);
        // (B) recompute conservative forces at the new positions
        compute_forces_kernel<<<grid, THREADS_PER_BLOCK>>>(P, d_pos, d_type, d_bi, d_bj, n_bonds, d_f);
        // (C) final half-kick
        kick_kernel<<<grid, THREADS_PER_BLOCK>>>(P, step, d_f, d_mass, d_inv, d_vel);
    }
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("md kernels");   // catch any launch/exec error from the loop

    // Copy the final state back into the host System (in place).
    CUDA_CHECK(cudaMemcpy(sys.pos.data(), d_pos, vbytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sys.vel.data(), d_vel, vbytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_vel));
    CUDA_CHECK(cudaFree(d_f));
    CUDA_CHECK(cudaFree(d_mass));
    CUDA_CHECK(cudaFree(d_inv));
    CUDA_CHECK(cudaFree(d_type));
    if (n_bonds > 0) { CUDA_CHECK(cudaFree(d_bi)); CUDA_CHECK(cudaFree(d_bj)); }
}
