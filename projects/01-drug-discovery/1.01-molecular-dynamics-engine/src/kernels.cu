// ===========================================================================
// src/kernels.cu  --  GPU velocity-Verlet MD: tiled all-pairs force + integrate
// ---------------------------------------------------------------------------
// Project 1.1 : Molecular Dynamics Engine  (reduced-scope teaching version)
//
// WHAT THIS FILE DOES
//   The GPU twin of integrate_cpu(). It keeps the whole trajectory ON THE DEVICE
//   for the entire run (positions, velocities, forces never round-trip to the host
//   between steps -- that would be the classic performance killer), launching a
//   small set of kernels each timestep:
//       forces_kernel : one thread per atom i sums the LJ force from all j,
//                       using SHARED-MEMORY TILING to reuse positions (the key
//                       optimization), and also tallies a per-atom potential and
//                       kinetic energy for the diagnostics.
//       half_kick_kernel / drift_kernel : the velocity-Verlet sub-steps, one
//                       thread per atom (pure data-parallel, no communication).
//   All per-pair physics and Verlet arithmetic come from md.h -- the SAME code the
//   CPU reference runs -- so the two trajectories agree to round-off (PATTERNS §2).
//
//   ENERGY REDUCTION & DETERMINISM: floating-point atomicAdd is not associative,
//   so summing energies with atomics across thousands of threads would give a
//   run-to-run-varying total (PATTERNS.md §3). To keep stdout reproducible we do
//   NOT reduce on the device with atomics: the kernel writes a per-atom energy
//   array, we copy it back, and main.cu's caller sums it in a FIXED index order
//   (here, inside integrate_gpu) -- deterministic, and matching the CPU's order as
//   closely as floating point allows. The remaining CPU/GPU gap is pure FMA/order
//   round-off and is absorbed by a documented physical tolerance (THEORY §numerics).
//
// READ THIS AFTER: kernels.cuh (the idea), md.h (the physics).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer

#include <vector>

// Threads per block == tile width. 128 is a good occupancy default on sm_75..89
// and keeps the shared-memory tile (128 * sizeof(Vec3) = 3 KiB) small. It must be
// the tile size because each thread loads exactly one atom into the shared tile.
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// forces_kernel: compute F[i] (total LJ force on atom i) for every atom, plus a
//   per-atom potential energy pe[i] and kinetic energy ke[i] for the diagnostics.
//
//   Launch config (set in integrate_gpu):
//     grid  = ceil(n / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads;  thread (blockIdx.x, threadIdx.x) owns
//             atom i = blockIdx.x*blockDim.x + threadIdx.x.
//   Shared memory: a tile of THREADS_PER_BLOCK atom positions, reloaded once per
//     tile and reused by all threads in the block -> ~TILE-fold fewer global loads.
//   Memory spaces: reads pos/vel from global, writes F/pe/ke to global, stages
//     positions through shared; no atomics (each thread owns its outputs).
//
//   POTENTIAL DOUBLE-COUNTING: thread i loops over ALL j (not just j>i), so each
//   unordered pair {i,j} is seen twice -- once by thread i, once by thread j. The
//   force is correct (thread i only wants the force ON i). For energy we therefore
//   accumulate HALF of each pair's U into pe[i]; summed over atoms this recovers
//   the each-pair-once total. (The CPU counts each pair once with i<j; same total.)
// ---------------------------------------------------------------------------
__global__ void forces_kernel(SimParams p,
                              const Vec3* __restrict__ pos,
                              const Vec3* __restrict__ vel,
                              Vec3* __restrict__ F,
                              double* __restrict__ pe,
                              double* __restrict__ ke) {
    // Shared tile of positions, cooperatively loaded by the block. Sized to the
    // block width so each thread loads exactly one atom per tile.
    __shared__ Vec3 tile[THREADS_PER_BLOCK];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's atom
    const int n = p.n;

    // Each thread keeps ITS atom's position in registers (read once). Guard the
    // ragged last block: threads with i>=n still participate in the cooperative
    // tile loads (so __syncthreads is uniform) but must not write results.
    Vec3 ri = (i < n) ? pos[i] : vec3_zero();
    Vec3 fi = vec3_zero();        // running force on atom i (registers)
    double ui = 0.0;              // running HALF-potential for atom i (registers)

    // Sweep all atoms in tiles of blockDim.x. `base` is the first atom in the tile.
    for (int base = 0; base < n; base += blockDim.x) {
        // (a) Cooperative load: thread t loads atom (base+t) into shared tile[t].
        //     One global read per atom per block instead of one per (i,j) pair.
        const int j_load = base + threadIdx.x;
        tile[threadIdx.x] = (j_load < n) ? pos[j_load] : vec3_zero();
        __syncthreads();          // tile fully populated before anyone reads it

        // (b) This thread interacts its atom i with every atom in the tile.
        const int tile_count = min(blockDim.x, n - base);  // valid entries in tile
        for (int t = 0; t < tile_count; ++t) {
            const int j = base + t;
            if (i < n && j != i) {                 // skip self; only real atoms
                Vec3 rij;
                rij.x = minimum_image(ri.x - tile[t].x, p.box);
                rij.y = minimum_image(ri.y - tile[t].y, p.box);
                rij.z = minimum_image(ri.z - tile[t].z, p.box);
                double upair = 0.0;
                Vec3 fij = lj_pair_force(rij, p, &upair);  // shared md.h physics
                fi.x += fij.x; fi.y += fij.y; fi.z += fij.z;
                ui   += 0.5 * upair;               // half (pair seen twice overall)
            }
        }
        __syncthreads();          // all done reading the tile before it is reloaded
    }

    // (c) Write this atom's results (real atoms only).
    if (i < n) {
        F[i]  = fi;
        pe[i] = ui;
        ke[i] = kinetic_energy_one(vel[i], p.mass);
    }
}

// ---------------------------------------------------------------------------
// half_kick_kernel: v[i] += (dt/2) * F[i]/m. One thread per atom, no comms.
//   Called twice per step (before and after the force eval) -> the two half-kicks
//   of velocity-Verlet. Uses the same arithmetic as the CPU reference.
// ---------------------------------------------------------------------------
__global__ void half_kick_kernel(SimParams p, Vec3* __restrict__ vel,
                                 const Vec3* __restrict__ F) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;                         // guard ragged last block
    const double s = 0.5 * p.dt / p.mass;         // (dt/2)/m, matches CPU
    vel[i].x += s * F[i].x;
    vel[i].y += s * F[i].y;
    vel[i].z += s * F[i].z;
}

// ---------------------------------------------------------------------------
// drift_kernel: x[i] += dt * v[i], then wrap back into the periodic box. One
//   thread per atom. Mirrors the CPU drift step exactly (wrap_into_box, md.h).
// ---------------------------------------------------------------------------
__global__ void drift_kernel(SimParams p, Vec3* __restrict__ pos,
                            const Vec3* __restrict__ vel) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;
    pos[i].x = wrap_into_box(pos[i].x + p.dt * vel[i].x, p.box);
    pos[i].y = wrap_into_box(pos[i].y + p.dt * vel[i].y, p.box);
    pos[i].z = wrap_into_box(pos[i].z + p.dt * vel[i].z, p.box);
}

// ---------------------------------------------------------------------------
// reduce_host: deterministic sum of a device array, copied back and summed in a
//   FIXED index order on the host. We sum in double precision in index order so
//   the result is reproducible (PATTERNS.md §3: avoid float-atomic nondeterminism)
//   and as close to the CPU reference's order as floating point permits.
//   Returns the total; `host_buf` is scratch reused across calls to avoid realloc.
// ---------------------------------------------------------------------------
static double reduce_host(const double* d_arr, int n, std::vector<double>& host_buf) {
    host_buf.resize(static_cast<std::size_t>(n));
    CUDA_CHECK(cudaMemcpy(host_buf.data(), d_arr,
                          static_cast<std::size_t>(n) * sizeof(double),
                          cudaMemcpyDeviceToHost));
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += host_buf[i];   // fixed order -> deterministic
    return s;
}

// ---------------------------------------------------------------------------
// integrate_gpu: orchestrate the device-side simulation (see kernels.cuh).
//   Steps: (1) alloc device state, (2) copy initial state H2D, (3) initial force
//   eval + E0, (4) the velocity-Verlet loop with the energy-drift diagnostic,
//   (5) read back final state for the checksum, (6) free. The kernel time is the
//   sum of integration-step kernels (the diagnostic energy copies are excluded).
// ---------------------------------------------------------------------------
MdResult integrate_gpu(const MdSystem& sys, float* kernel_ms) {
    const SimParams p = sys.params;
    const int n = p.n;
    const std::size_t vbytes = static_cast<std::size_t>(n) * sizeof(Vec3);
    const std::size_t dbytes = static_cast<std::size_t>(n) * sizeof(double);

    // (1) Device buffers (d_ prefix == device pointer, CLAUDE.md §12). pe/ke are
    //     per-atom energy scratch the forces_kernel fills for the diagnostics.
    Vec3 *d_pos = nullptr, *d_vel = nullptr, *d_F = nullptr;
    double *d_pe = nullptr, *d_ke = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pos, vbytes));
    CUDA_CHECK(cudaMalloc(&d_vel, vbytes));
    CUDA_CHECK(cudaMalloc(&d_F,   vbytes));
    CUDA_CHECK(cudaMalloc(&d_pe,  dbytes));
    CUDA_CHECK(cudaMalloc(&d_ke,  dbytes));

    // (2) Copy the initial positions/velocities to the device.
    CUDA_CHECK(cudaMemcpy(d_pos, sys.pos.data(), vbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, sys.vel.data(), vbytes, cudaMemcpyHostToDevice));

    const int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    std::vector<double> host_buf;     // scratch for deterministic host reductions

    GpuTimer timer;                   // measures only integration-kernel time
    float total_ms = 0.0f;

    // (3) Initial forces and total energy E0 (the conserved-quantity reference).
    timer.start();
    forces_kernel<<<blocks, THREADS_PER_BLOCK>>>(p, d_pos, d_vel, d_F, d_pe, d_ke);
    total_ms += timer.stop_ms();
    CUDA_CHECK_LAST("forces_kernel(initial)");

    MdResult r;
    r.E0 = reduce_host(d_pe, n, host_buf) + reduce_host(d_ke, n, host_buf);
    r.max_drift = 0.0;

    double kinetic = 0.0, potential = 0.0;
    // (4) The velocity-Verlet time loop, entirely on the device.
    for (int step = 0; step < p.steps; ++step) {
        timer.start();
        // (4.1) first half-kick with F(t)
        half_kick_kernel<<<blocks, THREADS_PER_BLOCK>>>(p, d_vel, d_F);
        // (4.2) drift positions with v(t+dt/2)
        drift_kernel<<<blocks, THREADS_PER_BLOCK>>>(p, d_pos, d_vel);
        // (4.3) recompute forces F(t+dt) (the O(N^2) tiled kernel)
        forces_kernel<<<blocks, THREADS_PER_BLOCK>>>(p, d_pos, d_vel, d_F, d_pe, d_ke);
        // (4.4) second half-kick with F(t+dt) -> completes v(t+dt)
        half_kick_kernel<<<blocks, THREADS_PER_BLOCK>>>(p, d_vel, d_F);
        total_ms += timer.stop_ms();
        CUDA_CHECK_LAST("verlet step");

        // Diagnostics, computed exactly as the CPU reference does at the END of a
        // step: potential from forces at the NEW positions (the d_pe the kernel
        // just wrote), and kinetic from the FULL-step velocity v(t+dt) -- i.e. the
        // velocities AFTER the second half-kick. The forces_kernel filled d_ke
        // from the pre-second-kick velocity, so for KE we read back d_vel (post
        // second kick) and sum on the host in index order (deterministic, and the
        // same order the CPU uses).
        potential = reduce_host(d_pe, n, host_buf);
        std::vector<Vec3> vtmp(static_cast<std::size_t>(n));
        CUDA_CHECK(cudaMemcpy(vtmp.data(), d_vel, vbytes, cudaMemcpyDeviceToHost));
        kinetic = 0.0;
        for (int i = 0; i < n; ++i) kinetic += kinetic_energy_one(vtmp[i], p.mass);

        const double E = kinetic + potential;
        const double drift = (E > r.E0) ? (E - r.E0) : (r.E0 - E);
        if (drift > r.max_drift) r.max_drift = drift;
    }
    *kernel_ms = total_ms;

    // (5) Final observables: read back final positions for the checksum.
    std::vector<Vec3> pos_final(static_cast<std::size_t>(n));
    CUDA_CHECK(cudaMemcpy(pos_final.data(), d_pos, vbytes, cudaMemcpyDeviceToHost));
    r.E_final = kinetic + potential;
    r.T_final = (n > 0) ? (2.0 * kinetic) / (3.0 * n) : 0.0;
    r.pos_checksum = 0.0;
    for (int i = 0; i < n; ++i)
        r.pos_checksum += pos_final[i].x + pos_final[i].y + pos_final[i].z;

    // (6) Free device memory (no GPU garbage collector).
    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_vel));
    CUDA_CHECK(cudaFree(d_F));
    CUDA_CHECK(cudaFree(d_pe));
    CUDA_CHECK(cudaFree(d_ke));
    return r;
}
