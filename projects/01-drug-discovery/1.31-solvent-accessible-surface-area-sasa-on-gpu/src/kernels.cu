// ===========================================================================
// src/kernels.cu  --  Shrake-Rupley SASA kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 1.31 : Solvent-Accessible Surface Area (SASA) on GPU
//
// This is the GPU twin of sasa_cpu() in reference_cpu.cpp. main.cu runs both and
// asserts the integer exposed-point counts match EXACTLY. The per-atom math is
// the shared __host__ __device__ code in sasa_core.h, so "the same arithmetic"
// is not a hope -- it is literally the same source compiled for both targets.
//
// TWO VIEWS OF THE INNER LOOP (a teaching contrast):
//   * The CPU reference calls count_exposed_points(), whose point_is_buried()
//     streams every neighbor straight from GLOBAL memory once per test point.
//   * This kernel makes the SAME burial decision but staging neighbors through
//     SHARED MEMORY in tiles -- the classic all-pairs / N-body optimization.
//     Each block cooperatively loads TILE atoms into fast on-chip memory, and all
//     128 threads of the block reuse that one tile. That is the real win: the
//     expensive global load of a neighbor is shared by the whole block instead of
//     repeated by every thread. The boolean reached for each test point is
//     identical to the CPU's ("buried by ANY neighbor" is an OR, order-
//     independent), so the exposed COUNT is bit-for-bit equal. See THEORY
//     "GPU mapping" and "How we verify".
// ===========================================================================
#include "kernels.cuh"
#include "sasa_core.h"           // fib_point, point burial math, constants
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer

// 128 threads/block: a multiple of the 32-lane warp with good occupancy on
// sm_75..sm_89, and it doubles as the shared-memory TILE size (one cached atom
// per thread). 128 Atoms * 32 bytes = 4 KB of shared memory per block -- tiny, so
// occupancy is not shared-memory-bound. See THEORY "GPU mapping" for the sizing.
static constexpr int THREADS_PER_BLOCK = 128;
static constexpr int TILE              = THREADS_PER_BLOCK;

// ---------------------------------------------------------------------------
// sasa_kernel: one thread per atom (grid-stride). For its atom i, the thread
//   counts how many of the N_SPHERE_POINTS test points on atom i's inflated
//   surface are NOT buried inside a neighbor, testing each point against every
//   other atom via shared-memory tiles.
//
//   Thread (blockIdx.x, threadIdx.x) starts at i = block*blockDim + thread and
//   strides by the total thread count, so a capped grid still covers any n.
//   Memory: d_atoms staged into __shared__ s_atoms; per-point work in registers;
//   one int written per atom (no atomics -- outputs are fully independent).
//
//   PARITY: the burial decision uses the SAME fib_point() directions and the SAME
//   strict squared-distance test (`d2 < rj*rj`) as the CPU's point_is_buried(),
//   so the per-atom exposed counts match the reference exactly.
//
//   Loop order: points OUTER, neighbor-tiles INNER. The tile is reused across all
//   threads of the block (cross-thread reuse), which is where the bandwidth win
//   comes from; per-point register state stays small (no 96-wide local arrays).
// ---------------------------------------------------------------------------
__global__ void sasa_kernel(const Atom* __restrict__ d_atoms, int n, double probe,
                            int* __restrict__ exposed_out) {
    // Block-local cache of one TILE of neighbor atoms, refilled as we sweep the
    // array. Shared memory is ~100x faster than global and reused by all threads.
    __shared__ Atom s_atoms[TILE];

    const int stride = blockDim.x * gridDim.x;             // total threads in grid

    // grid-stride over atoms. NOTE: every thread of the block must reach the
    // __syncthreads() inside the tile loop the same number of times, so the atom
    // loop bound must be uniform across the block. We therefore round the loop
    // limit up to a multiple of `stride`: out-of-range threads still participate
    // in the cooperative tile loads (so the barrier is well-formed) but compute
    // nothing. (A __syncthreads() that only some threads reach is undefined.)
    const int i_end = ((n + stride - 1) / stride) * stride;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < i_end; i += stride) {
        const bool active = (i < n);                       // does this thread own a real atom?
        Atom self{};                                       // value-init keeps it benign if inactive
        double surf_r = 0.0;
        if (active) { self = d_atoms[i]; surf_r = self.r + probe; }

        int exposed = 0;                                   // exposed points for atom i (register)

        // ---- OUTER loop: each of atom i's test points -----------------------
        for (int k = 0; k < N_SPHERE_POINTS; ++k) {
            // Direction of test point k on the unit sphere (shared generator).
            double ux, uy, uz;
            fib_point(k, N_SPHERE_POINTS, ux, uy, uz);
            // Absolute coordinates of the test point on atom i's inflated surface.
            const double pxk = self.x + surf_r * ux;
            const double pyk = self.y + surf_r * uy;
            const double pzk = self.z + surf_r * uz;
            bool buried = false;                            // innocent until covered

            // ---- INNER loop: sweep neighbors in shared-memory tiles ---------
            for (int base = 0; base < n; base += TILE) {
                // Cooperative load: thread t brings neighbor (base+t) into the
                // tile. The whole block loads TILE atoms in one coalesced burst.
                if (base + threadIdx.x < n)
                    s_atoms[threadIdx.x] = d_atoms[base + threadIdx.x];
                __syncthreads();                            // tile is now filled & visible

                // How many of the cached slots are valid (last tile may be short).
                const int tile_count = (n - base < TILE) ? (n - base) : TILE;
                // Only an ACTIVE, not-yet-buried thread needs to test; others just
                // ride along so the next __syncthreads() is block-wide.
                if (active && !buried) {
                    for (int t = 0; t < tile_count; ++t) {
                        const int j = base + t;             // global neighbor index
                        if (j == i) continue;               // never buried by own atom
                        const Atom nb = s_atoms[t];         // neighbor from shared cache
                        const double rj = nb.r + probe;     // neighbor's inflated radius
                        const double dx = pxk - nb.x;
                        const double dy = pyk - nb.y;
                        const double dz = pzk - nb.z;
                        const double d2 = dx * dx + dy * dy + dz * dz;
                        if (d2 < rj * rj) { buried = true; break; }  // SAME strict `<` as CPU
                    }
                }
                __syncthreads();                            // all done with tile before refilling
            }

            if (active && !buried) ++exposed;               // survived every neighbor -> exposed
        }

        if (active) exposed_out[i] = exposed;               // the exact integer the CPU also gets
    }
}

// ---------------------------------------------------------------------------
// sasa_gpu: the five canonical CUDA steps. We time ONLY the kernel (CUDA events),
//   not the H2D/D2H copies (those overheads are discussed in THEORY). The integer
//   counts come back from the device; the per-atom AREA is derived on the host
//   with the shared atom_sasa() so it matches the CPU reference to the last ULP.
// ---------------------------------------------------------------------------
void sasa_gpu(const Molecule& mol,
              std::vector<int>& exposed,
              std::vector<double>& sasa,
              float* kernel_ms) {
    const int n = mol.n;
    exposed.assign(static_cast<std::size_t>(n), 0);
    sasa.assign(static_cast<std::size_t>(n), 0.0);

    const std::size_t atoms_bytes  = static_cast<std::size_t>(n) * sizeof(Atom);
    const std::size_t counts_bytes = static_cast<std::size_t>(n) * sizeof(int);

    // (a) Allocate device buffers and upload the atoms in ONE contiguous copy --
    //     Atom is POD and the host/device layouts are identical, so no repacking.
    Atom* d_atoms = nullptr;   // [n] device atoms
    int*  d_exp   = nullptr;   // [n] device exposed-point counts
    CUDA_CHECK(cudaMalloc(&d_atoms, atoms_bytes));
    CUDA_CHECK(cudaMalloc(&d_exp,   counts_bytes));
    CUDA_CHECK(cudaMemcpy(d_atoms, mol.atoms.data(), atoms_bytes, cudaMemcpyHostToDevice));

    // (b) Launch: enough blocks for one-thread-per-atom, capped so the grid stays
    //     modest; the grid-stride loop in the kernel handles any larger n.
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks > 1024) blocks = 1024;
    GpuTimer timer;
    timer.start();
    sasa_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_atoms, n, PROBE_RADIUS, d_exp);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("sasa_kernel");

    // (c) Copy the integer counts back to the host.
    CUDA_CHECK(cudaMemcpy(exposed.data(), d_exp, counts_bytes, cudaMemcpyDeviceToHost));

    // (d) Derive per-atom areas on the host with the SHARED atom_sasa() so the
    //     float result is identical to the CPU reference's (same fn, same inputs).
    for (int i = 0; i < n; ++i) {
        const double surf_r = mol.atoms[static_cast<std::size_t>(i)].r + PROBE_RADIUS;
        sasa[static_cast<std::size_t>(i)] =
            atom_sasa(exposed[static_cast<std::size_t>(i)], surf_r);
    }

    // (e) Free device memory (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_atoms));
    CUDA_CHECK(cudaFree(d_exp));
}
