// ===========================================================================
// src/kernels.cu  --  GPU pencil-beam dose kernel (per-voxel gather over spots)
// ---------------------------------------------------------------------------
// Project 5.3 : Proton & Heavy-Ion Therapy Dose
//
// GPU twin of dose_cpu(): each thread owns ONE voxel and sums that voxel's dose
// over every spot, calling the SAME dose_from_spot() from proton_physics.h that
// the CPU reference calls. Spots live in constant memory (read by all threads,
// never written -> broadcast cache). No atomics: distinct voxels never collide.
// main.cu runs both and asserts agreement within tolerance. See ../THEORY.md.
// ===========================================================================
#include "kernels.cuh"
#include "proton_physics.h"        // Spot, Grid, BeamModel, dose_from_spot
#include "util/cuda_check.cuh"     // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"          // GpuTimer

#include <stdexcept>

// A good general-purpose block size on sm_75..sm_89: 256 threads = 8 warps,
// enough to hide arithmetic/memory latency while keeping many blocks resident.
static constexpr int THREADS_PER_BLOCK = 256;

// Upper bound on spots we can hold in constant memory. Constant memory is only
// 64 KB total; each Spot is 4 floats = 16 bytes, so 4096 spots = 64 KB is the
// hard ceiling. We cap at 2048 spots (32 KB) to leave headroom and keep the demo
// comfortably in range. A real plan has ~1e4 spots, which would instead be
// STREAMED from global memory in tiles (THEORY.md §real world).
static constexpr int MAX_CONST_SPOTS = 2048;

// ---------------------------------------------------------------------------
// c_spots: the spot list in CONSTANT memory. Every thread reads the whole list
// each launch but no thread writes it, so constant memory's broadcast cache can
// serve the same address to a whole warp in one transaction -- far cheaper than
// each thread pulling spots from global memory. This is the same "read-only data
// shared by all threads" trick as the query fingerprint in flagship 1.12.
// ---------------------------------------------------------------------------
__constant__ Spot c_spots[MAX_CONST_SPOTS];

// ---------------------------------------------------------------------------
// dose_kernel: compute the dose at one voxel by summing over all spots.
//   grid  : enough blocks to cover n_voxels (grid-stride loop handles any count)
//   block : THREADS_PER_BLOCK threads
//   thread global index -> voxel linear index `idx` (x fastest, then y, then z)
//
//   Each thread:
//     1. decodes its linear voxel index into (i,j,k),
//     2. computes that voxel's centre in world coordinates,
//     3. loops over the n_spots constant-memory spots, accumulating dose into a
//        PRIVATE register `acc` (no shared writes, no atomics -> deterministic),
//     4. writes the single result to global memory (coalesced: adjacent threads
//        have adjacent `i` -> adjacent addresses).
//
//   All the physics is in dose_from_spot() (proton_physics.h), identical to the
//   CPU path -- that shared core is what makes CPU and GPU agree tightly.
// ---------------------------------------------------------------------------
__global__ void dose_kernel(Grid g, BeamModel beam, float z_entry,
                            int n_spots, float* __restrict__ dose) {
    const std::size_t n_voxels =
        static_cast<std::size_t>(g.nx) * g.ny * g.nz;
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;   // total threads in grid

    for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < n_voxels; idx += stride) {
        // --- 1. decode linear index -> (i,j,k) --------------------------------
        // idx = (k*ny + j)*nx + i, so peel off i, then j, then k.
        const int i = static_cast<int>(idx % g.nx);
        const std::size_t rem = idx / g.nx;                 // = k*ny + j
        const int j = static_cast<int>(rem % g.ny);
        const int k = static_cast<int>(rem / g.ny);

        // --- 2. voxel-centre world coordinates -------------------------------
        const float vx = g.ox + (static_cast<float>(i) + 0.5f) * g.dx;
        const float vy = g.oy + (static_cast<float>(j) + 0.5f) * g.dx;
        const float vz = g.oz + (static_cast<float>(k) + 0.5f) * g.dx;

        // --- 3. gather over spots into a private register --------------------
        // Same spot order as the CPU reference -> identical FP32 summation.
        float acc = 0.0f;
        for (int s = 0; s < n_spots; ++s)
            acc += dose_from_spot(beam, c_spots[s], vx, vy, vz, z_entry);

        // --- 4. one coalesced write per thread -------------------------------
        dose[idx] = acc;
    }
}

// ---------------------------------------------------------------------------
// dose_gpu: host wrapper. Uploads spots to constant memory, allocates the dose
// volume on the device, launches the kernel (timed with CUDA events), and copies
// the dose back. All CUDA error checking goes through CUDA_CHECK so failures are
// loud and located (util/cuda_check.cuh). Mirrors the CPU path in main.cu.
// ---------------------------------------------------------------------------
void dose_gpu(const Plan& plan, std::vector<float>& dose, float* kernel_ms) {
    const Grid& g = plan.grid;
    const int n_spots = static_cast<int>(plan.spots.size());
    if (n_spots > MAX_CONST_SPOTS)
        throw std::runtime_error("dose_gpu: too many spots for constant memory "
                                 "(cap is 2048; stream from global memory instead)");

    const std::size_t n_voxels = voxel_count(g);
    dose.assign(n_voxels, 0.0f);

    // Copy the spot list into constant memory. cudaMemcpyToSymbol writes to the
    // named __constant__ array; it is a host->device copy into a special, cached,
    // read-only region. Done ONCE before the launch (the spots never change).
    CUDA_CHECK(cudaMemcpyToSymbol(c_spots, plan.spots.data(),
                                  static_cast<std::size_t>(n_spots) * sizeof(Spot)));

    // Device buffer for the dose volume (one float per voxel).
    float* d_dose = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dose, n_voxels * sizeof(float)));

    // Launch geometry: enough blocks to give the GPU many resident warps; the
    // grid-stride loop inside the kernel covers all voxels with this fixed grid.
    // Cap the block count so a huge volume never requests an absurd grid.
    const int max_blocks = 65535;
    std::size_t want = (n_voxels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int blocks = (want > static_cast<std::size_t>(max_blocks))
                     ? max_blocks
                     : static_cast<int>(want);
    if (blocks < 1) blocks = 1;

    GpuTimer timer;
    timer.start();
    dose_kernel<<<blocks, THREADS_PER_BLOCK>>>(g, plan.beam, plan.z_entry, n_spots, d_dose);
    *kernel_ms = timer.stop_ms();     // blocks until the kernel finishes
    CUDA_CHECK_LAST("dose_kernel");   // catch launch-config AND in-kernel errors

    // Copy the finished dose volume back to the host for verification/reporting.
    CUDA_CHECK(cudaMemcpy(dose.data(), d_dose, n_voxels * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_dose));
}
