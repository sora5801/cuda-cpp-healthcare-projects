// ===========================================================================
// src/kernels.cu  --  GIST grid-accumulation kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 2.26 : Hydrogen Bond Network & Water Placement Analysis
//
// GPU twin of gist_cpu(): one thread per (water, frame) sample, each thread finds
// its voxel and ATOMICALLY adds its occupancy + fixed-point energy into that
// voxel's tally. The fixed-point integers make the atomic adds commute, so the
// GPU tally equals the serial CPU tally exactly (PATTERNS.md §3). After the
// kernel, the SHARED host helper derive_voxels() produces the ranked list, so the
// CPU and GPU output identically. See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// 256 threads/block: a multiple of the 32-lane warp, 8 warps to hide the global
// loads of the atom list, and enough resident blocks for good occupancy on
// sm_75..sm_89. The work is atomic-bound, not compute-bound, so this is a safe
// default (THEORY.md discusses atomic contention).
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// device_atomic_add_fixed: atomicAdd into a SIGNED 64-bit fixed-point cell.
//   CUDA exposes a native atomicAdd for `unsigned long long` but not for the
//   signed `long long` we use for energy (which can be negative). Two's-complement
//   addition is bit-identical for signed and unsigned, so we reinterpret the cell
//   and the addend as unsigned, add atomically, and the stored bit pattern is the
//   correct signed sum. This is the standard idiom for signed integer atomics.
// ---------------------------------------------------------------------------
__device__ inline void device_atomic_add_fixed(gist_fixed_t* addr, gist_fixed_t val) {
    atomicAdd(reinterpret_cast<unsigned long long*>(addr),
              static_cast<unsigned long long>(val));
}

// ---------------------------------------------------------------------------
// gist_accumulate_kernel: the scatter. One thread per (water, frame) sample.
//   Thread t -> sample t; sample t lives at waters[t*3 .. t*3+2]. We:
//     (1) read the water position,
//     (2) map it to a voxel (gist_voxel_of, shared with the CPU) -- skip if it
//         falls outside the analysis box (v < 0),
//     (3) compute its water<->solute energy (gist_water_solute_energy, shared),
//     (4) atomicAdd 1 into counts[v] and the fixed-point energy into esum[v].
//   Every step (2)-(3) calls the SAME __host__ __device__ code the CPU runs, so
//   the only difference between this kernel and gist_cpu() is the parallel loop.
// ---------------------------------------------------------------------------
__global__ void gist_accumulate_kernel(const float* __restrict__ waters,
                                       long long num_samples,
                                       const float* __restrict__ atoms, int natoms,
                                       GistGrid grid,
                                       unsigned int* __restrict__ counts,
                                       gist_fixed_t* __restrict__ esum) {
    // Global sample index. We use a 64-bit index because num_samples (frames x
    // waters) can exceed 2^31 in a real run; the guard drops the ragged tail.
    const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= num_samples) return;

    // (1) This sample's water-oxygen position.
    const std::size_t base = static_cast<std::size_t>(t) * 3;
    const double wx = waters[base + 0];
    const double wy = waters[base + 1];
    const double wz = waters[base + 2];

    // (2) Which voxel does it occupy? -1 => outside the grid; not scored.
    const int v = gist_voxel_of(grid, wx, wy, wz);
    if (v < 0) return;

    // (3) Water<->solute interaction energy (kcal/mol), then quantize.
    const double e = gist_water_solute_energy(wx, wy, wz, atoms, natoms);

    // (4) Scatter into the voxel's tallies. Many threads share a voxel -> atomics.
    atomicAdd(&counts[v], 1u);                          // occupancy (unsigned int)
    device_atomic_add_fixed(&esum[v], gist_to_fixed(e));// energy (fixed-point i64)
}

// ---------------------------------------------------------------------------
// gist_gpu: host wrapper -- the five canonical CUDA steps plus the shared reduce.
//   (1) allocate device buffers   (2) copy waters + atoms H2D, zero the tallies
//   (3) launch the kernel (timed) (4) copy tallies D2H   (5) free device memory
//   Then derive_voxels() on the host builds the ranked list (identical to CPU).
//   We time ONLY the kernel (step 3) with CUDA events; transfer cost is discussed
//   separately in THEORY.md, never folded into the "kernel time" figure.
// ---------------------------------------------------------------------------
std::vector<VoxelResult> gist_gpu(const Dataset& d,
                                  std::vector<unsigned int>& counts,
                                  std::vector<gist_fixed_t>& esum,
                                  float* kernel_ms) {
    const int nv = d.grid.num_voxels();
    const long long ns = d.num_samples();

    // (1) Device buffers: waters (3 floats/sample), atoms (4 floats/atom), and the
    //     two voxel tally arrays. d_ marks DEVICE pointers (never deref on host).
    float* d_waters = nullptr;
    float* d_atoms  = nullptr;
    unsigned int* d_counts = nullptr;
    gist_fixed_t* d_esum   = nullptr;
    CUDA_CHECK(cudaMalloc(&d_waters, d.waters.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_atoms,  d.atoms.size()  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_counts, static_cast<std::size_t>(nv) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_esum,   static_cast<std::size_t>(nv) * sizeof(gist_fixed_t)));

    // (2) Copy inputs H2D; ZERO the tallies on the device (the kernel only adds).
    CUDA_CHECK(cudaMemcpy(d_waters, d.waters.data(), d.waters.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_atoms, d.atoms.data(), d.atoms.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_counts, 0, static_cast<std::size_t>(nv) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_esum, 0, static_cast<std::size_t>(nv) * sizeof(gist_fixed_t)));

    // (3) Launch one thread per sample, covering all ns with a ceiling division.
    const long long blocks_ll = (ns + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const unsigned int blocks = static_cast<unsigned int>(blocks_ll);
    GpuTimer timer;
    timer.start();
    gist_accumulate_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_waters, ns, d_atoms, d.natoms, d.grid, d_counts, d_esum);
    *kernel_ms = timer.stop_ms();           // GPU-measured kernel time
    CUDA_CHECK_LAST("gist_accumulate_kernel");  // catch launch + execution errors

    // (4) Bring the raw voxel tallies back to the host.
    counts.assign(static_cast<std::size_t>(nv), 0u);
    esum.assign(static_cast<std::size_t>(nv), 0);
    CUDA_CHECK(cudaMemcpy(counts.data(), d_counts,
                          static_cast<std::size_t>(nv) * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(esum.data(), d_esum,
                          static_cast<std::size_t>(nv) * sizeof(gist_fixed_t),
                          cudaMemcpyDeviceToHost));

    // (5) Free device memory (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_waters));
    CUDA_CHECK(cudaFree(d_atoms));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_esum));

    // Shared reduction: identical ranked list to the CPU path (derive_voxels lives
    // in reference_cpu.cpp and is reused here for an exact match).
    return derive_voxels(d, counts, esum);
}
