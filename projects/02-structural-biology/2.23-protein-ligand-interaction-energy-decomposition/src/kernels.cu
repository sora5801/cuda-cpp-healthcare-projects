// ===========================================================================
// src/kernels.cu  --  Per-residue MM-GBSA decomposition kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 2.23 : Protein-Ligand Interaction Energy Decomposition
//
// WHAT THIS FILE DOES
//   Implements the device kernel (decompose_kernel) and the host-side glue
//   (decompose_gpu) that allocates GPU memory, uploads the system, launches the
//   kernel, times it, and brings the [M] per-residue results back. This is the
//   GPU twin of decompose_cpu() in reference_cpu.cpp; both call the SAME shared
//   physics (residue_frame_energy() in mmgbsa.h), so main.cu can verify the GPU
//   against the CPU to a tight tolerance.
//
//   The mapping is "one thread per residue" (kernels.cuh "THE BIG IDEA"): a thread
//   owns residue m, loops over all F frames and L ligand atoms accumulating its
//   three energy components in REGISTERS, then writes one PerResidueEnergy. No
//   atomics, no shared memory -- residues are independent, so this is the
//   cleanest correct mapping (THEORY.md "GPU mapping" weighs the alternatives).
//
// READ THIS AFTER: kernels.cuh (declarations + the thread-mapping idea), mmgbsa.h
// (the per-pair physics). Compare with reference_cpu.cpp (the CPU twin).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 128 is a solid default here: it is a multiple of the
// 32-lane warp and, because each thread does a LOT of work (F*L pair evaluations
// with double-precision sqrt/exp -> high register pressure), a smaller block
// keeps enough blocks resident for occupancy without spilling registers. On a
// real protein (M in the hundreds) the grid is still many blocks. (THEORY.md
// "GPU mapping" discusses the register-pressure / occupancy trade-off.)
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// decompose_kernel: one thread per residue.
//   Launch config (set in decompose_gpu):
//     grid  = ceil(M / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: m = blockIdx.x * blockDim.x + threadIdx.x  (residue id).
//   Memory: residue/ligand params + coords come from GLOBAL memory; the running
//   sums (elec/vdw/gb) live in REGISTERS; the single output is one global write.
//   Determinism: this thread loops frames and ligand atoms in fixed index order
//   and accumulates in double precision -- exactly mirroring decompose_cpu(), so
//   the per-residue result matches the CPU to FMA-level rounding (THEORY.md).
// ---------------------------------------------------------------------------
__global__ void decompose_kernel(const ResidueParams* __restrict__ d_res,
                                 const LigandParams*  __restrict__ d_lig,
                                 const double* __restrict__ d_res_xyz,
                                 const double* __restrict__ d_lig_xyz,
                                 int F, int M, int L, double cutoff2,
                                 PerResidueEnergy* __restrict__ d_out) {
    // The residue this thread owns.
    const int m = blockIdx.x * blockDim.x + threadIdx.x;

    // GUARD THE RAGGED LAST BLOCK: M is rarely a multiple of the block size, so
    // some threads in the final block have m >= M and must do nothing (else they
    // would index out of bounds -> an illegal-address crash).
    if (m >= M) return;

    // Per-thread accumulators in registers -> no global traffic in the hot loop.
    double elec = 0.0, vdw = 0.0, gb = 0.0;
    const double inv_F = 1.0 / static_cast<double>(F);   // trajectory-average factor

    // Sum residue m's energy with the ligand over every frame. residue_frame_energy
    // (mmgbsa.h, __host__ __device__) is the SAME call the CPU reference makes, so
    // there is exactly ONE copy of the physics. It ADDS into elec/vdw/gb.
    for (int f = 0; f < F; ++f) {
        const double* res_f = d_res_xyz + static_cast<std::size_t>(f) * M * 3;  // frame f residues
        const double* lig_f = d_lig_xyz + static_cast<std::size_t>(f) * L * 3;  // frame f ligand
        residue_frame_energy(d_res, d_lig, res_f, lig_f, m, L, cutoff2, elec, vdw, gb);
    }

    // Average over frames and store the four numbers (one global write/thread).
    PerResidueEnergy e;
    e.elec  = elec * inv_F;
    e.vdw   = vdw  * inv_F;
    e.gb    = gb   * inv_F;
    e.total = e.elec + e.vdw + e.gb;
    d_out[m] = e;
}

// ---------------------------------------------------------------------------
// decompose_gpu: host wrapper. The five canonical steps of a CUDA computation:
//   (1) allocate device memory  (2) copy inputs host->device
//   (3) launch the kernel        (4) copy result device->host
//   (5) free device memory
// We time ONLY step (3) with CUDA events so the reported figure is the kernel
// cost, not the PCIe transfer cost (those are discussed separately in THEORY).
// ---------------------------------------------------------------------------
void decompose_gpu(const MmgbsaSystem& sys, std::vector<PerResidueEnergy>& out,
                   float* kernel_ms) {
    const int F = sys.F, M = sys.M, L = sys.L;
    out.assign(static_cast<std::size_t>(M), PerResidueEnergy{});

    // Byte sizes of every buffer we move to the device.
    const std::size_t res_bytes  = static_cast<std::size_t>(M) * sizeof(ResidueParams);
    const std::size_t lig_bytes  = static_cast<std::size_t>(L) * sizeof(LigandParams);
    const std::size_t rxyz_bytes = static_cast<std::size_t>(F) * M * 3 * sizeof(double);
    const std::size_t lxyz_bytes = static_cast<std::size_t>(F) * L * 3 * sizeof(double);
    const std::size_t out_bytes  = static_cast<std::size_t>(M) * sizeof(PerResidueEnergy);

    // (1) Device buffers. The d_ prefix marks DEVICE pointers (CLAUDE.md 12):
    //     dereferencing one on the host would crash, so the naming matters.
    ResidueParams* d_res = nullptr;   // [M]
    LigandParams*  d_lig = nullptr;   // [L]
    double* d_res_xyz = nullptr;      // [F*M*3]
    double* d_lig_xyz = nullptr;      // [F*L*3]
    PerResidueEnergy* d_out = nullptr;// [M]
    CUDA_CHECK(cudaMalloc(&d_res,     res_bytes));   // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_lig,     lig_bytes));
    CUDA_CHECK(cudaMalloc(&d_res_xyz, rxyz_bytes));
    CUDA_CHECK(cudaMalloc(&d_lig_xyz, lxyz_bytes));
    CUDA_CHECK(cudaMalloc(&d_out,     out_bytes));

    // (2) Copy inputs H2D. The structs are trivially copyable (POD with fixed-size
    //     members), so a flat memcpy of the vector's backing array is correct.
    CUDA_CHECK(cudaMemcpy(d_res,     sys.res.data(),     res_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lig,     sys.lig.data(),     lig_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_res_xyz, sys.res_xyz.data(), rxyz_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lig_xyz, sys.lig_xyz.data(), lxyz_bytes, cudaMemcpyHostToDevice));

    // (3) Launch. One thread per residue; blocks cover all M via ceiling division.
    const int blocks = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const double cutoff2 = sys.cutoff * sys.cutoff;
    GpuTimer timer;
    timer.start();
    decompose_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_res, d_lig, d_res_xyz, d_lig_xyz,
                                                    F, M, L, cutoff2, d_out);
    *kernel_ms = timer.stop_ms();          // GPU-measured kernel time
    CUDA_CHECK_LAST("decompose_kernel");   // catch launch + execution errors

    // (4) Bring the [M] decompositions back to the host vector.
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));

    // (5) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_lig));
    CUDA_CHECK(cudaFree(d_res_xyz));
    CUDA_CHECK(cudaFree(d_lig_xyz));
    CUDA_CHECK(cudaFree(d_out));
}
