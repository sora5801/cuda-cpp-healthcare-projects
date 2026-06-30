// ===========================================================================
// src/kernels.cu  --  GPU Debye-scattering kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 2.24 : SAXS / SANS Data-Driven Structure Modeling
//
// WHAT THIS FILE DOES
//   Implements debye_kernel (one GPU thread computes I(q) for one q value, by
//   summing over all atom pairs) and debye_gpu (the host glue: allocate, copy,
//   launch, time, copy back). It is the GPU twin of debye_profile_cpu() in
//   reference_cpu.cpp; both call the SAME per-q physics in saxs_core.h, so the
//   GPU and CPU intensities agree to floating-point rounding (verified in main.cu).
//
// WHY THIS MAPPING
//   Each q value's intensity is a fully independent O(n_atoms^2) reduction with
//   no shared output -> the cleanest possible parallelism: thread k owns q[k],
//   does its entire reduction in a register, and writes one number. No atomics,
//   no inter-thread sync -> deterministic and exactly comparable to the CPU.
//
//   A faster variant TILES the atom arrays through shared memory so the inner
//   loop reads on-chip memory; we keep that as a documented exercise (THEORY.md
//   §GPU mapping) because it reorders the pair sum and would only match the CPU
//   to a looser tolerance. For a teaching baseline we favor exact parity here.
//
// READ THIS AFTER: kernels.cuh and saxs_core.h.
// ===========================================================================
#include "kernels.cuh"
#include "saxs_core.h"           // debye_intensity_at_q -- shared CPU/GPU physics
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 128 is a good fit here: each thread does a heavy O(N^2)
// register-resident reduction, so we want enough warps to hide global-memory
// latency on the atom reads, but not so many that register pressure cuts
// occupancy. 128 = 4 warps/block is a balanced default on sm_75..sm_89.
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// debye_kernel: one thread computes I(q[k]) for its own k.
//   Launch config (set in debye_gpu):
//     grid  = ceil(n_q / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: k = blockIdx.x * blockDim.x + threadIdx.x  (a q index).
//   Memory: reads the atom arrays (x,y,z,f) and q[k] from GLOBAL memory; writes
//           one I_model[k]. No shared memory, no atomics -- the whole pair-sum
//           reduction for q[k] lives in this thread's registers.
//
//   Parameters (all device pointers are __restrict__: the compiler may assume
//   they do not alias, so it can keep the atom reads in registers across the loop):
//     x,y,z : [n_atoms] coordinates (Å)
//     f     : [n_atoms] scattering strengths (electron count)
//     n_atoms : number of atoms
//     q     : [n_q] momentum-transfer grid (1/Å)
//     n_q   : number of q values (guards the ragged last block)
//     I_out : [n_q] output intensities
// ---------------------------------------------------------------------------
__global__ void debye_kernel(const double* __restrict__ x,
                             const double* __restrict__ y,
                             const double* __restrict__ z,
                             const double* __restrict__ f,
                             int n_atoms,
                             const double* __restrict__ q,
                             int n_q,
                             double* __restrict__ I_out) {
    // This thread's q index.
    const int k = blockIdx.x * blockDim.x + threadIdx.x;

    // GUARD THE RAGGED LAST BLOCK: n_q is rarely a multiple of the block size,
    // so the final block has threads with k >= n_q. They must do nothing or they
    // would read q[k] / write I_out[k] out of bounds (an illegal-address fault).
    if (k >= n_q) return;

    // The entire Debye double sum for q[k], computed by the SHARED physics in
    // saxs_core.h -- byte-for-byte the same call the CPU reference makes. That
    // shared-core idiom (PATTERNS.md §2) is what makes GPU==CPU verification
    // exact-to-rounding instead of approximate.
    I_out[k] = debye_intensity_at_q(q[k], x, y, z, f, n_atoms);
}

// ---------------------------------------------------------------------------
// debye_gpu: host wrapper. The five canonical steps of a CUDA computation:
//   (1) allocate device memory  (2) copy inputs host->device
//   (3) launch the kernel        (4) copy result device->host
//   (5) free device memory
// We time ONLY step (3) with CUDA events so the reported figure is kernel cost,
// not the PCIe transfer cost (THEORY.md discusses the transfer separately).
// ---------------------------------------------------------------------------
void debye_gpu(const SaxsModel& m, std::vector<double>& I_model, float* kernel_ms) {
    const int n_atoms = m.n_atoms;
    const int n_q     = m.n_q;
    I_model.assign(static_cast<std::size_t>(n_q), 0.0);

    const std::size_t atom_bytes = static_cast<std::size_t>(n_atoms) * sizeof(double);
    const std::size_t q_bytes    = static_cast<std::size_t>(n_q)     * sizeof(double);

    // (1) Device buffers. The d_ prefix marks DEVICE pointers (CLAUDE.md §12):
    //     dereferencing one on the host would crash, so the naming matters.
    double *d_x = nullptr, *d_y = nullptr, *d_z = nullptr, *d_f = nullptr;
    double *d_q = nullptr, *d_I = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, atom_bytes));   // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_y, atom_bytes));
    CUDA_CHECK(cudaMalloc(&d_z, atom_bytes));
    CUDA_CHECK(cudaMalloc(&d_f, atom_bytes));
    CUDA_CHECK(cudaMalloc(&d_q, q_bytes));
    CUDA_CHECK(cudaMalloc(&d_I, q_bytes));

    // (2) Copy inputs H2D. .data() is the contiguous backing array of a vector.
    CUDA_CHECK(cudaMemcpy(d_x, m.x.data(), atom_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, m.y.data(), atom_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, m.z.data(), atom_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f, m.f.data(), atom_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, m.q.data(), q_bytes,    cudaMemcpyHostToDevice));

    // (3) Launch. Blocks must cover all n_q outputs -> ceiling division.
    const int blocks = (n_q + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    debye_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_x, d_y, d_z, d_f, n_atoms,
                                                d_q, n_q, d_I);
    *kernel_ms = timer.stop_ms();          // GPU-measured kernel time
    CUDA_CHECK_LAST("debye_kernel");       // catch launch + execution errors

    // (4) Bring the n_q intensities back to the host vector.
    CUDA_CHECK(cudaMemcpy(I_model.data(), d_I, q_bytes, cudaMemcpyDeviceToHost));

    // (5) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_f));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_I));
}
