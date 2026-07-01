// ===========================================================================
// src/kernels.cu  --  GPU TERMA ray-trace + collapsed-cone superposition kernels
// ---------------------------------------------------------------------------
// Project 5.4 : Collapsed-Cone / Superposition-Convolution Dose  (2-D teaching model)
//
// WHAT THIS FILE DOES
//   Implements the two device kernels declared in kernels.cuh and the host glue
//   (dose_gpu) that allocates GPU memory, runs both stages, times them, and
//   copies the result back. Every per-voxel formula comes from the SHARED header
//   ccc_physics.h, so these kernels compute bit-for-bit the same numbers as the
//   CPU reference (reference_cpu.cpp) -- main.cu asserts the two dose grids are
//   identical to the last integer dose-unit.
//
// READ THIS AFTER: ccc_physics.h (the physics) and kernels.cuh (the interface).
// ===========================================================================
#include "kernels.cuh"
#include "ccc_physics.h"         // CccParams + terma_at / cone_* / dose_to_units
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// 256 threads/block: a multiple of the 32-lane warp, 8 warps to hide latency,
// and it leaves plenty of resident blocks for occupancy on sm_75..sm_89.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// terma_kernel  --  STAGE 1: ray-trace TERMA down one beam column per thread.
//   Launch (set in dose_gpu):
//     grid  = ceil(beam_width / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: thread t owns beam column x = beam_x0 + t. Threads with
//   x > beam_x1 are outside the beam and return (the ragged-last-block guard).
//   Memory: reads rho[y*nx + x] top-to-bottom, writes terma[y*nx + x]. Each
//   column is disjoint, so NO atomics and NO shared memory are needed -- this is
//   a pure independent-1D-scan-per-thread (the same shape as the CPU's outer
//   column loop, just one column per thread instead of a serial sweep).
//
//   This is the axis-aligned specialization of Siddon's ray tracer: a vertical
//   ray crosses every voxel with intersection length voxel_cm, so the depth
//   integral is a simple running sum. THEORY.md derives the oblique version.
// ---------------------------------------------------------------------------
__global__ void terma_kernel(CccParams P,
                             const float* __restrict__ rho,
                             double* __restrict__ terma) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;  // 0-based within the beam
    const int x = P.beam_x0 + t;                          // this thread's column
    if (x > P.beam_x1) return;                            // guard: past the beam edge

    double rad_above = 0.0;                               // radiological depth to voxel TOP
    for (int y = 0; y < P.ny; ++y) {
        const double rho_here  = rho[static_cast<size_t>(y) * P.nx + x];
        // Depth to the voxel CENTER (matches the CPU convention exactly).
        const double rad_center = rad_above + 0.5 * rho_here * P.voxel_cm;
        terma[static_cast<size_t>(y) * P.nx + x] = terma_at(P, rad_center);
        rad_above += rho_here * P.voxel_cm;               // advance a full cell
    }
}

// ---------------------------------------------------------------------------
// ccc_kernel  --  STAGE 2: one thread per SOURCE voxel; scatter its collapsed-
//   cone dose into the grid.
//   Launch (set in dose_gpu):
//     grid  = ceil(nx*ny / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: thread s owns source voxel (sx,sy) with s = sy*nx + sx.
//   Memory: reads terma[s] (its own release) and rho along each cone ray; writes
//   dose_units[dst] with atomicAdd because many source voxels' cone rays overlap
//   in the same destination voxel.
//
//   WHY INTEGER ATOMICS (the load-bearing determinism trick, PATTERNS.md §3):
//     Threads finish in a nondeterministic order, so a FLOAT atomicAdd would sum
//     the same contributions in different orders each run -> different rounding ->
//     a grid that changes run to run and never exactly equals the CPU's. We instead
//     quantize each deposit to an INTEGER dose-unit (dose_to_units) and atomicAdd
//     that. Integer addition is associative/commutative, so the tally is exact,
//     reproducible, and identical to the serial CPU reference.
//
//   The per-thread recurrence is byte-identical to spread_one_source() on the CPU:
//     carry starts as T_s/n_cones; each step deposits carry*(1-transmit) and keeps
//     the rest. transmit = exp(-a * step_cm * rho) reaches farther in low density.
//
//   NOTE ON ATOMIC TYPE: CUDA's 64-bit atomicAdd takes `unsigned long long*`. All
//   our deposits are non-negative, so we accumulate into an unsigned tally whose
//   bit pattern equals the signed CPU tally for these values; dose_gpu copies it
//   straight into the host's long long grid.
// ---------------------------------------------------------------------------
__global__ void ccc_kernel(CccParams P,
                          const float* __restrict__ rho,
                          const double* __restrict__ terma,
                          long long* __restrict__ dose_units) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;  // source-voxel index
    const int total = P.nx * P.ny;
    if (s >= total) return;                               // ragged-last-block guard

    const double T_s = terma[s];
    if (T_s <= 0.0) return;                               // nothing released here

    const int sx = s % P.nx;                              // decode 1-D -> 2-D
    const int sy = s / P.nx;
    const double w = cone_weight(P);                      // 1/n_cones share per cone

    // Reinterpret the signed grid as unsigned for the 64-bit atomicAdd. Safe here
    // because every accumulated value is non-negative (see the note above).
    unsigned long long* udose = reinterpret_cast<unsigned long long*>(dose_units);

    for (int c = 0; c < P.n_cones; ++c) {
        const int dx = ccc_cone_dx(c);
        const int dy = ccc_cone_dy(c);
        const double step_cm = ccc_step_cm(c, P.voxel_cm);
        double carry = T_s * w;                           // dose entering this cone ray

        int x = sx, y = sy;
        for (int k = 0; k < P.nx + P.ny; ++k) {           // bounded by longest ray
            x += dx; y += dy;
            if (x < 0 || x >= P.nx || y < 0 || y >= P.ny) break;   // left the grid
            const double rho_here = rho[static_cast<size_t>(y) * P.nx + x];
            const double transmit = cone_transmit(P, step_cm, rho_here);
            const double deposit  = carry * (1.0 - transmit);
            carry -= deposit;                             // == carry * transmit
            const long long units = dose_to_units(P, deposit);
            if (units != 0)
                atomicAdd(&udose[static_cast<size_t>(y) * P.nx + x],
                          static_cast<unsigned long long>(units));
            if (carry * P.dose_scale < 0.5) break;        // < half a unit left
        }
    }
}

// ---------------------------------------------------------------------------
// dose_gpu  --  host wrapper: run both stages, time them, copy results back.
//   The canonical CUDA choreography:
//     (1) allocate device buffers  (2) copy the density map H2D + zero the tally
//     (3) launch stage 1 (TERMA)   (4) launch stage 2 (CCC, depends on stage 1)
//     (5) copy TERMA + dose grid D2H   (6) free device memory
//   We time steps (3)+(4) with CUDA events so the reported figure is kernel cost,
//   not the PCIe transfers (discussed separately in THEORY.md).
// ---------------------------------------------------------------------------
void dose_gpu(const DoseProblem& prob,
              std::vector<long long>& dose_units,
              std::vector<double>& terma_out,
              float* kernel_ms) {
    const CccParams& P = prob.P;
    const int total = P.nx * P.ny;
    const size_t nbytes_f  = static_cast<size_t>(total) * sizeof(float);
    const size_t nbytes_d  = static_cast<size_t>(total) * sizeof(double);
    const size_t nbytes_ll = static_cast<size_t>(total) * sizeof(long long);

    dose_units.assign(total, 0LL);
    terma_out.assign(total, 0.0);

    // (1) Device buffers. d_ prefix = DEVICE pointer (dereferencing on the host
    //     would crash), per the naming convention in CLAUDE.md §12.
    float*     d_rho   = nullptr;   // density map
    double*    d_terma = nullptr;   // stage-1 TERMA output / stage-2 input
    long long* d_dose  = nullptr;   // integer dose tally
    CUDA_CHECK(cudaMalloc(&d_rho,   nbytes_f));
    CUDA_CHECK(cudaMalloc(&d_terma, nbytes_d));
    CUDA_CHECK(cudaMalloc(&d_dose,  nbytes_ll));

    // (2) Upload density; zero the TERMA (columns outside the beam stay 0) and the
    //     dose tally (atomicAdd accumulates onto zero).
    CUDA_CHECK(cudaMemcpy(d_rho, prob.rho.data(), nbytes_f, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_terma, 0, nbytes_d));
    CUDA_CHECK(cudaMemset(d_dose,  0, nbytes_ll));

    GpuTimer timer;
    timer.start();

    // (3) STAGE 1: one thread per beam column.
    const int beam_width = P.beam_x1 - P.beam_x0 + 1;
    const int terma_blocks = (beam_width + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    terma_kernel<<<terma_blocks, THREADS_PER_BLOCK>>>(P, d_rho, d_terma);
    CUDA_CHECK_LAST("terma_kernel");

    // (4) STAGE 2: one thread per source voxel. This reads the stage-1 TERMA, so
    //     it must run AFTER stage 1 -- both use the default stream, which runs
    //     kernels in order, so the dependency is respected automatically.
    const int ccc_blocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    ccc_kernel<<<ccc_blocks, THREADS_PER_BLOCK>>>(P, d_rho, d_terma, d_dose);
    CUDA_CHECK_LAST("ccc_kernel");

    *kernel_ms = timer.stop_ms();   // total time for both kernels (GPU-measured)

    // (5) Copy both outputs back to the host.
    CUDA_CHECK(cudaMemcpy(terma_out.data(),  d_terma, nbytes_d,  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dose_units.data(), d_dose,  nbytes_ll, cudaMemcpyDeviceToHost));

    // (6) Free everything (no GPU garbage collector).
    CUDA_CHECK(cudaFree(d_rho));
    CUDA_CHECK(cudaFree(d_terma));
    CUDA_CHECK(cudaFree(d_dose));
}
