// ===========================================================================
// src/kernels.cu  --  PK/PD population kernel (one thread per patient)
// ---------------------------------------------------------------------------
// Project 6.15 : PK/PD & PBPK Modeling
//
// GPU twin of integrate_cpu(): each thread runs the SAME coupled PK/PD RK4 loop
// (pkpd.h) for one virtual patient and writes one PatientResult. main.cu compares
// the per-patient results against the CPU reference. See ../THEORY.md §4 for the
// GPU mapping and why this ensemble is the GPU's sweet spot (compute-bound, no
// inter-thread traffic).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 128 is a solid default for a register-heavy per-thread ODE
// integrator on sm_75..sm_89: a multiple of the 32-lane warp, enough warps to
// hide latency, and small enough that the per-thread register footprint of the
// RK4 loop does not throttle occupancy. (Tune per GPU; see THEORY §4.)
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// pkpd_kernel: thread idx owns patient idx.
//   Steps done entirely in registers (no global memory during integration):
//     sample this patient's physiology -> integrate the coupled PK/PD ODE with
//     RK4 -> write the one PatientResult. The heavy lifting is pkpd_integrate()
//     from pkpd.h, the exact same function the CPU reference calls, which is why
//     the two populations match to round-off.
//   No inter-thread communication -> pure ensemble parallelism over the cohort.
// ---------------------------------------------------------------------------
__global__ void pkpd_kernel(PkPdParams P, PatientResult* __restrict__ results) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's patient
    if (idx >= P.n_patients) return;                          // guard the ragged last block
    results[idx] = pkpd_integrate(P, idx);                    // the whole patient solve
}

// ---------------------------------------------------------------------------
// integrate_gpu: host wrapper. The canonical CUDA steps for an OUTPUT-only
//   ensemble (there is no input array to copy up -- the "input" is the small
//   PkPdParams struct passed by value into the kernel):
//     (1) allocate one PatientResult slot per patient on the device
//     (2) launch one thread per patient (timed with CUDA events)
//     (3) copy the results device->host
//     (4) free the device buffer
// ---------------------------------------------------------------------------
void integrate_gpu(const PkPdParams& P, std::vector<PatientResult>& results, float* kernel_ms) {
    const int M = P.n_patients;
    results.assign(M, PatientResult{});

    // (1) Device output buffer: M PatientResult structs. d_ prefix = device ptr
    //     (CLAUDE.md §12); dereferencing it on the host would crash.
    PatientResult* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<std::size_t>(M) * sizeof(PatientResult)));

    // (2) Launch: enough blocks to cover all M patients (ceiling division).
    const int blocks = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    pkpd_kernel<<<blocks, THREADS_PER_BLOCK>>>(P, d_out);
    *kernel_ms = timer.stop_ms();          // GPU-measured kernel time (events)
    CUDA_CHECK_LAST("pkpd_kernel");        // catch launch + execution errors

    // (3) Bring the per-patient results back to the host vector.
    CUDA_CHECK(cudaMemcpy(results.data(), d_out,
                          static_cast<std::size_t>(M) * sizeof(PatientResult),
                          cudaMemcpyDeviceToHost));

    // (4) Free the device buffer (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_out));
}
