// ===========================================================================
// src/kernels.cu  --  Cohort simulation kernel (one thread per virtual patient)
// ---------------------------------------------------------------------------
// Project 6.23 : Glucose-Insulin Dynamics & Artificial Pancreas
//
// WHAT THIS FILE DOES
//   Implements the device kernel (cohort_kernel) and the host-side glue
//   (simulate_cohort_gpu) that allocates a device result buffer, launches one
//   thread per patient, times the kernel, and brings the results back. This is
//   the GPU twin of simulate_cohort_cpu() in reference_cpu.cpp; main.cu runs
//   both and compares them per patient.
//
//   Because the per-patient work (simulate_patient) lives in bergman.h as a
//   shared __host__ __device__ function, this file is thin: the kernel just maps
//   thread -> patient and calls the same routine the CPU uses. See ../THEORY.md.
//
// READ THIS AFTER: kernels.cuh (declarations + the thread-mapping idea).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 128 is a solid default for a REGISTER-HEAVY per-thread
// integrator like this: each thread runs a long RK4 loop holding the full state
// + controller memory + RK4 temporaries in registers, so we keep the block small
// to leave enough registers per thread for good occupancy. (Multiple of the
// 32-lane warp; tune per GPU.)
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// cohort_kernel: one thread simulates one virtual patient, start to finish.
//   Launch config (set in simulate_cohort_gpu):
//     grid  = ceil(M / THREADS_PER_BLOCK) blocks, M = cohort_size(c)
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: idx = blockIdx.x * blockDim.x + threadIdx.x owns patient
//   idx. The thread builds that patient's parameters (patient_params) and runs
//   the entire closed-loop simulation (simulate_patient) in local registers,
//   then writes ONE PatientResult to global memory. No shared memory, no
//   atomics, no inter-thread communication -- pure embarrassing parallelism over
//   patients. Divergence is mild: all patients run the same step count; only the
//   min/max/in-range branches differ per step, which is cheap.
// ---------------------------------------------------------------------------
__global__ void cohort_kernel(CohortConfig c, PatientResult* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cohort_size(c)) return;    // guard the ragged last block

    // Build this patient's full parameter set from the cohort grid, then run the
    // SAME simulation function the CPU reference uses -> identical math.
    const PatientParams p = patient_params(c, idx);
    out[idx] = simulate_patient(p);
}

// ---------------------------------------------------------------------------
// simulate_cohort_gpu: host wrapper. The canonical steps of a CUDA computation,
//   minus H2D input copies (the only "input" is the small CohortConfig, which we
//   pass BY VALUE as a kernel argument -- it fits in the constant argument bank):
//     (1) allocate the device result buffer
//     (2) launch one thread per patient
//     (3) copy the results device->host
//     (4) free device memory
//   We time ONLY step (2) with CUDA events so the reported figure is the kernel
//   compute cost, not the (tiny) result copy.
// ---------------------------------------------------------------------------
void simulate_cohort_gpu(const CohortConfig& c, std::vector<PatientResult>& results,
                         float* kernel_ms) {
    const int M = cohort_size(c);
    results.assign(static_cast<std::size_t>(M), PatientResult{});

    // (1) Device buffer for the M per-patient summaries. d_ = DEVICE pointer.
    PatientResult* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<std::size_t>(M) * sizeof(PatientResult)));

    // (2) Launch. Blocks must cover all M patients: ceiling division "round up".
    const int blocks = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    cohort_kernel<<<blocks, THREADS_PER_BLOCK>>>(c, d_out);
    *kernel_ms = timer.stop_ms();            // GPU-measured kernel time
    CUDA_CHECK_LAST("cohort_kernel");        // catch launch + execution errors

    // (3) Bring the per-patient results back to the host vector.
    CUDA_CHECK(cudaMemcpy(results.data(), d_out,
                          static_cast<std::size_t>(M) * sizeof(PatientResult),
                          cudaMemcpyDeviceToHost));

    // (4) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_out));
}
