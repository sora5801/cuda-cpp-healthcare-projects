// ===========================================================================
// src/kernels.cu  --  cuFFT cross-correlation alignment search (the GPU twin)
// ---------------------------------------------------------------------------
// Project 2.4 : Cryo-ET Subtomogram Averaging  (reduced-scope teaching version)
//
// WHAT THIS FILE DOES
//   Implements align_gpu(): for every (candidate, trial-angle) job it rotates
//   the candidate, then uses cuFFT + the CROSS-CORRELATION THEOREM to get the
//   correlation against the reference at ALL translational shifts in O(V log V),
//   and reduces each job to its (zero-shift NCC, peak NCC). The CPU reference in
//   reference_cpu.cpp computes the zero-shift NCC the slow direct way; main.cu
//   checks the two agree -- proving the FFT identity to the learner.
//
//   The five device steps map 1:1 to the kernels below. See ../THEORY.md
//   "GPU mapping" for the full picture and kernels.cuh for the interface.
//
// READ THIS AFTER: kernels.cuh, reference_cpu.h.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <cufft.h>               // cufftHandle, cufftExecR2C/C2R, cufftPlanMany
#include <cmath>                 // cosf, sinf, floorf, sqrtf, fabsf
#include <cstdio>                // std::fprintf
#include <cstdlib>               // std::exit, EXIT_FAILURE
#include <vector>

// 256 threads/block: a multiple of the 32-lane warp, good occupancy on
// sm_75..sm_89, and a power of two so the block reduction below is clean.
static constexpr int THREADS_PER_BLOCK = 256;

// cuFFT has its own status enum, so it needs its own check macro (mirrors
// CUDA_CHECK but for cufftResult). EVERY cuFFT call is guarded and explained.
#define CUFFT_CHECK(call)                                                       \
    do {                                                                        \
        cufftResult st__ = (call);                                              \
        if (st__ != CUFFT_SUCCESS) {                                            \
            std::fprintf(stderr, "[CUFFT_CHECK] %s:%d -> error %d\n",           \
                         __FILE__, __LINE__, static_cast<int>(st__));           \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// ===========================================================================
// Kernel 1: rotate_kernel -- in-plane rotation about z (GPU twin of CPU).
//   Thread map: one thread per OUTPUT voxel of one (candidate, angle) job.
//     grid.x * blockDim.x covers the V voxels of a cube;
//     grid.y = n_angles  (which trial angle),  grid.z = n_sub (which candidate).
//   Job index j = cand * n_angles + angle; output cube j lives at out[j*V ...].
//   Arithmetic is bit-for-bit the same backward-map + bilinear blend as
//   rotate_cube_cpu(), so the rotated cubes match the CPU's (within float
//   rounding, which the NCC tolerance absorbs).
// ===========================================================================
__global__ void rotate_kernel(const float* __restrict__ cand,
                              float* __restrict__ out,
                              int d, int n_sub, int n_angles) {
    const int V = d * d * d;                       // voxels per cube
    const int voxel = blockIdx.x * blockDim.x + threadIdx.x;  // 0..V-1 within cube
    if (voxel >= V) return;                        // guard the ragged last block

    const int angle = blockIdx.y;                  // which trial angle
    const int cand_id = blockIdx.z;                // which candidate
    const int job = cand_id * n_angles + angle;    // flat job index

    // Decode the linear voxel index into (x,y,z) of this cube (z-major layout).
    const int x = voxel % d;
    const int y = (voxel / d) % d;
    const int z = voxel / (d * d);

    // Same angle formula as trial_angle() on the host (recomputed, not passed,
    // so host and device cannot drift). 2*pi*angle/n_angles.
    const float theta = 6.28318530717958647692f * static_cast<float>(angle)
                        / static_cast<float>(n_angles);
    const float c = cosf(theta);
    const float s = sinf(theta);
    const float center = 0.5f * (static_cast<float>(d) - 1.0f);

    // Backward map: where in the input did output (x,y) come from? (inverse rot)
    const float ox = static_cast<float>(x) - center;
    const float oy = static_cast<float>(y) - center;
    const float sx = center + (c * ox + s * oy);
    const float sy = center + (-s * ox + c * oy);

    // Bilinear interpolation at (sx, sy); out-of-bounds corners contribute 0.
    const int x0 = static_cast<int>(floorf(sx));
    const int y0 = static_cast<int>(floorf(sy));
    const float fx = sx - static_cast<float>(x0);
    const float fy = sy - static_cast<float>(y0);
    const float* src = &cand[static_cast<size_t>(cand_id) * V];   // this candidate
    float val = 0.0f;
    #pragma unroll
    for (int dy = 0; dy <= 1; ++dy) {
        for (int dx = 0; dx <= 1; ++dx) {
            const int xx = x0 + dx;
            const int yy = y0 + dy;
            if (xx < 0 || xx >= d || yy < 0 || yy >= d) continue;
            const float wx = dx ? fx : (1.0f - fx);
            const float wy = dy ? fy : (1.0f - fy);
            val += wx * wy * src[(static_cast<size_t>(z) * d + yy) * d + xx];
        }
    }
    out[static_cast<size_t>(job) * V + voxel] = val;
}

// ===========================================================================
// Kernel 2: xcorr_mul_kernel -- the cross-correlation theorem, per frequency.
//   For cross-correlation (not convolution) we multiply each job's spectrum by
//   the CONJUGATE of the reference spectrum:  H = conj(REF) .* JOB.
//   conj flips the sign of the imaginary part -> a shift in the result, which is
//   exactly the difference between correlation and convolution. cufftComplex IS
//   float2 (.x real, .y imag). One thread per (job, frequency bin); writes back
//   into the job's spectrum in place.
// ===========================================================================
__global__ void xcorr_mul_kernel(float2* __restrict__ job_spec,
                                 const float2* __restrict__ ref_spec,
                                 int nfreq, int n_jobs) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;  // over n_jobs*nfreq
    const int total = nfreq * n_jobs;
    if (idx >= total) return;

    const int f = idx % nfreq;                 // which frequency bin
    const float2 R = ref_spec[f];              // reference spectrum (shared by all jobs)
    const float2 J = job_spec[idx];            // this job's spectrum

    // Complex multiply conj(R) * J:
    //   conj(R) = (R.x, -R.y)
    //   (a - bi)(c + di) = (ac + bd) + (ad - bc) i
    float2 H;
    H.x = R.x * J.x + R.y * J.y;
    H.y = R.x * J.y - R.y * J.x;
    job_spec[idx] = H;
}

// ===========================================================================
// Kernel 3: reduce_kernel -- per-job peak + zero-shift, normalized to NCC.
//   One BLOCK per job. Threads cooperatively scan the job's V-voxel correlation
//   field to find the maximum (the best translational shift). corr was produced
//   by an unnormalized cuFFT R2C->C2R round trip, which scales the result by V,
//   so we divide by V (invV) to recover the true correlation sum. We then divide
//   by sqrt(ref_energy * job_energy) to get the normalized cross-correlation in
//   [-1, 1]. The zero-shift value is simply corr[0] (output voxel (0,0,0)).
//
//   The reduction is a deterministic tree reduction in shared memory (no
//   atomics), so the peak is bit-stable across runs (PATTERNS.md §3).
// ===========================================================================
__global__ void reduce_kernel(const float* __restrict__ corr,
                              const float* __restrict__ job_energy,
                              float ref_energy, int V, float invV,
                              float* __restrict__ out_zero,
                              float* __restrict__ out_peak) {
    extern __shared__ float smax[];           // blockDim.x partial maxima
    const int job = blockIdx.x;               // one block == one job
    const float* c = &corr[static_cast<size_t>(job) * V];

    // Grid-stride scan: each thread takes the max over its slice of the V voxels.
    float local = -3.4e38f;                   // ~ -FLT_MAX
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        const float v = c[i];
        if (v > local) local = v;
    }
    smax[threadIdx.x] = local;
    __syncthreads();

    // Tree reduction: halve the active threads each step (deterministic order).
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other = smax[threadIdx.x + stride];
            if (other > smax[threadIdx.x]) smax[threadIdx.x] = other;
        }
        __syncthreads();
    }

    // Thread 0 writes the two normalized scores for this job.
    if (threadIdx.x == 0) {
        const float denom = sqrtf(ref_energy * job_energy[job]);
        const float scale = (denom > 0.0f) ? (invV / denom) : 0.0f;
        out_peak[job] = smax[0] * scale;       // best over all shifts
        out_zero[job] = c[0] * scale;          // shift (0,0,0)
    }
}

// ===========================================================================
// energy_kernel: sum of squares of each cube (its "energy"), used to normalize
//   the correlation into an NCC. One block per cube; the same deterministic tree
//   reduction as above but summing v*v. Done on the GPU so the per-job energy is
//   computed from the EXACT rotated cube the FFT saw.
// ===========================================================================
__global__ void energy_kernel(const float* __restrict__ cubes, int V,
                              float* __restrict__ out_energy) {
    extern __shared__ float ssum[];
    const int job = blockIdx.x;
    const float* c = &cubes[static_cast<size_t>(job) * V];

    float local = 0.0f;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        const float v = c[i];
        local += v * v;
    }
    ssum[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) ssum[threadIdx.x] += ssum[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out_energy[job] = ssum[0];
}

// ===========================================================================
// align_gpu: the host wrapper that drives the whole pipeline (5 steps).
//   We time steps 1-5 with CUDA events (the GPU work) and leave the host-side
//   argmax-over-angles out of the timed region (it is trivial and serial).
// ===========================================================================
void align_gpu(const SubtomogramSet& set,
               std::vector<double>& ncc_zero_shift,
               std::vector<double>& ncc_peak,
               std::vector<int>& best_angle,
               float* kernel_ms) {
    const int d = set.d;
    const int V = set.vol();                       // voxels per cube = d^3
    const int n_jobs = set.n_sub * set.n_angles;   // total (cand, angle) jobs
    const int nfreq = d * d * (d / 2 + 1);         // R2C output bins per 3-D cube

    ncc_zero_shift.assign(static_cast<std::size_t>(n_jobs), 0.0);
    ncc_peak.assign(static_cast<std::size_t>(n_jobs), 0.0);
    best_angle.assign(static_cast<std::size_t>(set.n_sub), 0);

    // ---- Device buffers ---------------------------------------------------
    float*  d_cand = nullptr;     // [n_sub * V]   candidate cubes (input)
    float*  d_ref  = nullptr;     // [V]           reference cube
    float*  d_rot  = nullptr;     // [n_jobs * V]  rotated candidates (real)
    float2* d_job_spec = nullptr; // [n_jobs * nfreq] job spectra (complex)
    float2* d_ref_spec = nullptr; // [nfreq]       reference spectrum
    float*  d_corr = nullptr;     // [n_jobs * V]  correlation fields (real)
    float*  d_job_energy = nullptr; // [n_jobs]    per-rotated-cube energy
    float*  d_ref_energy = nullptr; // [1]         reference energy

    CUDA_CHECK(cudaMalloc(&d_cand, static_cast<size_t>(set.n_sub) * V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref,  static_cast<size_t>(V) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rot,  static_cast<size_t>(n_jobs) * V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_job_spec, static_cast<size_t>(n_jobs) * nfreq * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_ref_spec, static_cast<size_t>(nfreq) * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_corr, static_cast<size_t>(n_jobs) * V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_job_energy, static_cast<size_t>(n_jobs) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref_energy, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_cand, set.cand.data(),
                          static_cast<size_t>(set.n_sub) * V * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ref, set.ref.data(),
                          static_cast<size_t>(V) * sizeof(float),
                          cudaMemcpyHostToDevice));

    // ---- cuFFT plans (NOT black boxes) ------------------------------------
    // A 3-D real-to-complex FFT of a d x d x d cube produces a d x d x (d/2+1)
    // complex array (Hermitian symmetry of a real signal halves the last axis).
    // cufftPlanMany lets us BATCH n_jobs such transforms in one call: the input
    // stride is V (one cube), the output stride is nfreq (one spectrum). The
    // reference gets its own batch-of-1 plans. We need both directions:
    //   R2C: real cube       -> complex spectrum  (forward)
    //   C2R: complex spectrum -> real corr. field (inverse, unnormalized -> xV)
    int dims[3] = {d, d, d};      // logical transform size, slowest-varying first
    cufftHandle plan_fwd_jobs, plan_inv_jobs, plan_fwd_ref;
    CUFFT_CHECK(cufftPlanMany(&plan_fwd_jobs, 3, dims,
                              nullptr, 1, V,        // input:  contiguous, stride V
                              nullptr, 1, nfreq,    // output: contiguous, stride nfreq
                              CUFFT_R2C, n_jobs));
    CUFFT_CHECK(cufftPlanMany(&plan_inv_jobs, 3, dims,
                              nullptr, 1, nfreq,    // input:  complex, stride nfreq
                              nullptr, 1, V,        // output: real,    stride V
                              CUFFT_C2R, n_jobs));
    CUFFT_CHECK(cufftPlan3d(&plan_fwd_ref, d, d, d, CUFFT_R2C));   // batch of 1

    const float invV = 1.0f / static_cast<float>(V);

    // ---- Time the whole GPU pipeline --------------------------------------
    GpuTimer timer;
    timer.start();

    // STEP 1: rotate every candidate by every trial angle.
    //   grid.x covers V voxels; grid.y = n_angles; grid.z = n_sub.
    {
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid((V + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                  set.n_angles, set.n_sub);
        rotate_kernel<<<grid, block>>>(d_cand, d_rot, d, set.n_sub, set.n_angles);
        CUDA_CHECK_LAST("rotate_kernel");
    }

    // STEP 1b: energies of the reference and of every rotated cube (for NCC).
    {
        const int shmem = THREADS_PER_BLOCK * sizeof(float);
        energy_kernel<<<1, THREADS_PER_BLOCK, shmem>>>(d_ref, V, d_ref_energy);
        CUDA_CHECK_LAST("energy_kernel(ref)");
        energy_kernel<<<n_jobs, THREADS_PER_BLOCK, shmem>>>(d_rot, V, d_job_energy);
        CUDA_CHECK_LAST("energy_kernel(jobs)");
    }

    // STEP 2: forward FFT the reference and the whole batch of rotated cubes.
    //   cufftExecR2C computes X[f] = sum_v x[v] exp(-2*pi*i f.v / N) for each f.
    CUFFT_CHECK(cufftExecR2C(plan_fwd_ref, d_ref, d_ref_spec));
    CUFFT_CHECK(cufftExecR2C(plan_fwd_jobs, d_rot, d_job_spec));

    // STEP 3: per-frequency conj(REF) .* JOB (the cross-correlation theorem).
    {
        const int total = nfreq * n_jobs;
        const int grid = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        xcorr_mul_kernel<<<grid, THREADS_PER_BLOCK>>>(d_job_spec, d_ref_spec, nfreq, n_jobs);
        CUDA_CHECK_LAST("xcorr_mul_kernel");
    }

    // STEP 4: inverse FFT each job's product back to a real correlation field.
    //   cuFFT's C2R is UNNORMALIZED: the round trip scales by V, undone in step 5.
    CUFFT_CHECK(cufftExecC2R(plan_inv_jobs, d_job_spec, d_corr));

    // STEP 5: reduce each correlation field to (peak, zero-shift), as NCC.
    //   We copy ref_energy to a host float so we can pass it by value.
    float h_ref_energy = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_ref_energy, d_ref_energy, sizeof(float), cudaMemcpyDeviceToHost));
    float* d_zero = nullptr;  // [n_jobs] zero-shift NCC
    float* d_peak = nullptr;  // [n_jobs] peak NCC
    CUDA_CHECK(cudaMalloc(&d_zero, static_cast<size_t>(n_jobs) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_peak, static_cast<size_t>(n_jobs) * sizeof(float)));
    {
        const int shmem = THREADS_PER_BLOCK * sizeof(float);
        reduce_kernel<<<n_jobs, THREADS_PER_BLOCK, shmem>>>(
            d_corr, d_job_energy, h_ref_energy, V, invV, d_zero, d_peak);
        CUDA_CHECK_LAST("reduce_kernel");
    }

    *kernel_ms = timer.stop_ms();   // GPU-measured time of steps 1-5

    // ---- Copy results back and finish the argmax-over-angles on the host ---
    std::vector<float> h_zero(static_cast<std::size_t>(n_jobs));
    std::vector<float> h_peak(static_cast<std::size_t>(n_jobs));
    CUDA_CHECK(cudaMemcpy(h_zero.data(), d_zero, static_cast<size_t>(n_jobs) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_peak.data(), d_peak, static_cast<size_t>(n_jobs) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (int j = 0; j < n_jobs; ++j) {
        ncc_zero_shift[static_cast<std::size_t>(j)] = h_zero[static_cast<std::size_t>(j)];
        ncc_peak[static_cast<std::size_t>(j)] = h_peak[static_cast<std::size_t>(j)];
    }
    // Best angle per candidate = argmax of the PEAK NCC over its angles. Ties
    // resolve to the lowest angle index (strict >), matching the CPU rule.
    for (int s = 0; s < set.n_sub; ++s) {
        double best = -2.0;
        int best_k = 0;
        for (int k = 0; k < set.n_angles; ++k) {
            const double v = ncc_peak[static_cast<std::size_t>(s) * set.n_angles + k];
            if (v > best) { best = v; best_k = k; }
        }
        best_angle[static_cast<std::size_t>(s)] = best_k;
    }

    // ---- Tear down --------------------------------------------------------
    cufftDestroy(plan_fwd_jobs);
    cufftDestroy(plan_inv_jobs);
    cufftDestroy(plan_fwd_ref);
    CUDA_CHECK(cudaFree(d_cand));
    CUDA_CHECK(cudaFree(d_ref));
    CUDA_CHECK(cudaFree(d_rot));
    CUDA_CHECK(cudaFree(d_job_spec));
    CUDA_CHECK(cudaFree(d_ref_spec));
    CUDA_CHECK(cudaFree(d_corr));
    CUDA_CHECK(cudaFree(d_job_energy));
    CUDA_CHECK(cudaFree(d_ref_energy));
    CUDA_CHECK(cudaFree(d_zero));
    CUDA_CHECK(cudaFree(d_peak));
}
