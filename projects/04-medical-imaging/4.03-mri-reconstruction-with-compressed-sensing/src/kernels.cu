// ===========================================================================
// src/kernels.cu  --  GPU CS-MRI reconstruction: cuFFT + FISTA kernels
// ---------------------------------------------------------------------------
// Project 4.3 : MRI Reconstruction with Compressed Sensing
//
// WHAT THIS FILE DOES
//   The GPU twin of reconstruct_cpu(). It runs the ENTIRE FISTA loop on the device:
//   two cuFFT transforms per iteration (forward + inverse) plus three one-thread-
//   per-pixel kernels for the masking, prox-gradient, and momentum steps. All the
//   per-pixel arithmetic comes from cs_core.h, the same header the CPU reference
//   uses, so the only numerical difference between CPU and GPU is our hand radix-2
//   FFT vs cuFFT -- exactly the thing this comparison is meant to validate.
//
//   The FFTs are the expensive part and a solved problem, so we use cuFFT. Per the
//   "no black box" rule (CLAUDE.md 6.1.6) the plan/exec calls below spell out what
//   cuFFT computes, the data layout it expects, and what hand-rolling would take.
//
// READ THIS AFTER: kernels.cuh (declarations + the mapping), cs_core.h (the math).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <cufft.h>               // cufftHandle, cufftPlan2d, cufftExecC2C
#include <cmath>                 // std::sqrt (host, for the momentum parameter)
#include <cstdio>
#include <cstdlib>
#include <vector>

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, 8 warps to hide latency, and plenty of resident blocks.
static constexpr int THREADS_PER_BLOCK = 256;

// cuFFT has its own status enum, so it needs its own check macro (mirrors
// CUDA_CHECK but for cufftResult). Every cuFFT call below is guarded by it.
#define CUFFT_CHECK(call)                                                        \
    do {                                                                         \
        cufftResult st__ = (call);                                              \
        if (st__ != CUFFT_SUCCESS) {                                            \
            std::fprintf(stderr, "[CUFFT_CHECK] %s:%d -> cuFFT error %d\n",     \
                         __FILE__, __LINE__, static_cast<int>(st__));           \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                        \
    } while (0)

// Cplx is layout-identical to cufftComplex (both are {float,float} == float2), so
// we can reinterpret_cast a Cplx* to a cufftComplex* for the FFT calls with zero
// copying. This static_assert makes that assumption a COMPILE-TIME guarantee.
static_assert(sizeof(Cplx) == sizeof(cufftComplex), "Cplx must match cufftComplex");

// ===========================================================================
// SECTION 1 -- The per-pixel kernels (one thread per pixel over the n*n image)
// ---------------------------------------------------------------------------
// Launch config for all of them (set in reconstruct_gpu):
//   grid  = ceil(total / THREADS_PER_BLOCK) blocks ; block = THREADS_PER_BLOCK
//   thread-to-data map: i = blockIdx.x*blockDim.x + threadIdx.x owns pixel i.
// They touch only global memory; no shared memory or atomics are needed because
// every pixel updates independently. The heavy lifting is in the cuFFT calls.
// ===========================================================================

// kspace_residual_kernel: r = M (F z - y). `fz` arrives holding F{z} and is
// overwritten in place with the masked residual (so the next inverse FFT reads it).
__global__ void kspace_residual_kernel(Cplx* __restrict__ fz,
                                       const Cplx* __restrict__ y,
                                       const int* __restrict__ mask,
                                       int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;                                   // guard ragged block
    // data_consistency_residual() is the SHARED cs_core.h formula: sampled -> Fz-y,
    // unsampled -> 0. Identical to the CPU path, so the two agree bit-for-bit here.
    fz[i] = data_consistency_residual(fz[i], y[i], mask[i]);
}

// prox_grad_kernel: x = soft_threshold(z - t*grad, t*lambda). This single line is
// the ISTA step -- a gradient descent move followed by the L1 proximal operator
// (both from cs_core.h). Sparsity is enforced right here, per pixel, in parallel.
__global__ void prox_grad_kernel(const Cplx* __restrict__ z,
                                 const Cplx* __restrict__ grad,
                                 float t, float lambda,
                                 Cplx* __restrict__ x, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const Cplx step = c_sub(z[i], c_scale(grad[i], t));       // gradient descent
    x[i] = soft_threshold_cplx(step, t * lambda);             // L1 prox (shrinkage)
}

// momentum_kernel: z = x + beta*(x - x_prev). The Nesterov extrapolation that
// upgrades ISTA (O(1/k)) to FISTA (O(1/k^2)); beta is computed on the host.
__global__ void momentum_kernel(const Cplx* __restrict__ x,
                                const Cplx* __restrict__ x_prev,
                                float beta, Cplx* __restrict__ z, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    z[i] = c_add(x[i], c_scale(c_sub(x[i], x_prev[i]), beta));
}

// scale_kernel: v *= s. Used to apply cuFFT's missing 1/(n*n) inverse
// normalization (cuFFT, like our CPU FFT, leaves the inverse un-normalized).
__global__ void scale_kernel(Cplx* __restrict__ v, float s, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    v[i] = c_scale(v[i], s);
}

// magnitude_kernel: out[i] = |x[i]|. The final magnitude image (phase discarded).
__global__ void magnitude_kernel(const Cplx* __restrict__ x,
                                 float* __restrict__ out, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    out[i] = c_abs(x[i]);
}

// ===========================================================================
// SECTION 2 -- Small host helpers to launch a plan / kernel cleanly
// ===========================================================================

// n_blocks: ceiling division "round up" so the grid covers every pixel.
static inline int n_blocks(int total) {
    return (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

// ===========================================================================
// SECTION 3 -- The full GPU reconstruction (FISTA on the device)
// ===========================================================================
void reconstruct_gpu(const KSpaceData& d, std::vector<float>& out_mag, float* kernel_ms) {
    const int n = d.n;
    const int total = n * n;
    const std::size_t bytesC = static_cast<std::size_t>(total) * sizeof(Cplx);
    const std::size_t bytesI = static_cast<std::size_t>(total) * sizeof(int);
    const float t = 1.0f;                       // gradient step (Lipschitz const = 1)
    const float invN2 = 1.0f / (static_cast<float>(n) * static_cast<float>(n));

    // ---- Device buffers (d_ prefix marks DEVICE pointers) ------------------
    //   d_y    : measured, zero-filled k-space (constant across iterations)
    //   d_mask : sampling mask (constant)
    //   d_x, d_xprev, d_z : the FISTA image estimates (updated each iteration)
    //   d_work : scratch that holds F{z}, then the residual, then the gradient
    Cplx *d_y = nullptr, *d_x = nullptr, *d_xprev = nullptr, *d_z = nullptr, *d_work = nullptr;
    int  *d_mask = nullptr;
    float* d_mag = nullptr;
    CUDA_CHECK(cudaMalloc(&d_y,     bytesC));
    CUDA_CHECK(cudaMalloc(&d_x,     bytesC));
    CUDA_CHECK(cudaMalloc(&d_xprev, bytesC));
    CUDA_CHECK(cudaMalloc(&d_z,     bytesC));
    CUDA_CHECK(cudaMalloc(&d_work,  bytesC));
    CUDA_CHECK(cudaMalloc(&d_mask,  bytesI));
    CUDA_CHECK(cudaMalloc(&d_mag,   static_cast<std::size_t>(total) * sizeof(float)));

    // Upload the measured k-space and mask once. d.kspace is a vector<Cplx>, which
    // is bit-compatible with the device Cplx buffer -> a straight memcpy.
    CUDA_CHECK(cudaMemcpy(d_y,    d.kspace.data(), bytesC, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask, d.mask.data(),   bytesI, cudaMemcpyHostToDevice));

    // ---- cuFFT plans, NOT black boxes -------------------------------------
    // cufftPlan2d(&plan, n, n, CUFFT_C2C) builds a plan for one n-by-n complex-to-
    // complex 2D FFT laid out row-major (stride n between rows), exactly our layout.
    // cufftExecC2C(plan, in, out, dir) then computes, for direction CUFFT_FORWARD:
    //     X[k1,k2] = sum_{r,c} x[r,c] * exp(-2*pi*i*(k1*r + k2*c)/n)
    // i.e. the same double sum fft2_cpu does by hand via the separable radix-2 FFT.
    // CUFFT_INVERSE computes the same sum with +i in the exponent and leaves it
    // UN-normalized (no 1/n^2) -- so we apply scale_kernel(invN2) ourselves, which
    // is precisely what ifft2_cpu does, keeping the two paths identical.
    // Hand-rolling this would mean writing (and tuning) the bit-reversal + butterfly
    // FFT for the GPU across both dimensions; cuFFT does it faster and correctly.
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan2d(&plan, n, n, CUFFT_C2C));

    // Reinterpret our Cplx device pointers as cufftComplex (same layout, asserted).
    cufftComplex* fx    = reinterpret_cast<cufftComplex*>(d_work);
    cufftComplex* czin  = nullptr;  // set per-call below

    const int blocks = n_blocks(total);

    // ---- Warm start: x = z = xprev = F^{-1}{y} (zero-filled adjoint image) --
    // Copy y into work, inverse-FFT it, normalize, then broadcast to x/z/xprev.
    CUDA_CHECK(cudaMemcpy(d_work, d_y, bytesC, cudaMemcpyDeviceToDevice));
    czin = reinterpret_cast<cufftComplex*>(d_work);
    CUFFT_CHECK(cufftExecC2C(plan, czin, czin, CUFFT_INVERSE));      // un-normalized
    scale_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_work, invN2, total);
    CUDA_CHECK_LAST("scale_kernel(warmstart)");
    CUDA_CHECK(cudaMemcpy(d_x,     d_work, bytesC, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_z,     d_work, bytesC, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_xprev, d_work, bytesC, cudaMemcpyDeviceToDevice));

    // ---- FISTA loop (timed as the teaching artifact) -----------------------
    float theta = 1.0f;                          // FISTA momentum parameter t_k
    GpuTimer timer;
    timer.start();
    for (int it = 0; it < d.iters; ++it) {
        // (a) forward FFT of the look-ahead point: work = F{z}
        CUDA_CHECK(cudaMemcpy(d_work, d_z, bytesC, cudaMemcpyDeviceToDevice));
        CUFFT_CHECK(cufftExecC2C(plan, fx, fx, CUFFT_FORWARD));
        // (b) masked k-space residual: work = M (F z - y)
        kspace_residual_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_work, d_y, d_mask, total);
        CUDA_CHECK_LAST("kspace_residual_kernel");
        // (c) inverse FFT -> gradient in image space: work = F^{-1}{ residual }
        CUFFT_CHECK(cufftExecC2C(plan, fx, fx, CUFFT_INVERSE));
        scale_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_work, invN2, total);
        CUDA_CHECK_LAST("scale_kernel(grad)");
        // (d) prox-gradient update: x = soft_threshold(z - t*grad, t*lambda)
        prox_grad_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_z, d_work, t, d.lambda, d_x, total);
        CUDA_CHECK_LAST("prox_grad_kernel");
        // (e) FISTA momentum: theta and beta on the host (scalars), z on the device.
        const float theta_next = 0.5f * (1.0f + std::sqrt(1.0f + 4.0f * theta * theta));
        const float beta = (theta - 1.0f) / theta_next;
        momentum_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_x, d_xprev, beta, d_z, total);
        CUDA_CHECK_LAST("momentum_kernel");
        // (f) roll x -> x_prev for the next iteration's momentum term.
        CUDA_CHECK(cudaMemcpy(d_xprev, d_x, bytesC, cudaMemcpyDeviceToDevice));
        theta = theta_next;
    }
    *kernel_ms = timer.stop_ms();

    // ---- Final magnitude image |x| and copy back to the host --------------
    magnitude_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_x, d_mag, total);
    CUDA_CHECK_LAST("magnitude_kernel");
    out_mag.resize(static_cast<std::size_t>(total));
    CUDA_CHECK(cudaMemcpy(out_mag.data(), d_mag,
                          static_cast<std::size_t>(total) * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // ---- Tear down ---------------------------------------------------------
    cufftDestroy(plan);
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_xprev));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_mag));
}
