// ===========================================================================
// src/kernels.cu  --  GPU real-time MRI reconstruction: gridding scatter + cuFFT
// ---------------------------------------------------------------------------
// Project 4.33 : Real-Time MRI Reconstruction
//
// WHAT THIS FILE DOES
//   The GPU twin of reconstruct_frame_cpu(). For each sliding-window frame it runs
//   the full non-uniform-FFT (NUFFT) reconstruction on the device:
//       (a) SCATTER: one thread per radial sample density-compensates it and spreads
//           it onto the ~(W+1)^2 nearest Cartesian grid cells with the Kaiser-Bessel
//           kernel, accumulating into a FIXED-POINT integer grid via atomicAdd.
//       (b) fold the fixed-point grid to complex + apply the FFT-shift checkerboard.
//       (c) INVERSE FFT the grid with cuFFT.
//       (d) deapodize + magnitude, one thread per pixel.
//   All per-sample/per-pixel arithmetic comes from grid_core.h -- the SAME header the
//   CPU reference uses -- so the gridding is bit-identical (fixed-point integers are
//   associative) and the ONLY numerical difference is our radix-2 FFT vs cuFFT.
//
//   The FFT is the expensive, solved part, so we use cuFFT. Per the "no black box"
//   rule (CLAUDE.md section 6.1.6) the plan/exec calls below spell out what cuFFT
//   computes, the layout it expects, and what hand-rolling would take.
//
// READ THIS AFTER: kernels.cuh (declarations + the mapping), grid_core.h (the math),
// reference_cpu.cpp (the CPU twin to compare against).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <cufft.h>               // cufftHandle, cufftPlan2d, cufftExecC2C
#include <cmath>                 // std::cos, std::sin, std::floor (host, for setup)
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

// Cplx is layout-identical to cufftComplex (both are {float,float} == float2), so we
// can reinterpret_cast a Cplx* to a cufftComplex* for the FFT calls with zero copy.
static_assert(sizeof(Cplx) == sizeof(cufftComplex), "Cplx must match cufftComplex");

// ===========================================================================
// SECTION 1 -- The gridding SCATTER kernel (one thread per radial sample)
// ---------------------------------------------------------------------------
// This is the GPU heart of the project. Each thread owns ONE (spoke, readout)
// sample of the current window. It computes the sample's Cartesian k-space
// position, its density-compensation weight, then spreads the weighted value onto
// the ~(W+1)^2 nearest grid cells with the separable Kaiser-Bessel kernel. Because
// neighbouring samples' footprints OVERLAP, many threads add into the same grid
// cell -> we use atomicAdd. To stay deterministic (float atomicAdd is not
// associative), we accumulate in FIXED-POINT INTEGERS (grid_core.h to_fixed) and
// convert back once at the end. See PATTERNS.md section 3.
//
//   grid   : ceil(win*n_ro / 256) blocks ; block = 256 threads
//   thread : global id g -> sample (spoke = spoke0 + g/n_ro, readout = g%n_ro)
//   memory : reads d_samples (global), writes d_acc_re/d_acc_im (global, atomic)
//
// NOTE on the atomic type: CUDA's atomicAdd has no signed-long-long overload, but
// two's-complement addition is bit-identical whether you view the word as signed or
// unsigned. So we reinterpret each 64-bit accumulator as `unsigned long long` and
// add the (reinterpreted) signed contribution -- the stored bits are exactly the
// signed sum. from_fixed() later reads them back as signed.
// ===========================================================================
__global__ void grid_scatter_kernel(const Cplx* __restrict__ d_samples,
                                     int spoke0, int win, int n_ro, int n,
                                     GriddingParams p,
                                     unsigned long long* __restrict__ d_acc_re,
                                     unsigned long long* __restrict__ d_acc_im) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's sample
    const int n_in_window = win * n_ro;                    // samples in the window
    if (g >= n_in_window) return;                          // guard ragged last block

    const int sw   = g / n_ro;              // spoke offset within the window (0..win-1)
    const int j    = g % n_ro;              // readout index within the spoke
    const int sabs = spoke0 + sw;           // absolute spoke index (golden-angle order)

    // --- Sample position in Cartesian grid cells (SAME geometry as the CPU) -------
    // We inline sample_kpos()'s math here (reference_cpu.cpp's definition is host-
    // only); the formula is identical so both paths agree bit-for-bit on placement.
    const double center = 0.5 * n;
    const double ro_off = static_cast<double>(j) - 0.5 * n_ro;                // signed
    const double r      = ro_off * (static_cast<double>(n) / static_cast<double>(n_ro));
    const double theta  = golden_angle_rad(sabs);
    const float  kx = static_cast<float>(center + r * cos(theta));
    const float  ky = static_cast<float>(center + r * sin(theta));

    // --- Density compensation (|k| ramp weight, grid_core.h) ----------------------
    const float roc = static_cast<float>(ro_off) * (static_cast<float>(n) / n_ro);
    const float dcf = radial_dcf(roc);
    const Cplx  val = c_scale(d_samples[(size_t)sabs * n_ro + j], dcf);

    // --- Spread onto the nearby grid cells with the separable KB kernel -----------
    const int half = p.kb_w / 2;
    const int gx0  = static_cast<int>(floorf(kx)) - half;
    const int gy0  = static_cast<int>(floorf(ky)) - half;
    for (int gy = gy0; gy <= gy0 + p.kb_w; ++gy) {
        if (gy < 0 || gy >= n) continue;
        const float wy = kb_weight(fabsf(static_cast<float>(gy) - ky), p);
        if (wy == 0.0f) continue;
        for (int gx = gx0; gx <= gx0 + p.kb_w; ++gx) {
            if (gx < 0 || gx >= n) continue;
            const float wx = kb_weight(fabsf(static_cast<float>(gx) - kx), p);
            if (wx == 0.0f) continue;
            const float w = wx * wy;                          // separable weight
            const size_t idx = (size_t)gy * n + gx;
            // Fixed-point quantize + atomic integer add (deterministic scatter).
            const long long qre = to_fixed(val.re * w);
            const long long qim = to_fixed(val.im * w);
            atomicAdd(&d_acc_re[idx], static_cast<unsigned long long>(qre));
            atomicAdd(&d_acc_im[idx], static_cast<unsigned long long>(qim));
        }
    }
}

// ===========================================================================
// SECTION 2 -- Per-pixel helper kernels (one thread per grid cell / pixel)
// ===========================================================================

// fold_and_ifftshift_kernel: convert the fixed-point accumulators back to a complex
//   grid AND ifftshift it (circular roll by n/2 in both axes) in one pass. The
//   gridded k-space has DC at the center (n/2); the roll moves the cell holding
//   frequency (r-h, c-h) to its true bin ((r+h)%n, (c+h)%n) so a plain inverse FFT
//   lands the image centered -- the SAME operation reconstruct_frame_cpu does (an
//   array roll, NOT a checkerboard multiply, which would shift the wrong way). One
//   thread per SOURCE grid cell; a pure scatter-copy, no atomics (each destination
//   is written exactly once because the roll is a bijection).
__global__ void fold_and_ifftshift_kernel(const unsigned long long* __restrict__ d_acc_re,
                                          const unsigned long long* __restrict__ d_acc_im,
                                          Cplx* __restrict__ d_grid, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * n;
    if (i >= total) return;
    // Reinterpret the unsigned bits back to signed and de-quantize (grid_core.h).
    const long long qre = static_cast<long long>(d_acc_re[i]);
    const long long qim = static_cast<long long>(d_acc_im[i]);
    const int h = n / 2;
    const int r = i / n, c = i % n;
    const int rr = (r + h) % n, cc = (c + h) % n;           // rolled destination
    d_grid[(size_t)rr * n + cc] = c_make(from_fixed(qre), from_fixed(qim));
}

// scale_kernel: v *= s. Applies cuFFT's missing 1/(n*n) inverse normalization, so
// the GPU inverse FFT matches ifft2_cpu (which DOES normalize). One thread per pixel.
__global__ void scale_kernel(Cplx* __restrict__ v, float s, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    v[i] = c_scale(v[i], s);
}

// deapod_magnitude_kernel: FFTSHIFT the inverse-FFT image, divide each pixel by the
//   separable KB deapodization factor deapod(r)*deapod(c) (grid_core.h), then write
//   the magnitude |x| into the frame's output slice. Output pixel (r,c) reads the
//   un-shifted FFT pixel ((r+h)%n,(c+h)%n) -- the fftshift that re-centers the
//   anatomy, matching reconstruct_frame_cpu step (4). One thread per output pixel.
__global__ void deapod_magnitude_kernel(const Cplx* __restrict__ d_grid,
                                        GriddingParams p, float* __restrict__ d_mag) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = p.n * p.n;
    if (i >= total) return;
    const int h = p.n / 2;
    const int r = i / p.n, c = i % p.n;
    const int sr = (r + h) % p.n, sc = (c + h) % p.n;              // fftshift source
    const float deapod = kb_deapod_1d(r, p) * kb_deapod_1d(c, p);  // 2-D correction (centered)
    d_mag[i] = c_abs(c_scale(d_grid[(size_t)sr * p.n + sc], 1.0f / deapod));
}

// ===========================================================================
// SECTION 3 -- Small host helper
// ===========================================================================

// n_blocks: ceiling division so the grid covers every work item.
static inline int n_blocks(int total) {
    return (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

// ===========================================================================
// SECTION 4 -- The full GPU reconstruction of the whole sliding-window movie
// ===========================================================================
void reconstruct_frames_gpu(const RadialData& d,
                            std::vector<float>& out_frames,
                            float* kernel_ms) {
    const int n = d.n;
    const int total = n * n;
    const std::size_t bytesC   = static_cast<std::size_t>(total) * sizeof(Cplx);
    const std::size_t bytesAcc = static_cast<std::size_t>(total) * sizeof(unsigned long long);
    const std::size_t n_samp   = static_cast<std::size_t>(d.n_spokes) * d.n_ro;
    const GriddingParams p = d.params();
    const float invN2 = 1.0f / (static_cast<float>(n) * static_cast<float>(n));

    // ---- Device buffers (d_ prefix marks DEVICE pointers) ------------------
    //   d_samples : all measured radial samples (constant across frames)
    //   d_acc_re/im : fixed-point gridding accumulators (re-zeroed per frame)
    //   d_grid    : the complex Cartesian grid, then the inverse-FFT'd image
    //   d_mag     : one frame's magnitude image
    Cplx *d_samples = nullptr, *d_grid = nullptr;
    unsigned long long *d_acc_re = nullptr, *d_acc_im = nullptr;
    float *d_mag = nullptr;
    CUDA_CHECK(cudaMalloc(&d_samples, n_samp * sizeof(Cplx)));
    CUDA_CHECK(cudaMalloc(&d_grid,    bytesC));
    CUDA_CHECK(cudaMalloc(&d_acc_re,  bytesAcc));
    CUDA_CHECK(cudaMalloc(&d_acc_im,  bytesAcc));
    CUDA_CHECK(cudaMalloc(&d_mag,     static_cast<std::size_t>(total) * sizeof(float)));

    // Upload the measured samples once. d.samples is a vector<Cplx>, bit-compatible
    // with the device Cplx buffer -> a straight memcpy.
    CUDA_CHECK(cudaMemcpy(d_samples, d.samples.data(), n_samp * sizeof(Cplx),
                          cudaMemcpyHostToDevice));

    // ---- cuFFT plan, NOT a black box --------------------------------------
    // cufftPlan2d(&plan, n, n, CUFFT_C2C) builds a plan for one n-by-n complex-to-
    // complex 2D FFT laid out row-major (stride n between rows), exactly our layout.
    // cufftExecC2C(plan, in, out, CUFFT_INVERSE) computes
    //     x[r,c] = sum_{k1,k2} X[k1,k2] * exp(+2*pi*i*(k1*r + k2*c)/n)
    // i.e. the same double sum ifft2_cpu does by hand, and (like ours) leaves it
    // UN-normalized -- so we apply scale_kernel(invN2) ourselves. Hand-rolling this
    // would mean writing + tuning the bit-reversal/butterfly FFT across both axes on
    // the GPU; cuFFT does it faster and correctly. We reuse ONE plan for every frame.
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan2d(&plan, n, n, CUFFT_C2C));
    cufftComplex* fgrid = reinterpret_cast<cufftComplex*>(d_grid);

    const int blocksPix = n_blocks(total);
    const int n_in_win  = d.win * d.n_ro;
    const int blocksSamp = n_blocks(n_in_win);

    out_frames.assign(static_cast<std::size_t>(d.n_frames) * total, 0.0f);

    // ---- The sliding-window movie loop (timed as the teaching artifact) ----
    // Each iteration is ONE real-time frame: grid its window -> cuFFT -> deapodize.
    // In production these frames would pipeline with acquisition via CUDA streams
    // (double-buffering: acquire spoke f+1 while reconstructing frame f). We keep it
    // sequential and synchronous here so the learner can see each stage; THEORY
    // "real world" describes the streamed version.
    GpuTimer timer;
    timer.start();
    for (int f = 0; f < d.n_frames; ++f) {
        const int spoke0 = f * d.stride;                 // first spoke of this window

        // (a) zero the fixed-point accumulators, then scatter this window's samples.
        CUDA_CHECK(cudaMemset(d_acc_re, 0, bytesAcc));
        CUDA_CHECK(cudaMemset(d_acc_im, 0, bytesAcc));
        grid_scatter_kernel<<<blocksSamp, THREADS_PER_BLOCK>>>(
            d_samples, spoke0, d.win, d.n_ro, n, p, d_acc_re, d_acc_im);
        CUDA_CHECK_LAST("grid_scatter_kernel");

        // (b) fold fixed-point -> complex grid + ifftshift (roll DC to index 0).
        fold_and_ifftshift_kernel<<<blocksPix, THREADS_PER_BLOCK>>>(
            d_acc_re, d_acc_im, d_grid, n);
        CUDA_CHECK_LAST("fold_and_ifftshift_kernel");

        // (c) inverse FFT (un-normalized) then apply the 1/(n*n) scale.
        CUFFT_CHECK(cufftExecC2C(plan, fgrid, fgrid, CUFFT_INVERSE));
        scale_kernel<<<blocksPix, THREADS_PER_BLOCK>>>(d_grid, invN2, total);
        CUDA_CHECK_LAST("scale_kernel");

        // (d) deapodize + magnitude into d_mag, then copy this frame to the output.
        deapod_magnitude_kernel<<<blocksPix, THREADS_PER_BLOCK>>>(d_grid, p, d_mag);
        CUDA_CHECK_LAST("deapod_magnitude_kernel");
        CUDA_CHECK(cudaMemcpy(out_frames.data() + static_cast<std::size_t>(f) * total,
                              d_mag, static_cast<std::size_t>(total) * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
    *kernel_ms = timer.stop_ms();

    // ---- Tear down --------------------------------------------------------
    cufftDestroy(plan);
    CUDA_CHECK(cudaFree(d_samples));
    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_acc_re));
    CUDA_CHECK(cudaFree(d_acc_im));
    CUDA_CHECK(cudaFree(d_mag));
}
