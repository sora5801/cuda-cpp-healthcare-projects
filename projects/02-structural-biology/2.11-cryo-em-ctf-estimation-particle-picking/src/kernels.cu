// ===========================================================================
// src/kernels.cu  --  cuFFT 2-D power spectrum, radial average, defocus search
// ---------------------------------------------------------------------------
// Project 2.11 : Cryo-EM CTF Estimation & Particle Picking
//
// GPU twin of reference_cpu.cpp. Three custom kernels surround one cuFFT call:
//   * power_kernel          : |X|^2 of the cuFFT half-spectrum (one thread/bin)
//   * radial_average_kernel : scatter |X|^2 into ring bins with FIXED-POINT atomics
//                             (integer atomics commute -> reproducible & == CPU)
//   * ctf_search_kernel     : one thread per candidate defocus, scores via the
//                             SHARED ncc_model_vs_profile() (ctf_model.h)
// See ../THEORY.md "GPU mapping". cuFFT is explained in full below -- not a black
// box (CLAUDE.md §6.1.6).
// ===========================================================================
#include "kernels.cuh"
#include "ctf_model.h"            // shared CTF physics (host+device)
#include "util/cuda_check.cuh"    // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"         // GpuTimer

#include <cufft.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

// cuFFT has its own status type (cufftResult), so it needs its own check macro
// mirroring CUDA_CHECK. Every cuFFT call below is guarded and explained.
#define CUFFT_CHECK(call)                                                        \
    do {                                                                         \
        cufftResult st__ = (call);                                              \
        if (st__ != CUFFT_SUCCESS) {                                            \
            std::fprintf(stderr, "[CUFFT_CHECK] %s:%d -> cuFFT error %d\n",      \
                         __FILE__, __LINE__, static_cast<int>(st__));           \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                        \
    } while (0)

// Fixed-point scale for the radial-average atomics. The radial average sums
// |X|^2 values (large, positive) and divides by the ring count. Floating-point
// atomicAdd is NON-associative, so the thread-ordering would make the result (and
// thus the fit) irreproducible. Instead we accumulate in 64-bit INTEGER units of
// (power * FIXED_SCALE): integer adds commute, so the sum is identical every run
// AND identical to the CPU's ordered sum. This is the §3/§4 determinism rule.
//   |X|^2 here is O(image_energy^2); FIXED_SCALE=256 keeps ~8 fractional bits
//   while staying far below the 2^63 overflow ceiling for our bin counts.
__device__ __constant__ double FIXED_SCALE = 256.0;

// Constant-memory home for the observed radial profile during the defocus search.
// EVERY search thread reads the whole profile but never writes it -> constant
// memory's broadcast cache is the ideal placement (same idea as the query in the
// 1.12 Tanimoto flagship). MAX_BINS caps the side length at 2*MAX_BINS pixels.
#define MAX_BINS 4096
__device__ __constant__ double c_profile[MAX_BINS];

// ---------------------------------------------------------------------------
// power_kernel: power[i] = |X[i]|^2 for each complex bin of the cuFFT R2C output.
//   cufftComplex IS float2 (.x real, .y imag); one thread per (row, half-column).
//   We do NOT normalize by N^2 here -- the radial average and the NCC are both
//   scale-invariant, so any constant factor cancels. (The CPU reference likewise
//   compares un-normalized powers.)
// ---------------------------------------------------------------------------
__global__ void power_kernel(const float2* __restrict__ X, int total,
                             float* __restrict__ power) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's bin
    if (i < total) {
        const float2 v = X[i];
        power[i] = v.x * v.x + v.y * v.y;
    }
}

// ---------------------------------------------------------------------------
// radial_average_kernel: scatter the half-spectrum power into radial ring bins.
//   cuFFT R2C stores only the non-redundant half (columns u = 0..N/2 due to the
//   Hermitian symmetry of a real image's FFT). For each stored bin (u, v):
//     * the signed vertical frequency is fv = (v<=N/2)? v : v-N
//     * the horizontal frequency is fu = u (the stored half is u>=0)
//     * ring r = round(sqrt(fu^2 + fv^2))
//   We add the bin's power into ring r and increment its count, BOTH via integer
//   atomics (fixed-point power, integer count) so the reduction is deterministic.
//   nbins == N/2.
//
//   grid/block: one thread per stored bin (N rows x (N/2+1) half-cols). The thread
//   that owns linear index `i` decodes (u,v) = (i % halfW, i / halfW).
// ---------------------------------------------------------------------------
__global__ void radial_average_kernel(const float* __restrict__ power,
                                      int N, int halfW, int nbins,
                                      unsigned long long* __restrict__ ring_sum_fp,
                                      unsigned long long* __restrict__ ring_cnt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * halfW;                    // stored half-spectrum size
    if (i >= total) return;

    const int u = i % halfW;                        // horizontal freq (0..N/2)
    const int v = i / halfW;                        // row index (0..N-1)
    const int fv = (v <= N / 2) ? v : v - N;        // fold to signed vertical freq
    const int fu = u;                               // stored half is u >= 0
    const double rr = sqrt(static_cast<double>(fu) * fu +
                           static_cast<double>(fv) * fv);
    const int r = static_cast<int>(rr + 0.5);       // nearest ring (matches CPU)
    if (r < 0 || r >= nbins) return;

    // Fixed-point integer accumulation -> deterministic, commutes across threads.
    const unsigned long long q =
        static_cast<unsigned long long>(static_cast<double>(power[i]) * FIXED_SCALE + 0.5);
    atomicAdd(&ring_sum_fp[r], q);
    atomicAdd(&ring_cnt[r], 1ULL);
}

// ---------------------------------------------------------------------------
// ctf_search_kernel: one thread per CANDIDATE defocus. Thread i reads the whole
//   observed profile from constant memory and computes the NCC of its model
//   |CTF(k; dz_i)|^2 against it, using the SHARED ncc_model_vs_profile() so the
//   value is bit-identical to the CPU. Writes scores[i]; the host takes the argmax
//   (a tiny n_dz reduction not worth a second kernel launch).
//   thread (blockIdx.x, threadIdx.x) -> candidate index i = bx*blockDim.x + tx.
// ---------------------------------------------------------------------------
__global__ void ctf_search_kernel(int n_dz, double dz_min, double dz_step,
                                  int nbins, int r_lo, int r_hi,
                                  CtfParams optics, int half, double nyquist_k,
                                  double* __restrict__ scores) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dz) return;
    const double dz = dz_min + dz_step * i;         // this thread's defocus (A)
    scores[i] = ncc_model_vs_profile(c_profile, nbins, r_lo, r_hi,
                                     dz, optics, half, nyquist_k);
}

// ===========================================================================
// radial_power_profile_gpu  (stages 1+2): cuFFT 2-D FFT -> |X|^2 -> radial avg.
// ===========================================================================
void radial_power_profile_gpu(const Micrograph& m, int nbins,
                              std::vector<double>& raw, float* kernel_ms) {
    const int N     = m.n;
    const int halfW = N / 2 + 1;                     // R2C output columns per row
    const int total = N * halfW;                     // half-spectrum element count

    // --- device buffers ---
    cufftReal*    d_img   = nullptr;                  // [N*N]   real input image
    cufftComplex* d_spec  = nullptr;                  // [N*halfW] complex spectrum
    float*        d_power = nullptr;                  // [total] |X|^2
    unsigned long long* d_ring_sum = nullptr;         // [nbins] fixed-point sums
    unsigned long long* d_ring_cnt = nullptr;         // [nbins] ring counts
    CUDA_CHECK(cudaMalloc(&d_img,   static_cast<std::size_t>(N) * N * sizeof(cufftReal)));
    CUDA_CHECK(cudaMalloc(&d_spec,  static_cast<std::size_t>(total) * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_power, static_cast<std::size_t>(total) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ring_sum, static_cast<std::size_t>(nbins) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_ring_cnt, static_cast<std::size_t>(nbins) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_ring_sum, 0, static_cast<std::size_t>(nbins) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_ring_cnt, 0, static_cast<std::size_t>(nbins) * sizeof(unsigned long long)));

    // The CPU reference removes the image mean before the FFT (kills the DC spike).
    // Do the same on the host copy so the spectra match. cufftReal == float.
    double mean = 0.0;
    for (float v : m.pix) mean += v;
    mean /= static_cast<double>(N) * N;
    std::vector<float> centred(m.pix.size());
    for (std::size_t i = 0; i < m.pix.size(); ++i)
        centred[i] = static_cast<float>(static_cast<double>(m.pix[i]) - mean);
    CUDA_CHECK(cudaMemcpy(d_img, centred.data(),
                          static_cast<std::size_t>(N) * N * sizeof(cufftReal),
                          cudaMemcpyHostToDevice));

    // ---- The library call, NOT a black box -------------------------------
    // cufftPlan2d(plan, N, N, CUFFT_R2C) builds a plan for ONE 2-D real-to-complex
    // FFT of an N x N image. cufftExecR2C then computes, for output bin (u,v):
    //     X[v,u] = sum_{y=0}^{N-1} sum_{x=0}^{N-1} img(x,y) * exp(-2*pi*i*(u x + v y)/N)
    // i.e. the same 2-D DFT radial_power_profile_cpu() does by hand -- but in
    // O(N^2 log N) instead of O(N^4). R2C exploits the Hermitian symmetry of a real
    // image (X[-u,-v] = conj(X[u,v])), so it stores only the non-redundant half:
    // (N/2+1) columns per row, row-major [v * (N/2+1) + u]. Hand-rolling this means
    // writing a mixed-radix Cooley-Tukey FFT with bit-reversal and twiddle tables --
    // exactly the kind of solved problem a library should own.
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan2d(&plan, N, N, CUFFT_R2C));

    const int block = 256;
    const int gridP = (total + block - 1) / block;

    GpuTimer timer;
    timer.start();
    CUFFT_CHECK(cufftExecR2C(plan, d_img, d_spec));                  // the 2-D FFT
    power_kernel<<<gridP, block>>>(reinterpret_cast<const float2*>(d_spec),
                                   total, d_power);                  // |X|^2
    radial_average_kernel<<<gridP, block>>>(d_power, N, halfW, nbins,
                                            d_ring_sum, d_ring_cnt);  // ring scatter
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("power/radial kernels");

    // Copy the integer ring sums/counts back and form the rotational average on the
    // host (a tiny nbins-long loop). Dividing fixed-point sum by count and undoing
    // FIXED_SCALE recovers the mean power per ring -- matching the CPU's ordered sum.
    std::vector<unsigned long long> ring_sum(nbins), ring_cnt(nbins);
    CUDA_CHECK(cudaMemcpy(ring_sum.data(), d_ring_sum,
                          static_cast<std::size_t>(nbins) * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ring_cnt.data(), d_ring_cnt,
                          static_cast<std::size_t>(nbins) * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    raw.assign(nbins, 0.0);
    for (int r = 0; r < nbins; ++r) {
        if (ring_cnt[r] > 0) {
            const double sum_power = static_cast<double>(ring_sum[r]) / 256.0; // undo FIXED_SCALE
            raw[r] = sum_power / static_cast<double>(ring_cnt[r]);
        }
    }

    cufftDestroy(plan);
    CUDA_CHECK(cudaFree(d_img));
    CUDA_CHECK(cudaFree(d_spec));
    CUDA_CHECK(cudaFree(d_power));
    CUDA_CHECK(cudaFree(d_ring_sum));
    CUDA_CHECK(cudaFree(d_ring_cnt));
}

// ===========================================================================
// fit_ctf_gpu  (stage 3): defocus grid-search, one thread per candidate.
// ===========================================================================
CtfFitResult fit_ctf_gpu(const std::vector<double>& prof, const CtfParams& optics,
                         const CtfFitConfig& cfg, float* kernel_ms) {
    CtfFitResult res;
    res.scores.assign(cfg.n_dz, -2.0);

    if (cfg.nbins > MAX_BINS) {
        std::fprintf(stderr, "[fit_ctf_gpu] nbins=%d exceeds MAX_BINS=%d\n", cfg.nbins, MAX_BINS);
        std::exit(EXIT_FAILURE);
    }

    // Upload the observed profile to CONSTANT memory (read by every search thread).
    CUDA_CHECK(cudaMemcpyToSymbol(c_profile, prof.data(),
                                  static_cast<std::size_t>(cfg.nbins) * sizeof(double)));

    double* d_scores = nullptr;                      // [n_dz] NCC per candidate
    CUDA_CHECK(cudaMalloc(&d_scores, static_cast<std::size_t>(cfg.n_dz) * sizeof(double)));

    const int    half      = optics.n / 2;
    const double nyquist_k = 0.5 / optics.pixel_size;
    const double dz_step   = (cfg.n_dz > 1)
                             ? (cfg.dz_max - cfg.dz_min) / (cfg.n_dz - 1) : 0.0;

    const int block = 128;
    const int grid  = (cfg.n_dz + block - 1) / block;

    GpuTimer timer;
    timer.start();
    ctf_search_kernel<<<grid, block>>>(cfg.n_dz, cfg.dz_min, dz_step,
                                       cfg.nbins, cfg.r_lo, cfg.r_hi,
                                       optics, half, nyquist_k, d_scores);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("ctf_search_kernel");

    CUDA_CHECK(cudaMemcpy(res.scores.data(), d_scores,
                          static_cast<std::size_t>(cfg.n_dz) * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_scores));

    // Host-side argmax (deterministic: first maximum wins on ties, == CPU).
    double best = -3.0; int bidx = -1;
    for (int i = 0; i < cfg.n_dz; ++i)
        if (res.scores[i] > best) { best = res.scores[i]; bidx = i; }
    res.best_idx = bidx;
    res.best_dz  = (bidx >= 0) ? dz_of_index(cfg, bidx) : 0.0;
    return res;
}
