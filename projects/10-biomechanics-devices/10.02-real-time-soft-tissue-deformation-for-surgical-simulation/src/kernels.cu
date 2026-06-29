// ===========================================================================
// src/kernels.cu  --  PBD kernels (predict / project / finalize) + time loop
// ---------------------------------------------------------------------------
// Project 10.02 : Real-Time Soft-Tissue Deformation for Surgical Simulation
//
// GPU twin of simulate_cpu(): identical per-particle physics (pbd.h), one thread
// per particle, double-buffered Jacobi projection. main.cu compares the final
// mesh. See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

static constexpr int THREADS_PER_BLOCK = 256;

__global__ void predict_kernel(PbdParams P, const Vec3* __restrict__ x,
                               const Vec3* __restrict__ v, const double* __restrict__ w,
                               Vec3* __restrict__ pa) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P.R * P.C) return;
    pa[i] = pbd_predict(x[i], v[i], w[i], P.dt, P.gravity);
}

// One Jacobi iteration: every particle reads neighbours from `src` (read-only)
// and writes its corrected position to `dst`. Because reads are all from `src`,
// the particles are independent within the iteration -> no races, no atomics.
__global__ void constraint_kernel(PbdParams P, const Vec3* __restrict__ src,
                                  const double* __restrict__ w, Vec3* __restrict__ dst) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P.R * P.C) return;
    const int r = i / P.C;
    const int c = i % P.C;
    dst[i] = src[i] + pbd_correction(r, c, P, src, w);
}

__global__ void finalize_kernel(PbdParams P, const Vec3* __restrict__ p,
                                const double* __restrict__ w,
                                Vec3* __restrict__ x, Vec3* __restrict__ v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P.R * P.C) return;
    v[i] = pbd_new_velocity(p[i], x[i], w[i], P.dt);
    x[i] = p[i];
}

void simulate_gpu(const PbdParams& P, std::vector<Vec3>& x, std::vector<Vec3>& v,
                  const std::vector<double>& w, float* kernel_ms) {
    const int N = P.R * P.C;
    const std::size_t vbytes = static_cast<std::size_t>(N) * sizeof(Vec3);
    const std::size_t wbytes = static_cast<std::size_t>(N) * sizeof(double);

    Vec3 *d_x = nullptr, *d_v = nullptr, *d_pa = nullptr, *d_pb = nullptr;
    double* d_w = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, vbytes));
    CUDA_CHECK(cudaMalloc(&d_v, vbytes));
    CUDA_CHECK(cudaMalloc(&d_pa, vbytes));
    CUDA_CHECK(cudaMalloc(&d_pb, vbytes));
    CUDA_CHECK(cudaMalloc(&d_w, wbytes));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), vbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, v.data(), vbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, w.data(), wbytes, cudaMemcpyHostToDevice));

    const int grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    for (int step = 0; step < P.steps; ++step) {
        predict_kernel<<<grid, THREADS_PER_BLOCK>>>(P, d_x, d_v, d_w, d_pa);
        Vec3* src = d_pa;
        Vec3* dst = d_pb;
        for (int it = 0; it < P.iters; ++it) {
            constraint_kernel<<<grid, THREADS_PER_BLOCK>>>(P, src, d_w, dst);
            Vec3* tmp = src; src = dst; dst = tmp;          // ping-pong
        }
        finalize_kernel<<<grid, THREADS_PER_BLOCK>>>(P, src, d_w, d_x, d_v);
    }
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("pbd kernels");

    CUDA_CHECK(cudaMemcpy(x.data(), d_x, vbytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v.data(), d_v, vbytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_pa));
    CUDA_CHECK(cudaFree(d_pb));
    CUDA_CHECK(cudaFree(d_w));
}
