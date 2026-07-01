// ===========================================================================
// src/main.cu  --  Entry point: load protons, run SART on CPU + GPU, verify
// ---------------------------------------------------------------------------
// Project 5.15 : Proton CT & Ion Imaging Reconstruction
//
// 5-step shape (mirrors every project in this repo):
//   1. Load the list-mode proton data + geometry + ground truth (data/sample).
//   2a. CPU reference SART reconstruction (reference_cpu.cpp).
//   2b. GPU SART reconstruction (kernels.cu).
//   3. VERIFY: the GPU RSP image matches the CPU image within tolerance.
//   4. REPORT: deterministic RSP probes + recovery metrics -> stdout;
//      timing + run-varying detail -> stderr.
//
// STDOUT is kept byte-for-byte deterministic so demo/run_demo can diff it against
// demo/expected_output.txt; timings (which vary run to run) go to STDERR.
//
// The ground-truth RSP map (synthetic) is used ONLY for reporting how well the
// reconstruction recovered the known phantom -- never by the solver.
//
// READ THIS FIRST in the code tour, then kernels.cuh -> kernels.cu, and
// reference_cpu.cpp for the baseline. See ../THEORY.md for the "why".
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // reconstruct_gpu, PctProblem
#include "reference_cpu.h"    // load_pct, reconstruct_cpu
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

static const char* PROJECT_ID   = "5.15";
static const char* PROJECT_NAME = "Proton CT & Ion Imaging Reconstruction";

// TOLERANCE. The SART tally is fixed-point (order-independent), and the shared
// __host__ __device__ MLP + binning make CPU and GPU compute the same values.
// The only possible divergence is the float forward-projection sum rsp*seg_len,
// where host vs device FMA contraction can differ by ~1 ULP and, over `iters`
// sweeps, could nudge a voxel by a tiny amount (docs/PATTERNS.md section 4). We
// therefore verify to a small PHYSICAL tolerance in RSP units and say so plainly
// rather than pretending the two are bit-identical. In practice the observed
// error is far below this (printed on stderr).
static constexpr double TOLERANCE = 1.0e-3;

// mean RSP over voxels whose GROUND TRUTH exceeds a threshold (inside the
// phantom) -- one interpretable number for the demo output.
static double mean_inside(const std::vector<float>& img,
                          const std::vector<float>& truth, float thresh) {
    double sum = 0.0; long long cnt = 0;
    for (std::size_t i = 0; i < img.size(); ++i)
        if (truth[i] > thresh) { sum += img[i]; ++cnt; }
    return cnt ? sum / static_cast<double>(cnt) : 0.0;
}

// root-mean-square error of the reconstruction vs. the ground-truth phantom
// (over the whole image) -- "did we recover the known answer?" (PATTERNS 6).
static double rmse(const std::vector<float>& img, const std::vector<float>& truth) {
    double se = 0.0;
    for (std::size_t i = 0; i < img.size(); ++i) {
        const double d = static_cast<double>(img[i]) - static_cast<double>(truth[i]);
        se += d * d;
    }
    return img.empty() ? 0.0 : std::sqrt(se / static_cast<double>(img.size()));
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/protons_sample.txt";
    PctProblem prob;
    try {
        prob = load_pct(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int N     = prob.geom.n;
    const int cells = N * N;

    // ---- 2a. CPU reference SART (timed) -----------------------------------
    std::vector<float> img_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    reconstruct_cpu(prob, img_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 2b. GPU SART (kernel time summed over sweeps) --------------------
    std::vector<float> img_gpu;
    float gpu_kernel_ms = 0.0f;
    reconstruct_gpu(prob, img_gpu, &gpu_kernel_ms);

    // ---- 3. Verify GPU vs CPU ---------------------------------------------
    const double err  = util::max_abs_err(img_cpu, img_gpu);
    const bool   pass = err <= TOLERANCE;

    // ---- 4a. Deterministic report -> STDOUT -------------------------------
    // Probe voxels: image centre, and quarter/three-quarter along the diagonal.
    const int c  = (N / 2) * N + (N / 2);
    const int q  = (N / 4) * N + (N / 4);
    const int q3 = (3 * N / 4) * N + (3 * N / 4);

    // Recovery metrics on the CPU image (the trusted baseline; GPU matches it).
    const double mean_obj     = mean_inside(img_cpu, prob.truth, 0.5f);
    const double err_vs_truth = rmse(img_cpu, prob.truth);

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("list-mode SART: %d protons, %d sweeps, relax=%.2f, %d MLP samples/proton\n",
                static_cast<int>(prob.protons.size()), prob.iters,
                static_cast<double>(prob.relax), prob.path_samples);
    std::printf("grid: %dx%d voxels over world [%.2f,%.2f]^2 cm\n",
                N, N, -static_cast<double>(prob.geom.half),
                static_cast<double>(prob.geom.half));
    std::printf("reconstructed RSP: center=%.4f  q1=%.4f  q3=%.4f\n",
                img_cpu[c], img_cpu[q], img_cpu[q3]);
    std::printf("mean RSP inside phantom = %.4f\n", mean_obj);
    std::printf("RMSE vs ground-truth RSP = %.4f\n", err_vs_truth);
    std::printf("central row RSP profile (8 samples):");
    for (int s = 0; s < 8; ++s) {
        const int px = (s * (N - 1)) / 7;           // 8 evenly spaced columns
        std::printf(" %.4f", img_cpu[(N / 2) * N + px]);
    }
    std::printf("\n");
    std::printf("RESULT: %s (GPU matches CPU within tol=%.1e)\n",
                pass ? "PASS" : "FAIL", TOLERANCE);

    // ---- 4b. Run-varying detail -> STDERR ---------------------------------
    // Ground-truth mean inside the phantom, for context.
    double truth_sum = 0.0; long long truth_cnt = 0;
    for (int i = 0; i < cells; ++i)
        if (prob.truth[i] > 0.5f) { truth_sum += prob.truth[i]; ++truth_cnt; }
    const double truth_mean = truth_cnt ? truth_sum / static_cast<double>(truth_cnt) : 0.0;

    std::fprintf(stderr, "[data]   source: %s  (%d protons, %dx%d grid)\n",
                 path.c_str(), static_cast<int>(prob.protons.size()), N, N);
    std::fprintf(stderr, "[truth]  mean RSP inside phantom (ground truth) = %.4f\n", truth_mean);
    std::fprintf(stderr, "[timing] CPU SART: %.3f ms   GPU SART (kernels): %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the GPU's edge grows with proton count; "
                         "a clinical scan is ~10^8 protons vs the tiny sample here.\n");
    std::fprintf(stderr, "[verify] max_abs_err(GPU,CPU) = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
