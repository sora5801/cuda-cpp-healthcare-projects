// ===========================================================================
// src/main.cu  --  Entry point: load data, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 2.3 : Cryo-EM Single-Particle Reconstruction  (reduced-scope, 2D)
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the problem (a synthetic single-particle dataset from data/sample).
//   2. Compute the CPU reference (reference_cpu.cpp)         -> trusted answer.
//   3. Compute the GPU result    (kernels.cu)                -> the thing taught.
//   4. VERIFY: assert GPU agrees with CPU exactly             -> correctness.
//   5. REPORT: deterministic result to stdout; timing to stderr.
//
//   THE PIPELINE (two stages, mirroring real cryo-EM):
//     E-step  : projection MATCHING  -- assign each particle its best ref angle.
//     M-step  : back-PROJECTION      -- smear the assigned profiles into 2D.
//   We report three teaching metrics, all DETERMINISTIC:
//     * matching accuracy  : fraction of particles whose recovered angle equals
//                            the (synthetic) ground-truth angle index. This shows
//                            projection matching actually recovers orientations.
//     * reconstruction NCC : correlation of the reconstructed density with the
//                            known ground-truth image -> the reconstruction works.
//     * a 5-number density digest (corner/centre samples) -> a stable fingerprint.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Timings (which vary) go to STDERR (shown, not
//   diffed). All reported floats are printed at fixed precision.
//
// READ THIS FIRST in the code tour, then kernels.cuh -> kernels.cu, and
// reference_cpu.cpp for the baseline. See ../THEORY.md for the "why".
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // match_gpu, reconstruct_gpu (GPU path)
#include "reference_cpu.h"    // Dataset, match_cpu, reconstruct_cpu (CPU baseline)
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

static const char* PROJECT_ID   = "2.3";
static const char* PROJECT_NAME = "Cryo-EM Single-Particle Reconstruction";

// Verification tolerances.
//   * Assignments are INTEGER angle indices -> must match EXACTLY (0 mismatches).
//   * Both the CPU and GPU densities are produced by the SAME __host__ __device__
//     backproject_pixel() summed in the SAME particle order, so they agree to
//     near machine precision. We allow a tiny 1e-4 slack only to absorb the
//     compiler's freedom to contract a*b+c into an FMA differently on host vs.
//     device; in practice the error is ~1e-6 (see THEORY §"verification").
static constexpr double RECON_TOLERANCE = 1.0e-4;

// ---------------------------------------------------------------------------
// image_ncc: normalized cross-correlation between a reconstructed density and
//   the ground-truth image, in [-1, 1]. This is the SCIENCE check (not just
//   CPU==GPU): does the pipeline actually recover the molecule? Computed in
//   double for a stable, deterministic single number.
// ---------------------------------------------------------------------------
static double image_ncc(const std::vector<float>& recon, const std::vector<float>& truth) {
    const std::size_t m = recon.size();
    double mr = 0.0, mt = 0.0;
    for (std::size_t k = 0; k < m; ++k) { mr += recon[k]; mt += truth[k]; }
    mr /= static_cast<double>(m); mt /= static_cast<double>(m);
    double dot = 0.0, nr = 0.0, nt = 0.0;
    for (std::size_t k = 0; k < m; ++k) {
        const double dr = recon[k] - mr, dt = truth[k] - mt;
        dot += dr * dt; nr += dr * dr; nt += dt * dt;
    }
    const double denom = std::sqrt(nr * nt);
    return (denom > 0.0) ? (dot / denom) : 0.0;
}

int main(int argc, char** argv) {
    // ---- 1. Load the problem -----------------------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/cryoem_sample.txt";
    Dataset ds;
    try {
        ds = load_dataset(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<int>   assign_cpu;
    std::vector<float> score_cpu, recon_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    match_cpu(ds, assign_cpu, score_cpu);                 // E-step
    reconstruct_cpu(ds, assign_cpu, recon_cpu);           // M-step
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU result (each kernel timed inside its wrapper) --------------
    std::vector<int>   assign_gpu;
    std::vector<float> score_gpu, recon_gpu;
    float match_ms = 0.0f, recon_ms = 0.0f;
    match_gpu(ds, assign_gpu, score_gpu, &match_ms);       // E-step on GPU
    reconstruct_gpu(ds, assign_gpu, recon_gpu, &recon_ms); // M-step on GPU

    // ---- 4. Verify GPU vs CPU ----------------------------------------------
    // (a) Assignments: every angle index must match exactly (integer compare).
    int assign_mismatch = 0;
    for (int i = 0; i < ds.n_particles; ++i)
        if (assign_cpu[i] != assign_gpu[i]) ++assign_mismatch;
    // (b) Reconstruction: GPU density must match the CPU density within tol.
    const double recon_err = util::max_abs_err(recon_cpu, recon_gpu);
    const bool pass = (assign_mismatch == 0) && (recon_err <= RECON_TOLERANCE);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    // Projection-matching accuracy: how many particles recovered the TRUE angle
    // index (synthetic ground truth). Integer counts -> fully deterministic.
    int correct = 0;
    for (int i = 0; i < ds.n_particles; ++i)
        if (assign_gpu[i] == ds.true_angle[i]) ++correct;
    const double accuracy = 100.0 * correct / ds.n_particles;

    // Reconstruction quality: correlation with the known molecule (the science).
    const double rec_ncc = image_ncc(recon_gpu, ds.true_img);

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("2D single-particle reconstruction (synthetic phantom)\n");
    std::printf("geometry: image %dx%d, %d reference angles, %d particles\n",
                IMG_SIZE, IMG_SIZE, N_ANGLES, ds.n_particles);
    std::printf("E-step (projection matching, O(N*M)=%d comparisons):\n",
                ds.n_particles * N_ANGLES);
    std::printf("  orientation recovery accuracy = %.1f%% (%d/%d exact-angle hits)\n",
                accuracy, correct, ds.n_particles);
    std::printf("M-step (back-projection into %dx%d density):\n", IMG_SIZE, IMG_SIZE);
    std::printf("  reconstruction-vs-truth NCC = %.4f\n", rec_ncc);
    // A small stable digest of the density so the demo pins the actual numbers
    // (centre pixel + four sampled interior points). Fixed precision.
    const int c = IMG_SIZE / 2;
    std::printf("  density digest: centre=%.4f  q1=%.4f  q2=%.4f  q3=%.4f  q4=%.4f\n",
                recon_gpu[c * IMG_SIZE + c],
                recon_gpu[(IMG_SIZE/4) * IMG_SIZE + (IMG_SIZE/4)],
                recon_gpu[(IMG_SIZE/4) * IMG_SIZE + (3*IMG_SIZE/4)],
                recon_gpu[(3*IMG_SIZE/4) * IMG_SIZE + (IMG_SIZE/4)],
                recon_gpu[(3*IMG_SIZE/4) * IMG_SIZE + (3*IMG_SIZE/4)]);
    std::printf("RESULT: %s (GPU matches CPU: %d/%d assignments exact, density within tol=1.0e-04)\n",
                pass ? "PASS" : "FAIL", ds.n_particles - assign_mismatch, ds.n_particles);

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (n=%d particles)\n", path.c_str(), ds.n_particles);
    std::fprintf(stderr, "[timing] CPU (match+recon): %.3f ms   GPU match: %.3f ms   GPU recon: %.3f ms\n",
                 cpu_ms, match_ms, recon_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny sample is dominated by "
                         "launch/copy overhead; the GPU wins when N*M is large (millions of particles).\n");
    std::fprintf(stderr, "[verify] assignment mismatches = %d   recon max_abs_err = %.3e  (tol %.1e)\n",
                 assign_mismatch, recon_err, RECON_TOLERANCE);

    return pass ? 0 : 1;
}
