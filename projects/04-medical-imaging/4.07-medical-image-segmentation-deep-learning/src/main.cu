// ===========================================================================
// src/main.cu  --  Entry point: load volume, segment (CPU + GPU), verify, report
// ---------------------------------------------------------------------------
// Project 4.7 : Medical Image Segmentation (Deep Learning)   [REDUCED SCOPE]
//
// 5-step shape (every project follows it):
//   1. Load the synthetic CT-like volume + its known ground-truth lesion mask
//      (data/sample). The intensities drive the network; the mask scores it.
//   2. CPU reference: run the 2-layer segmentation head (reference_cpu.cpp).
//   3. GPU: run the SAME head with voxel-parallel kernels (kernels.cu).
//   4. VERIFY: GPU integer label map matches CPU EXACTLY; lesion logits match
//      within a small float tolerance.
//   5. REPORT: deterministic segmentation summary (voxel counts, Dice vs. ground
//      truth, a fixed-coordinate slice) to stdout; timing to stderr.
//
// Why a known ground-truth sphere: it lets us report a real accuracy number
// (Dice) and prove the "blob detector" actually finds the lesion, not just that
// CPU==GPU. The Dice is an exact integer-count metric, so stdout is reproducible.
//
// Code tour: start here, then kernels.cuh -> kernels.cu (the voxel-parallel
// 3D conv), and reference_cpu.{h,cpp} for the shared core. Science/GPU-mapping
// is in ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels.cuh"        // segment_gpu, Volume, SegNet
#include "reference_cpu.h"    // load_volume, make_segnet, segment_cpu, dice
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "4.7";
static const char* PROJECT_NAME = "Medical Image Segmentation (Deep Learning)";

// Float tolerance on the LESION-CLASS LOGITS. The integer label map must match
// the CPU EXACTLY (argmax of identical math), so it is checked with == ; the
// continuous logits can differ by ~1e-3 because the GPU fuses multiply-adds
// (FMA) while the host compiler may not. This is real and documented (PATTERNS
// §4): we verify the floats to a small tolerance and the labels exactly.
static constexpr double LOGIT_TOL = 1.0e-3;

// ---------------------------------------------------------------------------
// load_truth_mask: read the OPTIONAL ground-truth label block that our sample
//   appends after the intensity volume (data/README.md format):
//     D H W   <D*H*W intensities>   <D*H*W ground-truth labels 0/1>
//   We reopen the file, skip the header + intensities, and read n more ints.
//   Returns false (no ground truth) if the block is absent, so the program
//   still runs on a plain intensity-only volume.
// ---------------------------------------------------------------------------
static bool load_truth_mask(const std::string& path, const Volume& vol,
                            std::vector<int>& truth) {
    std::ifstream in(path);
    if (!in) return false;
    int D, H, W;
    if (!(in >> D >> H >> W)) return false;
    const long long n = vol.size();
    // Skip the n intensities we already loaded.
    float discard;
    for (long long i = 0; i < n; ++i) if (!(in >> discard)) return false;
    // Now try to read n ground-truth labels.
    truth.assign(static_cast<std::size_t>(n), 0);
    for (long long i = 0; i < n; ++i) {
        int label;
        if (!(in >> label)) return false;        // block absent/short -> no truth
        truth[static_cast<std::size_t>(i)] = (label != 0) ? 1 : 0;
    }
    return true;
}

int main(int argc, char** argv) {
    // ---- 1. Load the volume (+ optional ground-truth mask) -----------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/volume_sample.txt";
    Volume vol;
    try {
        vol = load_volume(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    std::vector<int> truth;
    const bool have_truth = load_truth_mask(path, vol, truth);

    const SegNet net = make_segnet();   // fixed, deterministic weights (no training)

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<int>   label_cpu;
    std::vector<float> logit_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    segment_cpu(vol, net, label_cpu, logit_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU (kernels timed inside the wrapper) -------------------------
    std::vector<int>   label_gpu;
    std::vector<float> logit_gpu;
    float gpu_kernel_ms = 0.0f;
    segment_gpu(vol, net, label_gpu, logit_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    // (a) Integer label maps must be IDENTICAL (exact check).
    long long label_mismatch = 0;
    for (std::size_t i = 0; i < label_cpu.size(); ++i)
        if (label_cpu[i] != label_gpu[i]) ++label_mismatch;
    // (b) Lesion logits must agree within LOGIT_TOL.
    double logit_err = 0.0;
    for (std::size_t i = 0; i < logit_cpu.size(); ++i) {
        const double d = std::fabs((double)logit_cpu[i] - (double)logit_gpu[i]);
        if (d > logit_err) logit_err = d;
    }
    const bool pass = (label_mismatch == 0) && (logit_err <= LOGIT_TOL);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    // Voxel counts (integer -> deterministic).
    long long pred_fg = 0;
    for (int c : label_gpu) pred_fg += (c != 0);

    std::printf("%s -- %s  [reduced-scope teaching version]\n", PROJECT_ID, PROJECT_NAME);
    std::printf("volume: %dx%dx%d (%lld voxels), 2-layer 3D conv head, %d classes\n",
                vol.D, vol.H, vol.W, vol.size(), N_CLASS);
    std::printf("predicted lesion voxels = %lld\n", pred_fg);
    if (have_truth) {
        long long true_fg = 0;
        for (int c : truth) true_fg += (c != 0);
        const double d = dice(label_gpu, truth);   // exact integer-count metric
        std::printf("ground-truth lesion voxels = %lld\n", true_fg);
        std::printf("Dice(prediction, ground truth) = %.4f\n", d);
    } else {
        std::printf("ground-truth lesion voxels = (none provided)\n");
        std::printf("Dice(prediction, ground truth) = n/a\n");
    }
    // A fixed mid-volume z-slice of the predicted mask: a deterministic visual
    // fingerprint of the segmentation (1 = lesion, . = background).
    const int zc = vol.D / 2;
    std::printf("predicted mask, central z=%d slice (1=lesion, .=bg):\n", zc);
    for (int y = 0; y < vol.H; ++y) {
        std::printf("  ");
        for (int x = 0; x < vol.W; ++x)
            std::printf("%c", label_gpu[vol.idx(zc, y, x)] ? '1' : '.');
        std::printf("\n");
    }
    std::printf("RESULT: %s (GPU label map == CPU exactly; logits within tol=1.0e-03)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%lld voxels%s)\n",
                 path.c_str(), vol.size(), have_truth ? ", with ground truth" : "");
    std::fprintf(stderr, "[timing] CPU segment: %.3f ms   GPU 2-layer conv: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- this toy volume is tiny and "
                         "launch-bound; the GPU's edge grows with real 512^3 CT volumes.\n");
    std::fprintf(stderr, "[verify] label mismatches = %lld (must be 0)   "
                         "max logit err = %.3e (tol %.1e)\n",
                 label_mismatch, logit_err, LOGIT_TOL);

    return pass ? 0 : 1;
}
