// ===========================================================================
// src/main.cu  --  Entry point: load variants, score, verify, rank, report
// ---------------------------------------------------------------------------
// Project 3.19 : Variant Effect / Pathogenicity Prediction
//
// THE 5-STEP SHAPE every project in this repo follows:
//   1. Load the problem (a batch of variants from data/sample).
//   2. CPU reference  (reference_cpu.cpp)  -> trusted per-variant delta scores.
//   3. GPU inference  (kernels.cu)         -> the thing being taught.
//   4. VERIFY: GPU agrees with CPU within tolerance.
//   5. REPORT: deterministic top-K most-pathogenic ranking to stdout; timing
//      and run-varying numbers to stderr.
//
// STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
// demo/expected_output.txt); run-to-run timings go to STDERR.
//
// Code tour: start here, then kernels.cuh -> kernels.cu (the GPU path) and
// vep_model.h (the shared math), then reference_cpu.* (the baseline + loader).
// ===========================================================================
#include <algorithm>
#include <cmath>      // std::fabs (verification: max abs error over doubles)
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "kernels.cuh"        // score_variants_gpu, VariantSet, VepModel
#include "reference_cpu.h"    // load_variants, init_model, score_variants_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "3.19";
static const char* PROJECT_NAME = "Variant Effect / Pathogenicity Prediction";

// Tolerance: CPU and GPU run the IDENTICAL double-precision forward pass (the
// shared vep_model.h core), so they differ only by the GPU's fused-multiply-add
// rounding vs. the host compiler -- well under 1e-9 for this short computation.
// 1e-9 is a strict, honest bound (PATTERNS.md sec 4: machine-precision class).
static constexpr double TOLERANCE = 1.0e-9;

// How many of the most-pathogenic-looking variants to print.
static constexpr int TOP_K = 5;

// Return the indices of the TOP_K LARGEST delta scores (most "pathogenic"),
// ties broken by lower index so the ranking is fully deterministic.
static std::vector<int> top_k(const std::vector<double>& effect, int k) {
    std::vector<int> idx(effect.size());
    std::iota(idx.begin(), idx.end(), 0);                 // 0,1,2,...,n-1
    const int kk = std::min<int>(k, static_cast<int>(idx.size()));
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
        [&](int a, int b) {
            if (effect[a] != effect[b]) return effect[a] > effect[b];  // higher first
            return a < b;                                             // tie -> lower idx
        });
    idx.resize(kk);
    return idx;
}

int main(int argc, char** argv) {
    // ---- 1. Load ------------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/variants_sample.txt";
    VariantSet vs;
    try {
        vs = load_variants(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // The fixed, synthetic model. init_model() is deterministic (no RNG at
    // runtime) so the scores -- and therefore stdout -- are reproducible.
    VepModel model;
    init_model(model);

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<double> effect_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    score_variants_cpu(model, vs, effect_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU inference (kernel timed inside the wrapper) ----------------
    std::vector<double> effect_gpu;
    float gpu_kernel_ms = 0.0f;
    score_variants_gpu(model, vs, effect_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ----------------------------------------------------------
    // We cannot reuse util::max_abs_err (it takes vector<float>); compute the
    // max absolute difference over the two double vectors directly.
    double err = 0.0;
    if (effect_cpu.size() != effect_gpu.size()) {
        err = 1.0e308;  // shape mismatch -> force FAIL
    } else {
        for (std::size_t i = 0; i < effect_cpu.size(); ++i) {
            const double d = std::fabs(effect_cpu[i] - effect_gpu[i]);
            if (d > err) err = d;
        }
    }
    const bool pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    // We rank on the CPU result so the printed ranking is independent of any GPU
    // FMA rounding (the two agree within 1e-9 anyway, but ranking on the trusted
    // baseline keeps stdout bit-stable across cards).
    const std::vector<int> best = top_k(effect_cpu, TOP_K);
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Batched in-silico mutagenesis: %d variants, %d-base context, "
                "delta = score(ALT) - score(REF)\n", vs.n, VEP_WINDOW);
    std::printf("top-%d most pathogenic-looking variants:\n",
                static_cast<int>(best.size()));
    for (std::size_t r = 0; r < best.size(); ++r) {
        const int v = best[r];
        // pos, REF>ALT substitution, and the model's delta score (6 decimals,
        // a deterministic, platform-stable amount of precision).
        std::printf("  #%zu  pos %d  %c>%c  delta = %+.6f\n",
                    r + 1, vs.pos[v],
                    base_char(vs.ref_base[v]), base_char(vs.alt_base[v]),
                    effect_cpu[v]);
    }
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-09)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (n=%d variants, window=%d)\n",
                 path.c_str(), vs.n, VEP_WINDOW);
    std::fprintf(stderr, "[model]  fixed synthetic CNN: K=%d filters of width %d, "
                         "global-max-pool, dense+sigmoid (NOT trained; no clinical meaning)\n",
                 VEP_KERNELS, VEP_KWIDTH);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny batch is dominated by "
                         "launch/copy overhead; the GPU wins at variant-atlas scale (millions).\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
