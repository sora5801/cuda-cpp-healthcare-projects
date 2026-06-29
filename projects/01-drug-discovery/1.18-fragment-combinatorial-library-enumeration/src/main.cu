// ===========================================================================
// src/main.cu  --  Entry point: load synthons, enumerate (CPU+GPU), verify, report
// ---------------------------------------------------------------------------
// Project 1.18 : Fragment / Combinatorial Library Enumeration
//
// THE 5-STEP SHAPE every project in this repo follows:
//   1. Load the problem (a synthon catalog from data/sample).
//   2. CPU reference  (reference_cpu.cpp)  -> trusted enumeration result.
//   3. GPU enumeration (kernels.cu)        -> the thing being taught.
//   4. VERIFY: GPU agrees with CPU (counts EXACT; MW sum EXACT in fixed point).
//   5. REPORT: deterministic result to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
//   demo/expected_output.txt). Run-to-run timings go to STDERR (shown, not
//   diffed). The "first K passing products" preview is the same on both paths.
//
// Code tour: start here, then product_core.h (the shared math), kernels.cuh ->
// kernels.cu (the GPU path), then reference_cpu.* (the baseline). The "why" is
// in ../THEORY.md.
// ===========================================================================
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "kernels.cuh"        // enumerate_gpu (GPU path)
#include "reference_cpu.h"    // load_synthons, enumerate_cpu, product_label
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.18";
static const char* PROJECT_NAME = "Fragment / Combinatorial Library Enumeration";

// Default catalog path if none is given on the command line.
static const char* DEFAULT_SAMPLE = "data/sample/synthons_sample.txt";

// ---------------------------------------------------------------------------
// Verification: the GPU and CPU run identical integer/fixed-point reductions, so
// they must agree EXACTLY. We compare the three deterministic fields field by
// field and return true only on a perfect match.
//   This is the strongest tolerance possible ("== 0") and the right one here:
//   the count is integer, the MW sum is fixed-point integer, and the first-K
//   indices are the same canonical prefix -- no floating-point rounding enters
//   the comparison (PATTERNS.md sec.4: exact when both sides do integer work).
// ---------------------------------------------------------------------------
static bool results_agree(const EnumResult& cpu, const EnumResult& gpu) {
    if (cpu.n_pass != gpu.n_pass) return false;
    if (cpu.sum_mw_pass_milli != gpu.sum_mw_pass_milli) return false;
    if (cpu.first_pass.size() != gpu.first_pass.size()) return false;
    for (std::size_t i = 0; i < cpu.first_pass.size(); ++i)
        if (cpu.first_pass[i] != gpu.first_pass[i]) return false;
    return true;
}

int main(int argc, char** argv) {
    // ---- 1. Load the synthon catalog ---------------------------------------
    const std::string path = (argc > 1) ? argv[1] : DEFAULT_SAMPLE;
    SynthonLibrary lib;
    try {
        lib = load_synthons(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int64_t N = lib.num_products();

    // ---- 2. CPU reference (timed) ------------------------------------------
    EnumResult res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    enumerate_cpu(lib, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU enumeration (kernel timed inside the wrapper) --------------
    EnumResult res_gpu;
    float gpu_kernel_ms = 0.0f;
    try {
        enumerate_gpu(lib, res_gpu, &gpu_kernel_ms);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] GPU path failed: %s\n", e.what());
        return 2;
    }

    // ---- 4. Verify ----------------------------------------------------------
    const bool pass = results_agree(res_cpu, res_gpu);

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    // Compute the pass fraction as a percentage with one decimal -- still
    // deterministic because n_pass and N are integers (fixed formatting).
    const double pct = (N > 0) ? (100.0 * static_cast<double>(res_gpu.n_pass) / static_cast<double>(N)) : 0.0;
    // The summed MW comes back in milli-g/mol; print g/mol with 3 decimals.
    const double sum_mw_g = static_cast<double>(res_gpu.sum_mw_pass_milli)
                          / static_cast<double>(MW_FIXED_SCALE);

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("combinatorial library: %d x %d x %d slots = %lld products\n",
                lib.sizes[0], lib.sizes[1], lib.sizes[2], static_cast<long long>(N));
    std::printf("drug-like (Lipinski+Veber) passes: %lld / %lld  (%.1f%%)\n",
                static_cast<long long>(res_gpu.n_pass), static_cast<long long>(N), pct);
    std::printf("sum of MW over passing products: %.3f g/mol\n", sum_mw_g);
    std::printf("first %d passing products (by index):\n",
                static_cast<int>(res_gpu.first_pass.size()));
    for (std::size_t r = 0; r < res_gpu.first_pass.size(); ++r) {
        const int64_t p = res_gpu.first_pass[r];
        std::printf("  #%zu  product[%lld]  = %s\n",
                    r + 1, static_cast<long long>(p), product_label(lib, p).c_str());
    }
    std::printf("RESULT: %s (GPU matches CPU exactly: count, MW-sum, indices)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d slots, sizes %d/%d/%d)\n",
                 path.c_str(), N_SLOTS, lib.sizes[0], lib.sizes[1], lib.sizes[2]);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny sample is dominated "
                         "by launch/copy overhead; the GPU wins at library scale (billions).\n");
    std::fprintf(stderr, "[verify] CPU n_pass=%lld  GPU n_pass=%lld  (exact-match check)\n",
                 static_cast<long long>(res_cpu.n_pass), static_cast<long long>(res_gpu.n_pass));

    return pass ? 0 : 1;
}
