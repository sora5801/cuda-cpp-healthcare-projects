// ===========================================================================
// src/main.cu  --  Entry: run BNCT MC on CPU + GPU, verify, report dose
// ---------------------------------------------------------------------------
// Project 5.13 : BNCT Dose Calculation & Optimization (reduced-scope teaching MC)
//
// 5-step shape (the shape EVERY project in this repo follows):
//   1. Load the BNCT problem (data/sample or the path given as argv[1]).
//   2. CPU reference Monte Carlo (reference_cpu.cpp)  -> trusted answer.
//   3. GPU Monte Carlo (kernels.cu) -- IDENTICAL histories (shared RNG+physics).
//   4. VERIFY: per-component integer dose tallies match EXACTLY (atomics commute
//      on integers), so the tolerance is ZERO.
//   5. REPORT: deterministic per-component depth-dose + CBE/RBE-weighted
//      biological dose to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Run-varying detail (timings) goes to STDERR.
//
// Code tour: start here, then bnct_physics.h (RNG + neutron transport),
// kernels.cu (GPU twin), reference_cpu.cpp (serial baseline). The science and
// GPU mapping are in ../THEORY.md.
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // dose_gpu, BnctProblem, DoseTally
#include "reference_cpu.h"    // load_bnct_problem, dose_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "5.13";
static const char* PROJECT_NAME = "BNCT Dose Calculation & Optimization";

// Human-readable component names, indexed by DoseComponent (fixed order).
static const char* COMP_NAME[DC_COUNT] = {
    "boron  (10B(n,a)7Li)",
    "nitro  (14N(n,p)14C)",
    "gamma  (1H(n,g)2H)  ",
    "fast   (recoil p)   ",
};

// Sum a component's integer keV tally over all depth bins.
static unsigned long long sum_component(const DoseTally& t, int c) {
    unsigned long long s = 0;
    for (unsigned long long v : t.dose[c]) s += v;
    return s;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/bnct_params.txt";
    BnctProblem prob;
    try {
        prob = load_bnct_problem(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    DoseTally tally_c;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    dose_cpu(prob, tally_c);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU Monte Carlo (kernel timed) --------------------------------
    DoseTally tally_g;
    float gpu_kernel_ms = 0.0f;
    dose_gpu(prob, tally_g, &gpu_kernel_ms);

    // ---- 4. Verify: EXACT per-component, per-bin integer match ------------
    unsigned long long mismatches = 0;
    for (int c = 0; c < DC_COUNT; ++c)
        for (int b = 0; b < prob.sp.n_bins; ++b)
            if (tally_c.dose[c][b] != tally_g.dose[c][b]) ++mismatches;
    const bool pass = (mismatches == 0);

    // ---- Derived quantities (computed from the GPU tally) ------------------
    // Per-component total keV, total physical dose, and the CBE/RBE-weighted
    // biological dose D_bio = sum_c w_c * D_c (weights are integer x1000).
    unsigned long long comp_keV[DC_COUNT];
    unsigned long long total_keV = 0;
    // Weighted biological "energy" in milli-keV-eq (integer, exact): sum of
    // keV * bio_weight_milli(c). We report it scaled to keV-Eq.
    unsigned long long bio_milli = 0;
    for (int c = 0; c < DC_COUNT; ++c) {
        comp_keV[c] = sum_component(tally_g, c);
        total_keV  += comp_keV[c];
        bio_milli  += comp_keV[c] * bio_weight_milli(c);
    }

    // Physical -> Gy scale (documented constant gray_per_keV from the sample).
    // These are teaching-scale numbers, NOT clinical dose. We print with fixed
    // precision so stdout is deterministic.
    const double phys_Gy = total_keV * prob.gray_per_keV;
    const double bio_GyEq = (bio_milli / 1000.0) * prob.gray_per_keV;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("REDUCED-SCOPE TEACHING MODEL (synthetic 1-D two-group MC; not clinical)\n");
    std::printf("slab L=%.1f cm, %d depth bins, histories=%llu\n",
                prob.sp.L, prob.sp.n_bins, prob.n_histories);
    std::printf("thermal Sigma_a (1/cm): B=%.4f N=%.4f H=%.4f  Sig_s_th=%.3f\n",
                prob.sp.Sig_a_B, prob.sp.Sig_a_N, prob.sp.Sig_a_H, prob.sp.Sig_s_th);
    std::printf("fast: Sig_s=%.3f /cm  p_thermalize=%.2f\n",
                prob.sp.Sig_s_fast, prob.sp.p_thermalize);

    // Per-component totals and their share of the physical dose (percent x10,
    // integer, so it is deterministic and rounding is explicit).
    std::printf("component totals (keV quanta and %% of physical dose):\n");
    for (int c = 0; c < DC_COUNT; ++c) {
        // tenths-of-percent via integer math: (comp*1000 + total/2) / total
        unsigned long long pct10 = total_keV
            ? (comp_keV[c] * 1000ULL + total_keV / 2) / total_keV : 0ULL;
        std::printf("  %s : %12llu  (%3llu.%llu%%)\n",
                    COMP_NAME[c], comp_keV[c], pct10 / 10, pct10 % 10);
    }
    std::printf("physical dose total = %.4f Gy (scale %.3e Gy/keV)\n",
                phys_Gy, prob.gray_per_keV);
    std::printf("CBE/RBE-weighted biological dose = %.4f Gy-Eq\n", bio_GyEq);

    // Boron depth-dose profile (the therapeutic component) -- the headline
    // curve. Integer keV per bin, so it is byte-deterministic.
    std::printf("boron depth-dose (keV per bin):\n ");
    for (int b = 0; b < prob.sp.n_bins; ++b) std::printf(" %llu", tally_g.dose[DC_BORON][b]);
    std::printf("\n");

    std::printf("RESULT: %s (GPU per-component dose tally matches CPU exactly)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s\n", path.c_str());
    std::fprintf(stderr, "[timing] CPU MC: %.3f ms   GPU MC: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- speed-up grows with history count; "
                         "clinical BNCT plans run 1e8-1e10 histories.\n");
    std::fprintf(stderr, "[verify] tally mismatches = %llu (integer keV => atomics commute; tol=0)\n",
                 mismatches);
    std::fprintf(stderr, "[note]   boron dose fraction = %.1f%% -- BNCT selectivity comes from "
                         "loading 10B into tumor cells (here Sigma_a_B is uniform for teaching).\n",
                 total_keV ? 100.0 * comp_keV[DC_BORON] / total_keV : 0.0);

    return pass ? 0 : 1;
}
