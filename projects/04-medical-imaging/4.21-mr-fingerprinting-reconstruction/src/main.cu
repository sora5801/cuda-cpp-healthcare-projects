// ===========================================================================
// src/main.cu  --  Entry point: load data, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 4.21 : MR Fingerprinting Reconstruction
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the MRF problem (schedule + dictionary grid + voxel signals).
//   2. CPU reference (reference_cpu.cpp): build dictionary -> normalize signals
//      -> per-voxel argmax-cosine match.
//   3. GPU result (kernels.cu): the same pipeline, with the V×D match matrix
//      formed by ONE cuBLAS SGEMM.
//   4. VERIFY: the GPU dictionary matches the CPU's within a float tolerance;
//      the per-voxel best-atom INDEX matches EXACTLY; the cosine scores agree
//      within a documented float tolerance.
//   5. REPORT: a DETERMINISTIC summary to stdout (accuracy vs the synthetic
//      ground truth, a few example voxels, the parameter-map ranges); timing
//      to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Anything that varies run-to-run (timings) goes to
//   STDERR, which the demo shows but does not diff (PATTERNS.md §3).
//
// THE SCIENCE IN ONE BREATH
//   MR Fingerprinting plays a pseudorandom pulse sequence so each tissue emits a
//   distinctive signal "fingerprint" over time. Reconstruction = match each
//   voxel's measured time course to a precomputed dictionary of simulated
//   fingerprints and read off the matched tissue's (T1, T2). The match is
//   ~10^11 inner products (every voxel vs every atom) -- one big matrix multiply
//   the GPU (cuBLAS SGEMM) does in a single call. Our synthetic phantom draws
//   each voxel from a KNOWN atom (plus noise), so a correct pipeline recovers
//   that atom -- the headline, human-meaningful result (PATTERNS.md §6).
//
// READ THIS FIRST in the code tour, then kernels.cuh -> kernels.cu, then
// reference_cpu.cpp for the baseline, and mrf_core.h for the shared math.
// See ../THEORY.md for the full "why".
// ===========================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // gpu_reconstruct, GpuTimings
#include "reference_cpu.h"    // load_problem, build/normalize/match references
#include "util/io.hpp"        // util::CpuTimer

// Identify the program. Must stay in sync with demo/expected_output.txt.
static const char* PROJECT_ID   = "4.21";
static const char* PROJECT_NAME = "MR Fingerprinting Reconstruction";

// ---- Verification tolerances (documented; PATTERNS.md §4) ------------------
// The dictionary and signal NORMALIZATION run the SAME shared double-precision
// math on both sides, so the normalized vectors agree to ~float epsilon. The
// MATCH scores differ slightly more: cuBLAS SGEMM sums each T-length inner
// product in a different ORDER (and uses fused multiply-add) than the CPU's
// serial loop, and float addition is not associative -- so cosines agree to
// ~1e-5, not bit-exactly. We therefore verify three things:
//   * DICT_TOL : the GPU-built dictionary matches the CPU's entrywise.
//   * SCORE_TOL: the winning cosine per voxel matches the CPU's.
//   * the best-atom INDEX matches EXACTLY (integer) for every voxel -- this is
//     the strong check, and it holds because the synthetic phantom is built so
//     each voxel's true atom wins by a comfortable margin (no near-ties).
static constexpr double DICT_TOL  = 1.0e-4;   // entrywise |dict_gpu - dict_cpu|
static constexpr double SCORE_TOL = 1.0e-4;   // per-voxel |cos_gpu - cos_cpu|

int main(int argc, char** argv) {
    // ---- 1. Load the MRF problem ------------------------------------------
    const std::string path =
        (argc > 1) ? argv[1] : "data/sample/mrf_sample.txt";
    MrfProblem p;
    try {
        p = load_problem(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int T = p.T, D = p.D, V = p.V;

    // ---- 2. CPU reference (timed) -----------------------------------------
    //   build dictionary -> normalize signals -> match, all serial and obvious.
    std::vector<float> dict_cpu, sig_norm_cpu, sig_scale_cpu;
    std::vector<MatchResult> match_cpu_res;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    build_dictionary_cpu(p, dict_cpu);
    normalize_signals_cpu(p, sig_norm_cpu, sig_scale_cpu);
    match_cpu(p, dict_cpu, sig_norm_cpu, sig_scale_cpu, match_cpu_res);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU result (kernels timed inside the wrapper) -----------------
    std::vector<MatchResult> match_gpu_res;
    std::vector<float> dict_gpu, score_row_v0_gpu;
    GpuTimings tm;
    gpu_reconstruct(p, match_gpu_res, dict_gpu, score_row_v0_gpu, &tm);

    // ---- 4. Verify GPU vs CPU ---------------------------------------------
    // (a) dictionary: worst entrywise difference over the whole D×T matrix.
    double dict_worst = 0.0;
    for (std::size_t i = 0; i < dict_cpu.size(); ++i)
        dict_worst = std::fmax(dict_worst,
                               std::fabs(static_cast<double>(dict_cpu[i]) - dict_gpu[i]));
    // (b) match: best-atom INDEX must agree exactly; track the worst cosine gap.
    int    index_mismatches = 0;
    double score_worst = 0.0;
    for (int v = 0; v < V; ++v) {
        if (match_cpu_res[v].best_atom != match_gpu_res[v].best_atom)
            ++index_mismatches;
        score_worst = std::fmax(score_worst,
            std::fabs(static_cast<double>(match_cpu_res[v].score) - match_gpu_res[v].score));
    }
    const bool pass = (dict_worst <= DICT_TOL) &&
                      (index_mismatches == 0) &&
                      (score_worst <= SCORE_TOL);

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    // Reconstruction accuracy against the synthetic ground truth: how often does
    // the matched atom equal the atom the voxel was drawn from? (PATTERNS.md §6)
    // We use the GPU results (verified == CPU above). Also report the median
    // T1/T2 absolute error over voxels whose truth atom is known.
    int   correct = 0, known = 0;
    std::vector<double> t1_err, t2_err;
    for (int v = 0; v < V; ++v) {
        int truth = p.truth_atom[v];
        if (truth < 0) continue;                       // unknown-truth voxel
        ++known;
        if (match_gpu_res[v].best_atom == truth) ++correct;
        t1_err.push_back(std::fabs(match_gpu_res[v].t1_ms - p.dict_t1[truth]));
        t2_err.push_back(std::fabs(match_gpu_res[v].t2_ms - p.dict_t2[truth]));
    }
    // Median error (deterministic: sort, take the middle). Empty -> 0.
    auto median = [](std::vector<double>& x) -> double {
        if (x.empty()) return 0.0;
        std::sort(x.begin(), x.end());
        std::size_t n = x.size();
        return (n & 1u) ? x[n / 2] : 0.5 * (x[n / 2 - 1] + x[n / 2]);
    };
    double med_t1 = median(t1_err);
    double med_t2 = median(t2_err);

    // Parameter-map ranges (min/max matched T1 and T2 over all voxels) -- a
    // compact, deterministic fingerprint of the reconstructed maps.
    double t1_min = 1e300, t1_max = -1e300, t2_min = 1e300, t2_max = -1e300;
    for (int v = 0; v < V; ++v) {
        t1_min = std::fmin(t1_min, match_gpu_res[v].t1_ms);
        t1_max = std::fmax(t1_max, match_gpu_res[v].t1_ms);
        t2_min = std::fmin(t2_min, match_gpu_res[v].t2_ms);
        t2_max = std::fmax(t2_max, match_gpu_res[v].t2_ms);
    }

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("problem: T=%d frames, D=%d dictionary atoms, V=%d voxels (synthetic)\n",
                T, D, V);
    std::printf("match accuracy: %d/%d voxels matched their ground-truth atom\n",
                correct, known);
    std::printf("median |T1 error| = %.3f ms ; median |T2 error| = %.3f ms\n",
                med_t1, med_t2);
    std::printf("recovered T1 map range: [%.1f, %.1f] ms ; T2 map range: [%.1f, %.1f] ms\n",
                t1_min, t1_max, t2_min, t2_max);

    // A few example voxels (first 5) so the learner can eyeball the matches.
    const int K = (V < 5) ? V : 5;
    std::printf("first %d voxels (voxel: truth_atom -> matched_atom  T1  T2  cos  PD):\n", K);
    for (int v = 0; v < K; ++v) {
        const MatchResult& r = match_gpu_res[v];
        std::printf("  v%02d: truth=%3d -> atom=%3d  T1=%6.1f  T2=%6.1f  cos=%.4f  PD=%.4f\n",
                    v, p.truth_atom[v], r.best_atom, r.t1_ms, r.t2_ms, r.score, r.pd);
    }
    std::printf("RESULT: %s (GPU dictionary + argmax match CPU; indices exact)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (T=%d, D=%d, V=%d)\n", path.c_str(), T, D, V);
    std::fprintf(stderr, "[timing] CPU reference (build+normalize+match): %.3f ms\n", cpu_ms);
    std::fprintf(stderr, "[timing] GPU build=%.3f  normsig=%.3f  cuBLAS SGEMM=%.3f  argmax=%.3f ms\n",
                 tm.build_ms, tm.normsig_ms, tm.sgemm_ms, tm.argmax_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this sample is tiny, so the GPU is "
                         "launch/copy bound; SGEMM's edge grows as O(V*D*T).\n");
    std::fprintf(stderr, "[verify] dictionary worst entry diff = %.3e  (tol %.1e)\n",
                 dict_worst, DICT_TOL);
    std::fprintf(stderr, "[verify] best-atom index mismatches  = %d / %d voxels\n",
                 index_mismatches, V);
    std::fprintf(stderr, "[verify] cosine worst diff           = %.3e  (tol %.1e)\n",
                 score_worst, SCORE_TOL);
    // Spot-check: the SGEMM's voxel-0 score row vs the CPU inner products.
    double row_worst = 0.0;
    for (int d = 0; d < D; ++d) {
        const float* sig0  = &sig_norm_cpu[0];
        const float* atomd = &dict_cpu[static_cast<std::size_t>(d) * T];
        double cpu_dot = 0.0;
        for (int t = 0; t < T; ++t) cpu_dot += static_cast<double>(sig0[t]) * atomd[t];
        row_worst = std::fmax(row_worst, std::fabs(cpu_dot - score_row_v0_gpu[d]));
    }
    std::fprintf(stderr, "[verify] voxel-0 SGEMM row vs CPU dot: worst = %.3e\n", row_worst);

    return pass ? 0 : 1;
}
