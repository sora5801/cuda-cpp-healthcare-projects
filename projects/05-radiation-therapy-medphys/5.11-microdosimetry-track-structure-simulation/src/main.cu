// ===========================================================================
// src/main.cu  --  Entry point: run the track sim on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 5.11 : Microdosimetry & Track-Structure Simulation
//
// 5-step shape (the shape EVERY project in this repo follows):
//   1. Load the simulation parameters (data/sample).
//   2. CPU reference track-structure Monte Carlo (reference_cpu.cpp).
//   3. GPU track-structure Monte Carlo (kernels.cu) -- IDENTICAL tracks.
//   4. VERIFY: every integer tally matches EXACTLY (atomics commute on ints).
//   5. REPORT: deterministic microdosimetry + DNA-damage summary to stdout;
//              timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Timings (which vary run-to-run) go to STDERR,
//   shown but not diffed.
//
// Code tour: start here, then ts_physics.h (RNG + transport + DNA scoring),
// kernels.cu (the GPU twin), reference_cpu.cpp (the serial baseline). The
// science / GPU-mapping / verification is in ../THEORY.md.
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // track_gpu, TrackProblem, TrackTally
#include "reference_cpu.h"    // load_track_problem, track_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "5.11";
static const char* PROJECT_NAME = "Microdosimetry & Track-Structure Simulation";

// ---------------------------------------------------------------------------
// dose_mean_lineal_energy: the headline microdosimetric summary yD (keV/µm).
//   From a lineal-energy histogram f(y) (counts per bin), the dose-mean lineal
//   energy is yD = sum(y^2 f(y)) / sum(y f(y)). It weights the spectrum by dose
//   (each event contributes energy ~ y), so it is dominated by the high-y,
//   high-LET events that drive biological effectiveness. Computed from integer
//   bin counts + fixed bin centres, so it is a deterministic function of the
//   (exact) histogram -> identical for CPU and GPU.
// ---------------------------------------------------------------------------
static double dose_mean_lineal_energy(const std::vector<unsigned long long>& hist,
                                      double y_max, int n_bins) {
    const double bin_w = y_max / n_bins;         // width of one y bin (keV/µm)
    double num = 0.0, den = 0.0;                 // sum(y^2 f), sum(y f)
    for (int b = 0; b < n_bins; ++b) {
        const double y_c = (b + 0.5) * bin_w;    // bin-centre lineal energy
        const double f   = static_cast<double>(hist[b]);
        num += y_c * y_c * f;
        den += y_c * f;
    }
    return (den > 0.0) ? num / den : 0.0;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path =
        (argc > 1) ? argv[1] : "data/sample/track_params.txt";
    TrackProblem prob;
    try {
        prob = load_track_problem(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) ------------------------------------------
    TrackTally tally_c;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    track_cpu(prob, tally_c);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU track-structure Monte Carlo (kernel timed) -----------------
    TrackTally tally_g;
    float gpu_kernel_ms = 0.0f;
    track_gpu(prob, tally_g, &gpu_kernel_ms);

    // ---- 4. Verify (exact integer match across ALL tallies) ----------------
    int y_mismatches = 0;
    for (int b = 0; b < prob.tp.n_y_bins; ++b)
        if (tally_c.y_hist[b] != tally_g.y_hist[b]) ++y_mismatches;
    const bool pass =
        (tally_c.total_quanta == tally_g.total_quanta) &&
        (tally_c.total_ssb    == tally_g.total_ssb)    &&
        (tally_c.total_dsb    == tally_g.total_dsb)    &&
        (y_mismatches == 0);

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    // Derived, deterministic summaries from the (exact) GPU tally.
    const double total_keV = tally_g.total_quanta * prob.tp.quantum_eV / 1000.0;
    const double dsb_per_track =
        static_cast<double>(tally_g.total_dsb) / static_cast<double>(prob.n_tracks);
    // SSB:DSB ratio -- low-LET radiation makes mostly SSBs (high ratio), high-LET
    // makes clustered DSBs (low ratio). A classic radiobiology fingerprint.
    const double ssb_dsb_ratio =
        (tally_g.total_dsb > 0)
            ? static_cast<double>(tally_g.total_ssb) / static_cast<double>(tally_g.total_dsb)
            : 0.0;
    const double yD =
        dose_mean_lineal_energy(tally_g.y_hist, prob.tp.y_max_keV_um, prob.tp.n_y_bins);

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("water box %.1f nm, sigma_ion=%.3f /nm (LET spread %.2f), "
                "dna_radius=%.2f nm, %d DNA segments\n",
                prob.tp.box_nm, prob.tp.sigma_ion, prob.tp.let_spread,
                prob.tp.dna_radius_nm, prob.tp.n_dna_segments);
    std::printf("tracks = %llu, quantum = %.1f eV\n",
                prob.n_tracks, prob.tp.quantum_eV);
    std::printf("energy imparted = %llu quanta (%.3f keV total)\n",
                tally_g.total_quanta, total_keV);
    std::printf("DNA damage: SSB = %llu, DSB = %llu  (SSB/DSB = %.3f)\n",
                tally_g.total_ssb, tally_g.total_dsb, ssb_dsb_ratio);
    std::printf("DSB per track = %.4f\n", dsb_per_track);
    std::printf("dose-mean lineal energy yD = %.3f keV/um\n", yD);
    std::printf("lineal-energy spectrum f(y) (counts per bin):\n ");
    for (int b = 0; b < prob.tp.n_y_bins; ++b)
        std::printf(" %llu", tally_g.y_hist[b]);
    std::printf("\n");
    std::printf("RESULT: %s (GPU tallies match CPU exactly)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s\n", path.c_str());
    std::fprintf(stderr, "[timing] CPU MC: %.3f ms   GPU MC: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- speed-up grows with track "
                         "count; real nanodosimetry runs 1e6-1e8 tracks.\n");
    std::fprintf(stderr, "[verify] quanta match=%d ssb match=%d dsb match=%d "
                         "y-bin mismatches=%d (integer tallies => atomics commute)\n",
                 (tally_c.total_quanta == tally_g.total_quanta),
                 (tally_c.total_ssb == tally_g.total_ssb),
                 (tally_c.total_dsb == tally_g.total_dsb), y_mismatches);

    return pass ? 0 : 1;
}
