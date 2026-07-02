// ===========================================================================
// src/main.cu  --  Entry point: run the ABM on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 6.9 : Agent-Based Tissue / Immune Simulation
//
// 5-step shape:
//   1. Load the scenario parameters + generate the cell layout (data/sample).
//   2. CPU reference run  (reference_cpu.cpp).
//   3. GPU run            (kernels.cu) -- identical per-element physics (abm_core.h).
//   4. VERIFY: GPU field totals equal the CPU's EXACTLY (fixed-point quanta) and
//      the final cell positions agree within a tiny tolerance.
//   5. REPORT: a deterministic summary to stdout; timings to stderr.
//
// Biology of the demo: tumor cells at the centre secrete a chemokine that
// diffuses outward; immune cells CHEMOTAX up that gradient toward the tumor, so
// the mean immune->tumor distance SHRINKS over the run -- the "science" check.
//
// Code tour: start here, then abm_core.h (the shared physics), kernels.cu,
// reference_cpu.cpp. The full derivation is in ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // abm_gpu
#include "reference_cpu.h"    // load_abm, abm_cpu, AbmResult
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "6.9";
static const char* PROJECT_NAME = "Agent-Based Tissue / Immune Simulation";

// Position tolerance. CPU and GPU run the SAME double-precision operations in
// the SAME order, so they agree to ~machine precision; the GPU's fused multiply-
// add can differ from the host compiler by ~1e-13 per op, accumulating slightly
// over the steps. We verify positions to a physically-negligible 1e-6 domain
// units (sub-nanometre in tissue terms) and say so (PATTERNS.md §4). The chemokine
// FIELD is summed in integer quanta, so its total matches the CPU EXACTLY.
static constexpr double POS_TOL = 1.0e-6;

int main(int argc, char** argv) {
    // ---- 1. Load scenario + generate cells ---------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/tissue_params.txt";
    AbmParams p;
    Cells cells;
    try {
        p = load_abm(path, cells);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // Initial mean immune->tumor distance (for the "before vs after" science
    // check). summarize() with an empty field gives us this metric cheaply.
    const std::vector<double> empty_field(p.grid_cells(), 0.0);
    const AbmResult r0 = summarize(p, cells.x, cells.y, cells.type, empty_field);
    const double start_dist = r0.mean_immune_tumor_dist;

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<double> field_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    const AbmResult rc = abm_cpu(p, cells, field_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU run (loop timed with CUDA events) --------------------------
    std::vector<double> field_gpu;
    float gpu_ms = 0.0f;
    const AbmResult rg = abm_gpu(p, cells, field_gpu, &gpu_ms);

    // ---- 4. Verify ----------------------------------------------------------
    // (a) Chemokine field total: EXACT integer-quanta match (order-free atomics).
    const bool field_exact = (rc.total_quanta == rg.total_quanta);
    // (b) Final cell positions agree within POS_TOL.
    double pos_err = 0.0;
    for (int i = 0; i < cells.n; ++i) {
        pos_err = std::fmax(pos_err, std::fabs(rc.x[i] - rg.x[i]));
        pos_err = std::fmax(pos_err, std::fabs(rc.y[i] - rg.y[i]));
    }
    const bool pass = field_exact && (pos_err <= POS_TOL);

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    // We print the GPU summary; it equals the CPU one on a PASS. Distances are
    // printed at 6 decimals (deterministic across runs).
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("grid %dx%d (dx=%.2f), %d cells (%d tumor + %d immune), %d steps\n",
                p.gx, p.gy, p.dx, cells.n, rg.n_tumor, rg.n_immune, p.steps);
    std::printf("chemokine: total=%.6f  peak=%.6f at (col=%d,row=%d)\n",
                rg.total_chemokine, rg.peak_chemokine, rg.peak_col, rg.peak_row);
    std::printf("chemokine total quanta (exact integer): %llu\n", rg.total_quanta);
    std::printf("mean immune->tumor distance: start=%.6f  end=%.6f\n",
                start_dist, rg.mean_immune_tumor_dist);
    std::printf("RESULT: %s (GPU field total == CPU exactly; positions within tol=%.1e)\n",
                pass ? "PASS" : "FAIL", POS_TOL);

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s\n", path.c_str());
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU: %.3f ms\n", cpu_ms, gpu_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- on this tiny sample the GPU is "
                         "launch-bound (many small kernels + a host bin rebuild each step); "
                         "the GPU's edge grows with cell count and grid size.\n");
    std::fprintf(stderr, "[verify] field quanta: CPU=%llu GPU=%llu (%s)\n",
                 rc.total_quanta, rg.total_quanta, field_exact ? "exact" : "MISMATCH");
    std::fprintf(stderr, "[verify] max position diff = %.3e (tolerance %.1e)\n", pos_err, POS_TOL);

    return pass ? 0 : 1;
}
