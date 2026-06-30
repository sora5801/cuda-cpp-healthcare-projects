// ===========================================================================
// src/main.cu  --  Entry point: load protein, scan, verify, report
// ---------------------------------------------------------------------------
// Project 2.16 : ΔΔG Stability Prediction (reduced-scope teaching version)
//
// The 5-step shape every project in this repo follows:
//   1. Load the problem (a protein: wild-type residues + burial fractions).
//   2. CPU reference  (reference_cpu.cpp)  -> trusted ΔΔG scan grid.
//   3. GPU scan       (kernels.cu)         -> the thing being taught.
//   4. VERIFY: GPU agrees with CPU within a documented tolerance.
//   5. REPORT: deterministic findings to stdout; timing + error to stderr.
//
// STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
// demo/expected_output.txt); run-to-run timings go to STDERR.
//
// Code tour: start here, then ddg_model.h (the science), then reference_cpu.*,
// then kernels.cuh -> kernels.cu. The "why" is in ../THEORY.md.
// ===========================================================================
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "kernels.cuh"        // ddg_scan_gpu, MAX_RESIDUES
#include "reference_cpu.h"    // load_protein, ddg_scan_cpu, Protein, NUM_AA
#include "ddg_model.h"        // AA_ONE_LETTER (for printing mutant codes)
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

static const char* PROJECT_ID   = "2.16";
static const char* PROJECT_NAME = "Delta-Delta-G Stability Prediction";

// Tolerance: CPU and GPU evaluate the IDENTICAL ddg_predict(); they differ only
// in the last bits of tanhf (device intrinsic vs host <cmath>) and FMA
// contraction. That divergence is ~1e-5 kcal/mol -- physically negligible. We
// verify to 1e-3 kcal/mol and say so (PATTERNS.md §4, "small physical
// tolerance"; the full reasoning is in ../THEORY.md "Numerical considerations").
static constexpr double TOLERANCE = 1.0e-3;   // kcal/mol
static constexpr int    TOP_K     = 5;        // how many extreme mutations to list

// A single mutation result, used for the deterministic "most destabilising" list.
struct MutHit {
    int   pos;     // 0-based residue position
    int   wt;      // wild-type amino-acid index
    int   mut;     // mutant amino-acid index
    float ddg;     // predicted ΔΔG (kcal/mol)
};

// Return the TOP_K most DESTABILISING mutations (smallest ΔΔG = most negative),
// ties broken by (position, mutant index) so the ranking is fully deterministic.
// We build the candidate list excluding self-mutations (mut == wt, ΔΔG == 0).
static std::vector<MutHit> most_destabilising(const Protein& prot,
                                              const std::vector<float>& grid, int k) {
    std::vector<MutHit> hits;
    hits.reserve(static_cast<std::size_t>(prot.L) * NUM_AA);
    for (int p = 0; p < prot.L; ++p) {
        const int wt = prot.wt_code[static_cast<std::size_t>(p)];
        for (int a = 0; a < NUM_AA; ++a) {
            if (a == wt) continue;                       // skip the trivial self-mutation
            hits.push_back(MutHit{p, wt, a,
                                  grid[static_cast<std::size_t>(p) * NUM_AA + a]});
        }
    }
    const int kk = std::min<int>(k, static_cast<int>(hits.size()));
    std::partial_sort(hits.begin(), hits.begin() + kk, hits.end(),
        [](const MutHit& x, const MutHit& y) {
            if (x.ddg != y.ddg) return x.ddg < y.ddg;    // more negative first
            if (x.pos != y.pos) return x.pos < y.pos;    // tie -> lower position
            return x.mut < y.mut;                        // tie -> lower mutant idx
        });
    hits.resize(kk);
    return hits;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/protein_sample.txt";
    Protein prot;
    try {
        prot = load_protein(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    if (prot.L > MAX_RESIDUES) {
        std::fprintf(stderr, "[error] protein has %d residues but this build caps at "
                             "MAX_RESIDUES=%d (raise it in kernels.cuh).\n", prot.L, MAX_RESIDUES);
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<float> grid_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    ddg_scan_cpu(prot, grid_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU scan (kernel timed inside the wrapper) --------------------
    std::vector<float> grid_gpu;
    float gpu_kernel_ms = 0.0f;
    ddg_scan_gpu(prot, grid_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    const double err  = util::max_abs_err(grid_cpu, grid_gpu);
    const bool   pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    // Report is computed from the GPU grid (the thing under test); because it
    // agrees with the CPU within tolerance, the printed values are stable.
    const std::vector<MutHit> worst = most_destabilising(prot, grid_gpu, TOP_K);

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Saturation mutagenesis scan: protein '%s', %d residues x %d AA = %d mutations\n",
                prot.name.c_str(), prot.L, NUM_AA, prot.L * NUM_AA);
    std::printf("top-%d most DESTABILISING mutations (most negative ddG):\n",
                static_cast<int>(worst.size()));
    for (std::size_t r = 0; r < worst.size(); ++r) {
        // Human-readable mutation code, e.g. "L12P": <wt><1-based pos><mut>.
        std::printf("  #%zu  %c%d%c   ddG = %+8.4f kcal/mol\n",
                    r + 1,
                    AA_ONE_LETTER[worst[r].wt],
                    worst[r].pos + 1,
                    AA_ONE_LETTER[worst[r].mut],
                    worst[r].ddg);
    }
    // A couple of deterministic aggregate statistics over the full grid (these
    // recover the synthetic "buried hydrophobic core" signal -- see README).
    double sum = 0.0; int destab = 0, mutations = 0;
    for (int p = 0; p < prot.L; ++p) {
        const int wt = prot.wt_code[static_cast<std::size_t>(p)];
        for (int a = 0; a < NUM_AA; ++a) {
            if (a == wt) continue;
            const float d = grid_gpu[static_cast<std::size_t>(p) * NUM_AA + a];
            sum += d; ++mutations;
            if (d < 0.0f) ++destab;
        }
    }
    std::printf("summary: %d of %d non-self mutations are destabilising (ddG<0); mean ddG = %+.4f kcal/mol\n",
                destab, mutations, sum / (mutations ? mutations : 1));
    std::printf("RESULT: %s (GPU matches CPU within tol=%.1e kcal/mol)\n",
                pass ? "PASS" : "FAIL", TOLERANCE);

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (protein '%s', L=%d residues)\n",
                 path.c_str(), prot.name.c_str(), prot.L);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny scan is dominated by "
                         "launch/copy overhead; the GPU's edge grows with protein length and "
                         "with a real (heavier) per-mutation model.\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.3e kcal/mol  (tolerance %.1e)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
