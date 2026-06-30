// ===========================================================================
// src/main.cu  --  Entry point: run GIST on CPU + GPU, verify, report sites
// ---------------------------------------------------------------------------
// Project 2.26 : Hydrogen Bond Network & Water Placement Analysis
//
// 5-step shape (every project in this repo follows it):
//   1. LOAD the MD frames + solute + grid (data/sample, or arg path).
//   2. CPU reference GIST (reference_cpu.cpp)  -> trusted voxel tallies + ranking.
//   3. GPU GIST (kernels.cu): grid accumulation with atomic fixed-point updates.
//   4. VERIFY: per-voxel occupancy + fixed-point energy match EXACTLY (integer
//      atomics commute), and the ranked hydration-site list is identical.
//   5. REPORT: deterministic top hydration sites -> stdout; timing -> stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Run-varying numbers (timings) go to STDERR.
//
// Code tour: start here, then gist.h (the physics + fixed-point), reference_cpu.*
// (loader + serial baseline + shared reduce), kernels.cuh -> kernels.cu (the GPU
// scatter). See ../THEORY.md for the science and the GPU mapping.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // gist_gpu, Dataset, VoxelResult
#include "reference_cpu.h"    // load_dataset, gist_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "2.26";
static const char* PROJECT_NAME = "Hydrogen Bond Network & Water Placement Analysis";

// How many top-ranked hydration sites to print. Small + fixed so stdout is short
// and deterministic; the full list is available programmatically.
static constexpr int TOP_SITES = 8;

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1)
        ? argv[1]
        : "data/sample/water_sample.txt";
    Dataset d;
    try {
        d = load_dataset(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<unsigned int> counts_cpu;
    std::vector<gist_fixed_t> esum_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    const std::vector<VoxelResult> sites_cpu = gist_cpu(d, counts_cpu, esum_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU GIST (kernel timed inside the wrapper) --------------------
    std::vector<unsigned int> counts_gpu;
    std::vector<gist_fixed_t> esum_gpu;
    float gpu_kernel_ms = 0.0f;
    const std::vector<VoxelResult> sites_gpu = gist_gpu(d, counts_gpu, esum_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (exact: integer/fixed-point atomics commute) -----------
    // (a) Per-voxel raw tallies must match bit-for-bit.
    const int nv = d.grid.num_voxels();
    long long count_mismatch = 0;     // voxels whose occupancy differs
    long long esum_mismatch  = 0;     // voxels whose fixed-point energy differs
    for (int v = 0; v < nv; ++v) {
        if (counts_cpu[v] != counts_gpu[v]) ++count_mismatch;
        if (esum_cpu[v]   != esum_gpu[v])   ++esum_mismatch;
    }
    // (b) The ranked hydration-site lists must be identical (same length, same
    //     voxel order, same dG to within a hair -- they are computed from equal
    //     tallies so any diff would be a bug, not floating-point noise).
    bool ranking_match = (sites_cpu.size() == sites_gpu.size());
    double max_dG_diff = 0.0;
    if (ranking_match) {
        for (std::size_t i = 0; i < sites_cpu.size(); ++i) {
            if (sites_cpu[i].index != sites_gpu[i].index) { ranking_match = false; break; }
            max_dG_diff = std::fmax(max_dG_diff, std::fabs(sites_cpu[i].dG - sites_gpu[i].dG));
        }
    }
    const bool pass = (count_mismatch == 0) && (esum_mismatch == 0) && ranking_match;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) ----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("GIST grid: %dx%dx%d voxels @ %.2f A spacing  (%d voxels)\n",
                d.grid.nx, d.grid.ny, d.grid.nz, d.grid.spacing, nv);
    std::printf("samples: %d frames x %d waters = %lld water observations; %d solute atoms\n",
                d.nframes, d.waters_per_frame, d.num_samples(), d.natoms);
    std::printf("hydration sites (voxels with adequate occupancy): %zu\n", sites_gpu.size());
    std::printf("\n");
    // Header for the ranked table. Sites are ranked by OCCUPANCY (the water
    // occupancy map identifies them), and each is annotated with its GIST
    // thermodynamics. Columns:
    //   rank, (ix,iy,iz) voxel, occupancy count, density g, dE, -TdS, dG.
    std::printf("top %d hydration sites (ranked by occupancy; GIST dG = displaceability, kcal/mol):\n",
                TOP_SITES);
    std::printf("  rank  voxel(ix,iy,iz)   n      g      dE     -TdS      dG\n");
    const int show = (static_cast<int>(sites_gpu.size()) < TOP_SITES)
                     ? static_cast<int>(sites_gpu.size()) : TOP_SITES;
    for (int i = 0; i < show; ++i) {
        const VoxelResult& s = sites_gpu[i];
        // Recover (ix,iy,iz) from the flat index for a human-readable location.
        const int ix = s.index % d.grid.nx;
        const int iy = (s.index / d.grid.nx) % d.grid.ny;
        const int iz = s.index / (d.grid.nx * d.grid.ny);
        std::printf("  %3d   (%2d,%2d,%2d)      %5u  %5.2f  %6.2f  %6.2f  %6.2f\n",
                    i + 1, ix, iy, iz, s.count, s.g, s.dE, s.mTdS, s.dG);
    }
    std::printf("\n");
    std::printf("RESULT: %s (GPU voxel tallies + site ranking match CPU exactly)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) -----------------
    std::fprintf(stderr, "[data]   source: %s\n", path.c_str());
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernel: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- the GPU edge grows with the number of "
                         "(water x frame) samples; real GIST runs stream 10^6-10^9 of them.\n");
    std::fprintf(stderr, "[verify] count mismatches = %lld, fixed-point energy mismatches = %lld, "
                         "max dG diff = %.3e kcal/mol, ranking identical = %s\n",
                 count_mismatch, esum_mismatch, max_dG_diff, ranking_match ? "yes" : "no");

    return pass ? 0 : 1;
}
