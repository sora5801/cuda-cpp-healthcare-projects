// ===========================================================================
// src/main.cu  --  Entry point: load plan, run CPU + GPU dose, verify, report
// ---------------------------------------------------------------------------
// Project 5.3 : Proton & Heavy-Ion Therapy Dose
//
// WHAT THIS FILE DOES  (the 5-step shape EVERY project in this repo follows)
//   1. Load the treatment plan (grid + beam model + spots) from data/sample,
//      or fall back to a built-in synthetic plan if no file is given.
//   2. Compute the CPU reference dose volume (reference_cpu.cpp)   -> trusted.
//   3. Compute the GPU dose volume           (kernels.cu)          -> taught.
//   4. VERIFY: GPU agrees with CPU within a documented tolerance   -> correct.
//   5. REPORT: a deterministic central-axis depth-dose (Bragg) curve + the
//      recovered Bragg-peak depth to stdout; timings to stderr.
//
//   STDOUT is kept byte-for-byte deterministic so demo/run_demo can diff it
//   against demo/expected_output.txt. Anything that varies run-to-run (timings)
//   goes to STDERR, which the demo shows but does not diff (docs/PATTERNS.md §3).
//
//   NOT FOR CLINICAL USE. This is a reduced teaching model in arbitrary dose
//   units; it must never inform a real treatment (CLAUDE.md §1, §8).
//
// READ THIS FIRST in the code tour, then proton_physics.h -> kernels.cuh ->
// kernels.cu, and reference_cpu.cpp for the baseline. See ../THEORY.md for "why".
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // dose_gpu (GPU path), Plan/Grid/Spot via reference_cpu.h
#include "reference_cpu.h"    // load_plan, dose_cpu, voxel_index, Plan
#include "proton_physics.h"   // default_beam_model, Spot
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

// Program identity (kept in sync with demo/expected_output.txt).
static const char* PROJECT_ID   = "5.3";
static const char* PROJECT_NAME = "Proton & Heavy-Ion Therapy Dose";

// Correctness tolerance. CPU and GPU evaluate the SAME FP32 formula
// (proton_physics.h) summing spots in the SAME order, so they agree to a few
// ULPs; a single deposited spot involves expf/powf whose host vs. device
// implementations can differ by a hair. We verify to 1e-4 in ABSOLUTE dose
// units AND require the recovered peak-depth bins to match exactly. See
// THEORY.md §7 (why not bit-exact) and docs/PATTERNS.md §4.
static constexpr double TOLERANCE = 1.0e-4;

// ---------------------------------------------------------------------------
// make_synthetic_plan: the built-in fallback plan, IDENTICAL to what
// scripts/make_synthetic.py writes, so the program runs with zero arguments and
// still produces the exact expected_output.txt.
//
//   The default is the CLEANEST possible teaching case: a SINGLE on-axis pencil
//   beam of range 12 cm. Its central-axis depth-dose is the textbook PRISTINE
//   BRAGG CURVE -- a low, slowly rising entrance plateau, a sharp peak just
//   proximal to 12 cm, then a hard fall to zero distal to the range. That distal
//   zero is the whole point of proton therapy (spare tissue behind the target),
//   and with one spot the reported "Bragg-peak depth" is unambiguous. Stacking
//   several spots of different ranges into a flat-topped SPREAD-OUT BRAGG PEAK
//   (SOBP) is the natural next step -- left as an exercise and available via
//   scripts/make_synthetic.py --ranges (README §Exercises, THEORY §algorithm).
// ---------------------------------------------------------------------------
static Plan make_synthetic_plan() {
    Plan p;
    // Grid: a thin central "pencil" of voxels along depth. nx=ny=9 lateral
    // voxels (so an on-axis spot is sampled at the centre), nz=40 depth voxels of
    // 0.5 cm each => 20 cm deep. Origin centres the lateral extent on x=y=0.
    p.grid.nx = 9;  p.grid.ny = 9;  p.grid.nz = 40;
    p.grid.dx = 0.5f;
    p.grid.ox = -2.25f;  p.grid.oy = -2.25f;  p.grid.oz = 0.0f;  // 9*0.5=4.5 cm wide, centred
    p.beam    = default_beam_model();
    p.z_entry = 0.0f;                              // beam enters at the grid front face
    // One on-axis spot at range 12 cm => a single, unmistakable Bragg peak.
    p.spots.push_back(Spot{0.0f, 0.0f, 12.0f, 1.0f});
    return p;
}

// ---------------------------------------------------------------------------
// integral_depth_dose: the canonical depth-dose curve a medical physicist plots
// -- the dose SUMMED over the whole lateral plane at each depth slice k, i.e.
//     IDD(k) = sum over (i,j) of dose(i,j,k).
// WHY sum laterally instead of reading the central axis: a pencil beam spreads
// with depth (sigma grows), so its lateral Gaussian normalisation 1/(2*pi*sigma^2)
// makes the ON-AXIS value FALL with depth even where total energy per slice is
// rising -- that would flatten and distort the Bragg peak on the central axis.
// Integrating over the plane cancels the lateral normalisation (a Gaussian
// integrates to 1), leaving the pure DEPTH shape -- the clean Bragg peak. This
// is exactly the "integral depth dose" measured with a large parallel-plate
// chamber in a water tank (THEORY.md §science). Returns one value per depth slice.
// ---------------------------------------------------------------------------
static std::vector<float> integral_depth_dose(const Grid& g, const std::vector<float>& dose) {
    std::vector<float> idd(static_cast<std::size_t>(g.nz), 0.0f);
    for (int k = 0; k < g.nz; ++k) {
        // Sum the whole (i,j) plane for this depth. Fixed (i,j) order matches the
        // storage layout so the FP32 summation is deterministic and CPU==GPU.
        double acc = 0.0;                       // double accumulator: the plane sum is
                                                // a REPORTING quantity (not compared to
                                                // the GPU element-wise), so higher
                                                // precision here only sharpens the plot.
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
                acc += dose[voxel_index(g, i, j, k)];
        idd[static_cast<std::size_t>(k)] = static_cast<float>(acc);
    }
    return idd;
}

// argmax over a vector: index of the largest value (first on ties). Used to
// locate the Bragg-peak depth bin on the integral depth-dose curve.
static int argmax(const std::vector<float>& v) {
    int best = 0;
    for (int i = 1; i < static_cast<int>(v.size()); ++i)
        if (v[static_cast<std::size_t>(i)] > v[static_cast<std::size_t>(best)]) best = i;
    return best;
}

int main(int argc, char** argv) {
    // ---- 1. Load the plan ---------------------------------------------------
    Plan plan;
    std::string source;
    if (argc > 1) {
        try {
            plan   = load_plan(argv[1]);
            source = argv[1];
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[error] %s\n", e.what());
            return 2;
        }
    } else {
        plan   = make_synthetic_plan();
        source = "synthetic (built-in)";
    }
    const Grid& g = plan.grid;

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<float> dose_cpu_v;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    dose_cpu(plan, dose_cpu_v);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU dose (kernel timed inside the wrapper) ---------------------
    std::vector<float> dose_gpu_v;
    float gpu_kernel_ms = 0.0f;
    try {
        dose_gpu(plan, dose_gpu_v, &gpu_kernel_ms);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 4. Verify (absolute-dose tolerance + exact peak-bin match) --------
    // Primary check: every voxel of the GPU volume matches the CPU volume within
    // TOLERANCE (both are the SAME FP32 formula). Secondary, science-level check:
    // the integral-depth-dose Bragg peak lands in the same bin on both sides.
    const double err = util::max_abs_err(dose_cpu_v, dose_gpu_v);
    const std::vector<float> idd_cpu = integral_depth_dose(g, dose_cpu_v);
    const std::vector<float> idd_gpu = integral_depth_dose(g, dose_gpu_v);
    const int peak_cpu = argmax(idd_cpu);
    const int peak_gpu = argmax(idd_gpu);
    const bool pass = (err <= TOLERANCE) && (peak_cpu == peak_gpu);

    // Depth of the Bragg peak (voxel-centre depth of the peak slice). For a single
    // spot this should recover the input range to within half a voxel.
    const float peak_depth_cm = g.oz + (static_cast<float>(peak_gpu) + 0.5f) * g.dx - plan.z_entry;
    // Normalise the printed curve to its own maximum so the shape is scale-free
    // and reproducible regardless of the arbitrary dose units (peak prints 1.000).
    const float peak_val = idd_gpu[static_cast<std::size_t>(peak_gpu)];
    const float inv_peak = (peak_val > 0.0f) ? (1.0f / peak_val) : 0.0f;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("[teaching model: analytic pencil-beam dose; arbitrary units; not clinical]\n");
    std::printf("grid = %dx%dx%d voxels @ %.2f cm, entry z = %.2f cm, spots = %d\n",
                g.nx, g.ny, g.nz, g.dx, plan.z_entry, static_cast<int>(plan.spots.size()));
    std::printf("integral depth-dose (lateral sum per slice, normalized to Bragg peak):\n");
    // One line per depth slice: depth in cm + normalized dose (6 dp) + a tiny
    // ASCII bar so the Bragg peak is visible at a glance in the terminal.
    for (int k = 0; k < g.nz; ++k) {
        const float depth = g.oz + (static_cast<float>(k) + 0.5f) * g.dx - plan.z_entry;
        const float norm  = idd_gpu[static_cast<std::size_t>(k)] * inv_peak;
        int bar = static_cast<int>(norm * 40.0f + 0.5f);   // 0..40 char bar
        char buf[41];
        for (int c = 0; c < bar && c < 40; ++c) buf[c] = '#';
        buf[(bar < 40 ? bar : 40)] = '\0';
        std::printf("  z=%5.2f cm  %8.6f  %s\n", depth, norm, buf);
    }
    std::printf("Bragg-peak depth = %.2f cm (bin %d)\n", peak_depth_cm, peak_gpu);
    std::printf("RESULT: %s (GPU dose matches CPU within tol=1.0e-04; peak bins agree)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s\n", source.c_str());
    std::fprintf(stderr, "[timing] CPU dose: %.3f ms   GPU dose: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- tiny grids are launch/copy bound; "
                         "the GPU's edge grows with voxel*spot count (clinical: ~1e7 voxels x 1e4 spots).\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.3e (tol %.1e); peak bin CPU=%d GPU=%d\n",
                 err, TOLERANCE, peak_cpu, peak_gpu);

    return pass ? 0 : 1;
}
