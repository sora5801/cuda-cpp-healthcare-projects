// ===========================================================================
// src/main.cu  --  Entry point: load volume, march cubes on CPU+GPU, verify
// ---------------------------------------------------------------------------
// Project 4.18 : Image-Based 3D Printing / Model Generation for Surgery
//
// 5-step shape (the same skeleton every project in this repo uses):
//   1. Load the scalar volume + iso-value (data/sample).
//   2. CPU reference marching cubes (reference_cpu.cpp)  -- the trusted mesh.
//   3. GPU marching cubes (kernels.cu)                   -- count/scan/generate.
//   4. VERIFY: GPU mesh matches CPU mesh (same #triangles, vertices within tol).
//   5. REPORT: deterministic mesh stats to stdout; timing to stderr.
//
// The headline teaching artifact is that two completely different execution
// models -- a serial CPU loop and a massively-parallel GPU pipeline with a
// prefix-sum in the middle -- produce the SAME triangle mesh, because they share
// the per-cube math in mc_core.h.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "kernels.cuh"       // marching_cubes_gpu
#include "reference_cpu.h"   // load_problem, marching_cubes_cpu, mesh_* metrics
#include "util/io.hpp"       // util::CpuTimer

static const char* PROJECT_ID   = "4.18";
static const char* PROJECT_NAME = "Image-Based 3D Printing / Model Generation for Surgery";

// Vertex agreement tolerance (world units = mm here). The CPU and GPU run the
// SAME float interpolation (mc_core.h interp_edge), so in principle the meshes
// are bit-identical; we allow a tiny 1e-3 mm slack to absorb the GPU's fused
// multiply-add (FMA) contracting `p0 + t*(p1-p0)` differently from the host
// compiler. That is far below any printable resolution (FDM layers are ~0.1 mm),
// so it is physically meaningless -- documented honestly per docs/PATTERNS.md §4.
static constexpr double TOLERANCE = 1.0e-3;

// ---------------------------------------------------------------------------
// max_vertex_err: the largest per-coordinate difference between two meshes.
//   Returns +inf if the meshes differ in triangle count (a topology mismatch is
//   worse than any coordinate slop and must never be mistaken for agreement).
//   The meshes are in the SAME order (both ascending cell index), so triangle i
//   on the CPU corresponds to triangle i on the GPU -- a direct comparison.
// ---------------------------------------------------------------------------
static double max_vertex_err(const std::vector<Triangle>& a,
                             const std::vector<Triangle>& b) {
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double worst = 0.0;
    for (std::size_t t = 0; t < a.size(); ++t) {
        for (int v = 0; v < 3; ++v) {
            worst = std::fmax(worst, std::fabs((double)a[t].v[v].x - b[t].v[v].x));
            worst = std::fmax(worst, std::fabs((double)a[t].v[v].y - b[t].v[v].y));
            worst = std::fmax(worst, std::fabs((double)a[t].v[v].z - b[t].v[v].z));
        }
    }
    return worst;
}

int main(int argc, char** argv) {
    // ---- 1. Load ----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/volume_sample.txt";
    MCProblem prob;
    try {
        prob = load_problem(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const VolDims& d = prob.dims;

    // ---- 2. CPU reference (timed) ----------------------------------------
    std::vector<Triangle> mesh_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    marching_cubes_cpu(prob, mesh_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU pipeline (kernel timed) ----------------------------------
    std::vector<Triangle> mesh_gpu;
    float gpu_kernel_ms = 0.0f;
    marching_cubes_gpu(prob, mesh_gpu, &gpu_kernel_ms);

    // ---- 4. Verify --------------------------------------------------------
    const double err = max_vertex_err(mesh_cpu, mesh_gpu);
    const bool pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT ------------------------------
    // All numbers below are pure functions of the (committed) input volume and
    // the shared math, so this block is byte-identical on every run/machine.
    const double area = mesh_surface_area(mesh_gpu);
    Vec3 lo, hi;
    mesh_bbox(mesh_gpu, lo, hi);
    const double checksum = mesh_checksum(mesh_gpu);

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("volume: %d x %d x %d samples, spacing %.3f mm, iso = %.3f\n",
                d.nx, d.ny, d.nz, d.spacing, prob.iso);
    std::printf("cells marched: %d\n", num_cells(d));
    std::printf("triangles extracted: %zu\n", mesh_gpu.size());
    std::printf("surface area: %.4f mm^2\n", area);
    std::printf("bbox min: (%.4f, %.4f, %.4f) mm\n", lo.x, lo.y, lo.z);
    std::printf("bbox max: (%.4f, %.4f, %.4f) mm\n", hi.x, hi.y, hi.z);
    std::printf("mesh checksum: %.3f\n", checksum);
    std::printf("RESULT: %s (GPU mesh matches CPU within tol=%.1e mm)\n",
                pass ? "PASS" : "FAIL", TOLERANCE);

    // ---- 5b. Varying / scientific detail -> STDERR -----------------------
    // The synthetic sample is a sphere implicit field, so we can compare the
    // extracted area to the ANALYTIC sphere area 4*pi*r^2 -- validating the
    // science (the surface is where it should be), not just CPU==GPU agreement.
    // The radius is recorded in data/README.md; we recompute it from the bbox.
    const double rx = 0.5 * (hi.x - lo.x);
    const double analytic = 4.0 * 3.14159265358979323846 * rx * rx;
    std::fprintf(stderr, "[data]   source: %s  (%d^3-ish volume, %d cells)\n",
                 path.c_str(), d.nx, num_cells(d));
    std::fprintf(stderr, "[timing] CPU marching cubes: %.3f ms   GPU (count+scan+gen): %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- on this tiny sample the GPU is "
                         "launch-bound; its edge grows with volume size (clinical CTs are 512^3+).\n");
    std::fprintf(stderr, "[science] extracted area %.3f mm^2 vs analytic sphere 4*pi*r^2 = %.3f mm^2 "
                         "(r~%.3f mm from bbox; MC slightly under-estimates curved area).\n",
                 area, analytic, rx);
    std::fprintf(stderr, "[verify] max_vertex_err = %.3e mm  (tolerance %.1e mm)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
