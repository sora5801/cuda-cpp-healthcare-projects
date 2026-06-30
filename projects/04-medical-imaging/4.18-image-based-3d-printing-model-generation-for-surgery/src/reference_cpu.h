// ===========================================================================
// src/reference_cpu.h  --  Volume loader + serial marching-cubes reference
// ---------------------------------------------------------------------------
// Project 4.18 : Image-Based 3D Printing / Model Generation for Surgery
//
// WHAT THIS PROJECT COMPUTES
//   Given a 3-D scalar VOLUME (a CT-like intensity field) and an ISO-VALUE
//   (e.g. the bone threshold), extract the ISOSURFACE -- the triangle mesh that
//   traces value == iso -- using the Marching Cubes algorithm. That mesh is the
//   print-ready surgical model (an STL is just a triangle list). See THEORY.md.
//
// WHY A GPU  (the lesson)
//   The volume is chopped into (nx-1)*(ny-1)*(nz-1) little cubes. Each cube's
//   triangles depend ONLY on its own 8 corner values, so every cube is an
//   independent job -> one GPU thread per cube. A clinical 512^3 CT has ~1.3e8
//   cubes; the serial CPU walk below is the trusted-but-slow baseline the GPU
//   kernel is checked against.
//
// THIS HEADER (pure C++ -- NO CUDA, so the host compiler can build it)
//   * MCProblem        : the volume + iso-value + parsed dims.
//   * load_problem()   : read the tiny text volume in data/sample/.
//   * marching_cubes_cpu() : the serial reference. Fills a flat Triangle list in
//                            a DETERMINISTIC order (ascending cell index), the
//                            same order the GPU produces, so the two meshes can
//                            be compared vertex-by-vertex.
//   * mesh_surface_area(), mesh_bbox(), mesh_checksum() : tiny reductions used
//     for the deterministic stdout report and the analytic sphere-area check.
//
// The per-cube geometry/topology lives in mc_core.h and is SHARED with the GPU
// kernel (the HD-macro idiom) -- read mc_core.h first.
//
// READ THIS BEFORE: reference_cpu.cpp, main.cu.   READ mc_core.h FIRST.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "mc_core.h"   // VolDims, Vec3, Triangle, the MC tables + HD helpers

// One marching-cubes problem: the scalar volume plus the iso-value to extract.
//   vol[(k*ny+j)*nx+i] is the sample at grid point (i,j,k).
struct MCProblem {
    VolDims dims;             // nx,ny,nz, spacing, origin (see mc_core.h)
    float iso = 0.0f;         // iso-value to extract the surface at
    std::vector<float> vol;   // [nx*ny*nz] scalar field, row-major (x fastest)
};

// Load an MCProblem from the text format documented in data/README.md:
//   header line: "<nx> <ny> <nz> <spacing> <origin_x> <origin_y> <origin_z> <iso>"
//   then nx*ny*nz whitespace-separated floats in (k,j,i) row-major order.
// Throws std::runtime_error if the file is missing or malformed (fail loud).
MCProblem load_problem(const std::string& path);

// CPU reference marching cubes. Walks every cell in ascending linear cell index
// (cz outer, cy middle, cx inner) and appends its triangles to `out`. The
// ordering is fixed so the GPU result (also ordered by cell index) matches
// exactly. `out` is cleared first.
void marching_cubes_cpu(const MCProblem& prob, std::vector<Triangle>& out);

// ---- tiny mesh reductions (shared report metrics) ------------------------
// Total surface area = sum of triangle areas (each via the cross-product / 2).
// Used both for the stdout report and to compare against the analytic area of
// the known synthetic sphere (the "science" check in THEORY.md).
double mesh_surface_area(const std::vector<Triangle>& tris);

// Axis-aligned bounding box of all vertices (min/max corner). A cheap,
// deterministic sanity number that also confirms the model's physical extent.
void mesh_bbox(const std::vector<Triangle>& tris, Vec3& lo, Vec3& hi);

// A scale-/order-robust checksum of the mesh: the summed magnitude of all
// vertex coordinates, quantized to a fixed number of decimals. Printed to
// stdout so the demo's expected_output is a stable fingerprint of the mesh.
double mesh_checksum(const std::vector<Triangle>& tris);
