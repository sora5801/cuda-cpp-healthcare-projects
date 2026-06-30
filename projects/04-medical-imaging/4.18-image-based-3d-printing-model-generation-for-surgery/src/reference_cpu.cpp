// ===========================================================================
// src/reference_cpu.cpp  --  Serial marching cubes: the trusted baseline
// ---------------------------------------------------------------------------
// Project 4.18 : Image-Based 3D Printing / Model Generation for Surgery
//
// This is the plain-C++ implementation of the SAME computation the GPU does. It
// is intentionally simple and serial -- one nested loop over every cell -- so a
// learner can read it top-to-bottom and trust it. main.cu runs this AND the GPU
// kernel and asserts the two meshes agree (see TOLERANCE in main.cu).
//
// All the per-cube math (classify, interpolate, the lookup tables) lives in
// mc_core.h and is shared verbatim with kernels.cu, so "CPU" and "GPU" here run
// byte-identical float arithmetic. The only thing this file owns is the SERIAL
// ITERATION ORDER -- and that order (ascending cell index) is exactly the order
// the GPU writes into after its prefix-sum, which is what makes the two meshes
// line up vertex-for-vertex.
//
// READ mc_core.h and reference_cpu.h FIRST.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::sqrt, std::fabs, std::floor
#include <fstream>     // std::ifstream
#include <sstream>     // std::ostringstream (error messages)
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// load_problem: parse the tiny text volume in data/sample/.
//   Format (see data/README.md):
//     line 1 : nx ny nz spacing origin_x origin_y origin_z iso
//     rest   : nx*ny*nz floats, row-major with x fastest then y then z.
//   We validate the count so a truncated file fails loudly instead of silently
//   marching garbage.
// ---------------------------------------------------------------------------
MCProblem load_problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open volume file: " + path);

    MCProblem p;
    VolDims& d = p.dims;
    // Read the header. operator>> skips whitespace/newlines, so the layout is
    // forgiving about exactly how the numbers are spread across lines.
    if (!(in >> d.nx >> d.ny >> d.nz >> d.spacing
             >> d.origin_x >> d.origin_y >> d.origin_z >> p.iso)) {
        throw std::runtime_error("malformed header in volume file: " + path);
    }
    if (d.nx < 2 || d.ny < 2 || d.nz < 2) {
        throw std::runtime_error("volume must be at least 2x2x2 to have a cell");
    }

    const std::size_t n = static_cast<std::size_t>(d.nx) * d.ny * d.nz;
    p.vol.resize(n);
    for (std::size_t idx = 0; idx < n; ++idx) {
        if (!(in >> p.vol[idx])) {
            std::ostringstream msg;
            msg << "volume file " << path << " has fewer samples than declared ("
                << idx << " of " << n << ")";
            throw std::runtime_error(msg.str());
        }
    }
    return p;
}

// ---------------------------------------------------------------------------
// emit_cell_triangles: the per-cell core, shared in spirit with the GPU kernel.
//   Given a cell's base corner (ci,cj,ck), classify it, then for each triangle
//   in its TRI_TABLE row, interpolate the three edge vertices and append the
//   triangle. This is literally what one GPU thread does for one cube.
//
//   We factor it out so the COUNT (just num_tris_for_cube) and the GENERATE
//   (this function) use the identical classification -- no chance of the two
//   passes disagreeing about how many triangles a cell has.
// ---------------------------------------------------------------------------
static void emit_cell_triangles(const MCProblem& prob, int ci, int cj, int ck,
                                std::vector<Triangle>& out) {
    const VolDims& d = prob.dims;

    // 1. Load the 8 corner values + world positions for this cube.
    float val[8];
    Vec3  pos[8];
    gather_corners(prob.vol.data(), d, ci, cj, ck, val, pos);

    // 2. Which of the 256 corner-inside patterns is this? Indexes the tables.
    const int cube_index = classify_cube(val, prob.iso);

    // 3. Walk the triangle list for this pattern. Each group of 3 edge indices
    //    is one triangle; -1 terminates the row.
    const int* row = TRI_TABLE[cube_index];
    for (int t = 0; row[t * 3] != -1; ++t) {
        Triangle tri;
        for (int corner = 0; corner < 3; ++corner) {
            const int e = row[t * 3 + corner];          // edge this vertex lies on
            tri.v[corner] = interp_edge(e, pos, val, prob.iso);
        }
        out.push_back(tri);
    }
}

// ---------------------------------------------------------------------------
// marching_cubes_cpu: the full serial sweep (the reference).
//   The triple loop visits cells in ascending LINEAR cell index
//       cell = (ck*cells_y + cj)*cells_x + ci
//   i.e. cz outermost, cx innermost. The GPU assigns thread `cell` to exactly
//   this same cell index and writes its triangles at the prefix-summed offset
//   for `cell`, so both meshes end up in identical order -> exact comparison.
// ---------------------------------------------------------------------------
void marching_cubes_cpu(const MCProblem& prob, std::vector<Triangle>& out) {
    out.clear();
    const VolDims& d = prob.dims;
    const int CX = cells_x(d), CY = cells_y(d), CZ = cells_z(d);
    for (int ck = 0; ck < CZ; ++ck)
        for (int cj = 0; cj < CY; ++cj)
            for (int ci = 0; ci < CX; ++ci)
                emit_cell_triangles(prob, ci, cj, ck, out);
}

// ---------------------------------------------------------------------------
// mesh_surface_area: sum of triangle areas.
//   Area of a triangle (a,b,c) = 0.5 * |(b-a) x (c-a)|. We accumulate in double
//   so the headline number is stable regardless of triangle count. This is a
//   real geometry-processing primitive: surface area / volume drive how much
//   filament an FDM print uses.
// ---------------------------------------------------------------------------
double mesh_surface_area(const std::vector<Triangle>& tris) {
    double area = 0.0;
    for (const Triangle& T : tris) {
        // Two edge vectors from vertex 0.
        const double ux = T.v[1].x - T.v[0].x;
        const double uy = T.v[1].y - T.v[0].y;
        const double uz = T.v[1].z - T.v[0].z;
        const double vx = T.v[2].x - T.v[0].x;
        const double vy = T.v[2].y - T.v[0].y;
        const double vz = T.v[2].z - T.v[0].z;
        // Cross product u x v.
        const double cx = uy * vz - uz * vy;
        const double cy = uz * vx - ux * vz;
        const double cz = ux * vy - uy * vx;
        area += 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
    }
    return area;
}

// ---------------------------------------------------------------------------
// mesh_bbox: axis-aligned bounding box over all vertices.
//   Confirms the model's physical extent (mm). An empty mesh returns a
//   degenerate box at the origin (callers report 0 triangles anyway).
// ---------------------------------------------------------------------------
void mesh_bbox(const std::vector<Triangle>& tris, Vec3& lo, Vec3& hi) {
    if (tris.empty()) { lo = {0,0,0}; hi = {0,0,0}; return; }
    lo = hi = tris[0].v[0];
    for (const Triangle& T : tris) {
        for (int i = 0; i < 3; ++i) {
            const Vec3& p = T.v[i];
            if (p.x < lo.x) lo.x = p.x; if (p.x > hi.x) hi.x = p.x;
            if (p.y < lo.y) lo.y = p.y; if (p.y > hi.y) hi.y = p.y;
            if (p.z < lo.z) lo.z = p.z; if (p.z > hi.z) hi.z = p.z;
        }
    }
}

// ---------------------------------------------------------------------------
// mesh_checksum: an order-independent fingerprint of the whole mesh.
//   We sum |x|+|y|+|z| over every vertex (a commutative reduction, so the value
//   does NOT depend on triangle order) and round to 3 decimals. Printed to
//   stdout as a single stable number that changes if ANY vertex moves -- a cheap
//   way to make the demo's expected_output a real regression guard.
// ---------------------------------------------------------------------------
double mesh_checksum(const std::vector<Triangle>& tris) {
    double s = 0.0;
    for (const Triangle& T : tris)
        for (int i = 0; i < 3; ++i)
            s += std::fabs(T.v[i].x) + std::fabs(T.v[i].y) + std::fabs(T.v[i].z);
    // Quantize so tiny last-bit float noise cannot flip the printed digits.
    return std::floor(s * 1000.0 + 0.5) / 1000.0;
}
