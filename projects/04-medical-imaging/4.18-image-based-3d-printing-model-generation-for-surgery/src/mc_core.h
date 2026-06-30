// ===========================================================================
// src/mc_core.h  --  The ONE TRUE marching-cubes core (CPU/GPU shared math)
// ---------------------------------------------------------------------------
// Project 4.18 : Image-Based 3D Printing / Model Generation for Surgery
//
// ROLE IN THE PROJECT  (read this FIRST -- it is the heart of the project)
//   "Marching cubes" turns a 3-D SCALAR FIELD (here: a CT-like intensity volume)
//   into a TRIANGLE MESH that traces one chosen iso-value -- the "isosurface".
//   That mesh is exactly what a surgeon's 3-D printer consumes (an STL is just a
//   list of triangles). So this file answers: "given the 8 corner intensities of
//   one little cube of the volume, which triangles does the surface cut through
//   it, and where exactly are their vertices?"
//
//   This header is included by BOTH compilers:
//     * nvcc compiles kernels.cu, which calls these functions from a __device__
//       thread (one thread per cube).
//     * the host C++ compiler compiles reference_cpu.cpp, which calls the SAME
//       functions in a serial loop.
//   Because both sides run the identical float arithmetic (the HD-macro idiom,
//   see docs/PATTERNS.md §2), the CPU and GPU meshes come out essentially
//   bit-for-bit identical -- which makes verification an honest, tight check
//   rather than a hand-wave.
//
// WHY MARCHING CUBES IS THE GPU LESSON HERE
//   Every cube ("cell") of the volume is processed INDEPENDENTLY -- a cube's
//   triangles depend only on its own 8 corners. That is embarrassingly parallel:
//   millions of cubes, one GPU thread each. The only twist is that different
//   cubes emit different numbers of triangles (0..5), so the output is RAGGED.
//   We solve that with the classic GPU idiom: COUNT per cell -> PREFIX-SUM the
//   counts to get each cell's output offset -> WRITE. See kernels.cu + THEORY.md.
//
//   IMPORTANT: this header must stay free of <cuda_runtime.h> and __global__ so
//   the plain host compiler can include it. Only the HD inline helpers live here.
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

#include <cstdint>   // std::uint8_t

// ---------------------------------------------------------------------------
// HD: the "host+device" decorator macro (docs/PATTERNS.md §2).
//   When nvcc is compiling (__CUDACC__ is defined) we tag these helpers with
//   __host__ __device__ so they can run on BOTH the CPU and inside a kernel.
//   When the plain host compiler is compiling reference_cpu.cpp, __CUDACC__ is
//   NOT defined and the decorators would be a syntax error, so we expand to
//   nothing. One source of truth, two back ends -> identical math.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define MC_HD __host__ __device__
#else
#define MC_HD
#endif

// A single 3-D point / vertex in world space (float = FP32, what STL/printers
// use, and what the GPU is fastest at). We keep it a POD struct so it copies
// trivially between host and device memory with a flat cudaMemcpy.
struct Vec3 {
    float x, y, z;
};

// ---------------------------------------------------------------------------
// VOLUME LAYOUT  (shared by CPU + GPU so indexing never diverges)
//   The scalar field is an nx*ny*nz grid of samples, stored row-major with x
//   fastest:  value(i,j,k) = vol[ (k*ny + j)*nx + i ].
//   A "cell" (cube) (ci,cj,ck) spans samples i..i+1, j..j+1, k..k+1, so there
//   are (nx-1)*(ny-1)*(nz-1) cells. One GPU thread will own one cell.
// ---------------------------------------------------------------------------
struct VolDims {
    int nx, ny, nz;       // number of SAMPLES along each axis
    float spacing;        // physical distance between samples (e.g. mm/voxel)
    float origin_x, origin_y, origin_z;  // world coord of sample (0,0,0)
};

// Flatten a sample coordinate to a linear index (host+device identical).
MC_HD inline int vol_index(const VolDims& d, int i, int j, int k) {
    return (k * d.ny + j) * d.nx + i;
}

// Number of cells along each axis and in total.
MC_HD inline int cells_x(const VolDims& d) { return d.nx - 1; }
MC_HD inline int cells_y(const VolDims& d) { return d.ny - 1; }
MC_HD inline int cells_z(const VolDims& d) { return d.nz - 1; }
MC_HD inline int num_cells(const VolDims& d) {
    return cells_x(d) * cells_y(d) * cells_z(d);
}

// ---------------------------------------------------------------------------
// THE MARCHING-CUBES CONNECTIVITY TABLES
//   These two tables are the canonical Lorensen-&-Cline (1987) lookup tables.
//   They encode, for each of the 2^8 = 256 possible "which corners are inside"
//   patterns, WHICH edges the surface crosses and HOW to stitch the crossing
//   points into triangles. They are pure combinatorics -- no geometry -- so they
//   are the same on every machine. We mark them `static const` in a header,
//   which on the device becomes per-translation-unit constant data the compiler
//   places in constant/global memory; the access pattern (every thread indexes
//   by its own cube_index) is cache-friendly.
//
//   CORNER NUMBERING (standard MC convention), within a cell:
//        4---------5        z
//       /|        /|        ^   y
//      7---------6 |        |  /
//      | |       | |        | /
//      | 0-------|-1        +----> x
//      |/        |/
//      3---------2
//   Corner i has local offset (dx,dy,dz) given by CORNER_OFFSET[i] below.
//
//   EDGE NUMBERING: 12 edges, each connecting two of the 8 corners; EDGE_VERTS
//   lists those corner pairs. The surface vertex on a crossed edge is the
//   linearly-interpolated zero of (value - iso) between the two corners.
// ---------------------------------------------------------------------------

// Local (dx,dy,dz) of each of the 8 corners relative to the cell's base sample.
MC_HD inline void corner_offset(int c, int& dx, int& dy, int& dz) {
    // Encoded so corner order matches the tables above. Kept as a switch (not a
    // table) so it is trivially __device__-callable with no static storage.
    switch (c) {
        case 0: dx = 0; dy = 0; dz = 0; break;
        case 1: dx = 1; dy = 0; dz = 0; break;
        case 2: dx = 1; dy = 1; dz = 0; break;
        case 3: dx = 0; dy = 1; dz = 0; break;
        case 4: dx = 0; dy = 0; dz = 1; break;
        case 5: dx = 1; dy = 0; dz = 1; break;
        case 6: dx = 1; dy = 1; dz = 1; break;
        default: dx = 0; dy = 1; dz = 1; break;  // case 7
    }
}

// ---------------------------------------------------------------------------
// HOST+DEVICE LOOKUP TABLES -- how to make a const array readable from BOTH the
// CPU reference and a GPU kernel WITHOUT duplicating the literals.
//
//   A plain `static const int T[] = {...};` at file scope is HOST-ONLY: if a
//   __device__ function indexes T[i] at run time, nvcc errors with "identifier
//   undefined in device code". The fix is to give the SAME definition device
//   storage when nvcc is compiling the device pass. nvcc compiles each .cu in
//   two passes and defines __CUDA_ARCH__ only in the device pass, so:
//       MC_TABLE  ==  __device__   during the device pass  -> device storage
//       MC_TABLE  ==  (nothing)    during the host pass     -> host storage
//   `static` keeps internal linkage so each translation unit gets its own copy
//   (these tables are tiny). This is the standard "shared host/device table"
//   idiom and avoids maintaining two copies of the 256-row TRI_TABLE.
// ---------------------------------------------------------------------------
#ifdef __CUDA_ARCH__
#define MC_TABLE __device__ static const
#else
#define MC_TABLE static const
#endif

// EDGE_VERTS[e] = {a,b}: edge e connects corner a to corner b. Used to know
// which two corner values to interpolate between for the vertex on edge e.
MC_TABLE int EDGE_VERTS[12][2] = {
    {0,1},{1,2},{2,3},{3,0},   // 4 bottom edges (z=0 face)
    {4,5},{5,6},{6,7},{7,4},   // 4 top edges    (z=1 face)
    {0,4},{1,5},{2,6},{3,7}    // 4 vertical edges connecting bottom->top
};

// TRI_TABLE[cube_index] lists the triangles as triples of EDGE indices,
// terminated by -1. At most 5 triangles (15 edge entries) + terminator per row.
// Reading "e0,e1,e2" means: emit a triangle whose 3 vertices sit on edges
// e0,e1,e2 (each vertex found by interpolating along that edge). This is the
// canonical 256x16 table; it is long but it is the entire "intelligence" of MC.
// MC_TABLE gives it host+device storage (see the idiom note above EDGE_VERTS).
MC_TABLE int TRI_TABLE[256][16] = {
{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,1,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,8,3,9,8,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,8,3,1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{9,2,10,0,2,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{2,8,3,2,10,8,10,9,8,-1,-1,-1,-1,-1,-1,-1},
{3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,11,2,8,11,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,9,0,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,11,2,1,9,11,9,8,11,-1,-1,-1,-1,-1,-1,-1},
{3,10,1,11,10,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,10,1,0,8,10,8,11,10,-1,-1,-1,-1,-1,-1,-1},
{3,9,0,3,11,9,11,10,9,-1,-1,-1,-1,-1,-1,-1},
{9,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,3,0,7,3,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,1,9,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,1,9,4,7,1,7,3,1,-1,-1,-1,-1,-1,-1,-1},
{1,2,10,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{3,4,7,3,0,4,1,2,10,-1,-1,-1,-1,-1,-1,-1},
{9,2,10,9,0,2,8,4,7,-1,-1,-1,-1,-1,-1,-1},
{2,10,9,2,9,7,2,7,3,7,9,4,-1,-1,-1,-1},
{8,4,7,3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{11,4,7,11,2,4,2,0,4,-1,-1,-1,-1,-1,-1,-1},
{9,0,1,8,4,7,2,3,11,-1,-1,-1,-1,-1,-1,-1},
{4,7,11,9,4,11,9,11,2,9,2,1,-1,-1,-1,-1},
{3,10,1,3,11,10,7,8,4,-1,-1,-1,-1,-1,-1,-1},
{1,11,10,1,4,11,1,0,4,7,11,4,-1,-1,-1,-1},
{4,7,8,9,0,11,9,11,10,11,0,3,-1,-1,-1,-1},
{4,7,11,4,11,9,9,11,10,-1,-1,-1,-1,-1,-1,-1},
{9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{9,5,4,0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,5,4,1,5,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{8,5,4,8,3,5,3,1,5,-1,-1,-1,-1,-1,-1,-1},
{1,2,10,9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{3,0,8,1,2,10,4,9,5,-1,-1,-1,-1,-1,-1,-1},
{5,2,10,5,4,2,4,0,2,-1,-1,-1,-1,-1,-1,-1},
{2,10,5,3,2,5,3,5,4,3,4,8,-1,-1,-1,-1},
{9,5,4,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,11,2,0,8,11,4,9,5,-1,-1,-1,-1,-1,-1,-1},
{0,5,4,0,1,5,2,3,11,-1,-1,-1,-1,-1,-1,-1},
{2,1,5,2,5,8,2,8,11,4,8,5,-1,-1,-1,-1},
{10,3,11,10,1,3,9,5,4,-1,-1,-1,-1,-1,-1,-1},
{4,9,5,0,8,1,8,10,1,8,11,10,-1,-1,-1,-1},
{5,4,0,5,0,11,5,11,10,11,0,3,-1,-1,-1,-1},
{5,4,8,5,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1},
{9,7,8,5,7,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{9,3,0,9,5,3,5,7,3,-1,-1,-1,-1,-1,-1,-1},
{0,7,8,0,1,7,1,5,7,-1,-1,-1,-1,-1,-1,-1},
{1,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{9,7,8,9,5,7,10,1,2,-1,-1,-1,-1,-1,-1,-1},
{10,1,2,9,5,0,5,3,0,5,7,3,-1,-1,-1,-1},
{8,0,2,8,2,5,8,5,7,10,5,2,-1,-1,-1,-1},
{2,10,5,2,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1},
{7,9,5,7,8,9,3,11,2,-1,-1,-1,-1,-1,-1,-1},
{9,5,7,9,7,2,9,2,0,2,7,11,-1,-1,-1,-1},
{2,3,11,0,1,8,1,7,8,1,5,7,-1,-1,-1,-1},
{11,2,1,11,1,7,7,1,5,-1,-1,-1,-1,-1,-1,-1},
{9,5,8,8,5,7,10,1,3,10,3,11,-1,-1,-1,-1},
{5,7,0,5,0,9,7,11,0,1,0,10,11,10,0,-1},
{11,10,0,11,0,3,10,5,0,8,0,7,5,7,0,-1},
{11,10,5,7,11,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,8,3,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{9,0,1,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,8,3,1,9,8,5,10,6,-1,-1,-1,-1,-1,-1,-1},
{1,6,5,2,6,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,6,5,1,2,6,3,0,8,-1,-1,-1,-1,-1,-1,-1},
{9,6,5,9,0,6,0,2,6,-1,-1,-1,-1,-1,-1,-1},
{5,9,8,5,8,2,5,2,6,3,2,8,-1,-1,-1,-1},
{2,3,11,10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{11,0,8,11,2,0,10,6,5,-1,-1,-1,-1,-1,-1,-1},
{0,1,9,2,3,11,5,10,6,-1,-1,-1,-1,-1,-1,-1},
{5,10,6,1,9,2,9,11,2,9,8,11,-1,-1,-1,-1},
{6,3,11,6,5,3,5,1,3,-1,-1,-1,-1,-1,-1,-1},
{0,8,11,0,11,5,0,5,1,5,11,6,-1,-1,-1,-1},
{3,11,6,0,3,6,0,6,5,0,5,9,-1,-1,-1,-1},
{6,5,9,6,9,11,11,9,8,-1,-1,-1,-1,-1,-1,-1},
{5,10,6,4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,3,0,4,7,3,6,5,10,-1,-1,-1,-1,-1,-1,-1},
{1,9,0,5,10,6,8,4,7,-1,-1,-1,-1,-1,-1,-1},
{10,6,5,1,9,7,1,7,3,7,9,4,-1,-1,-1,-1},
{6,1,2,6,5,1,4,7,8,-1,-1,-1,-1,-1,-1,-1},
{1,2,5,5,2,6,3,0,4,3,4,7,-1,-1,-1,-1},
{8,4,7,9,0,5,0,6,5,0,2,6,-1,-1,-1,-1},
{7,3,9,7,9,4,3,2,9,5,9,6,2,6,9,-1},
{3,11,2,7,8,4,10,6,5,-1,-1,-1,-1,-1,-1,-1},
{5,10,6,4,7,2,4,2,0,2,7,11,-1,-1,-1,-1},
{0,1,9,4,7,8,2,3,11,5,10,6,-1,-1,-1,-1},
{9,2,1,9,11,2,9,4,11,7,11,4,5,10,6,-1},
{8,4,7,3,11,5,3,5,1,5,11,6,-1,-1,-1,-1},
{5,1,11,5,11,6,1,0,11,7,11,4,0,4,11,-1},
{0,5,9,0,6,5,0,3,6,11,6,3,8,4,7,-1},
{6,5,9,6,9,11,4,7,9,7,11,9,-1,-1,-1,-1},
{10,4,9,6,4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,10,6,4,9,10,0,8,3,-1,-1,-1,-1,-1,-1,-1},
{10,0,1,10,6,0,6,4,0,-1,-1,-1,-1,-1,-1,-1},
{8,3,1,8,1,6,8,6,4,6,1,10,-1,-1,-1,-1},
{1,4,9,1,2,4,2,6,4,-1,-1,-1,-1,-1,-1,-1},
{3,0,8,1,2,9,2,4,9,2,6,4,-1,-1,-1,-1},
{0,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{8,3,2,8,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1},
{10,4,9,10,6,4,11,2,3,-1,-1,-1,-1,-1,-1,-1},
{0,8,2,2,8,11,4,9,10,4,10,6,-1,-1,-1,-1},
{3,11,2,0,1,6,0,6,4,6,1,10,-1,-1,-1,-1},
{6,4,1,6,1,10,4,8,1,2,1,11,8,11,1,-1},
{9,6,4,9,3,6,9,1,3,11,6,3,-1,-1,-1,-1},
{8,11,1,8,1,0,11,6,1,9,1,4,6,4,1,-1},
{3,11,6,3,6,0,0,6,4,-1,-1,-1,-1,-1,-1,-1},
{6,4,8,11,6,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{7,10,6,7,8,10,8,9,10,-1,-1,-1,-1,-1,-1,-1},
{0,7,3,0,10,7,0,9,10,6,7,10,-1,-1,-1,-1},
{10,6,7,1,10,7,1,7,8,1,8,0,-1,-1,-1,-1},
{10,6,7,10,7,1,1,7,3,-1,-1,-1,-1,-1,-1,-1},
{1,2,6,1,6,8,1,8,9,8,6,7,-1,-1,-1,-1},
{2,6,9,2,9,1,6,7,9,0,9,3,7,3,9,-1},
{7,8,0,7,0,6,6,0,2,-1,-1,-1,-1,-1,-1,-1},
{7,3,2,6,7,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{2,3,11,10,6,8,10,8,9,8,6,7,-1,-1,-1,-1},
{2,0,7,2,7,11,0,9,7,6,7,10,9,10,7,-1},
{1,8,0,1,7,8,1,10,7,6,7,10,2,3,11,-1},
{11,2,1,11,1,7,10,6,1,6,7,1,-1,-1,-1,-1},
{8,9,6,8,6,7,9,1,6,11,6,3,1,3,6,-1},
{0,9,1,11,6,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{7,8,0,7,0,6,3,11,0,11,6,0,-1,-1,-1,-1},
{7,11,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{3,0,8,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,1,9,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{8,1,9,8,3,1,11,7,6,-1,-1,-1,-1,-1,-1,-1},
{10,1,2,6,11,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,2,10,3,0,8,6,11,7,-1,-1,-1,-1,-1,-1,-1},
{2,9,0,2,10,9,6,11,7,-1,-1,-1,-1,-1,-1,-1},
{6,11,7,2,10,3,10,8,3,10,9,8,-1,-1,-1,-1},
{7,2,3,6,2,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{7,0,8,7,6,0,6,2,0,-1,-1,-1,-1,-1,-1,-1},
{2,7,6,2,3,7,0,1,9,-1,-1,-1,-1,-1,-1,-1},
{1,6,2,1,8,6,1,9,8,8,7,6,-1,-1,-1,-1},
{10,7,6,10,1,7,1,3,7,-1,-1,-1,-1,-1,-1,-1},
{10,7,6,1,7,10,1,8,7,1,0,8,-1,-1,-1,-1},
{0,3,7,0,7,10,0,10,9,6,10,7,-1,-1,-1,-1},
{7,6,10,7,10,8,8,10,9,-1,-1,-1,-1,-1,-1,-1},
{6,8,4,11,8,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{3,6,11,3,0,6,0,4,6,-1,-1,-1,-1,-1,-1,-1},
{8,6,11,8,4,6,9,0,1,-1,-1,-1,-1,-1,-1,-1},
{9,4,6,9,6,3,9,3,1,11,3,6,-1,-1,-1,-1},
{6,8,4,6,11,8,2,10,1,-1,-1,-1,-1,-1,-1,-1},
{1,2,10,3,0,11,0,6,11,0,4,6,-1,-1,-1,-1},
{4,11,8,4,6,11,0,2,9,2,10,9,-1,-1,-1,-1},
{10,9,3,10,3,2,9,4,3,11,3,6,4,6,3,-1},
{8,2,3,8,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1},
{0,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,9,0,2,3,4,2,4,6,4,3,8,-1,-1,-1,-1},
{1,9,4,1,4,2,2,4,6,-1,-1,-1,-1,-1,-1,-1},
{8,1,3,8,6,1,8,4,6,6,10,1,-1,-1,-1,-1},
{10,1,0,10,0,6,6,0,4,-1,-1,-1,-1,-1,-1,-1},
{4,6,3,4,3,8,6,10,3,0,3,9,10,9,3,-1},
{10,9,4,6,10,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,9,5,7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,8,3,4,9,5,11,7,6,-1,-1,-1,-1,-1,-1,-1},
{5,0,1,5,4,0,7,6,11,-1,-1,-1,-1,-1,-1,-1},
{11,7,6,8,3,4,3,5,4,3,1,5,-1,-1,-1,-1},
{9,5,4,10,1,2,7,6,11,-1,-1,-1,-1,-1,-1,-1},
{6,11,7,1,2,10,0,8,3,4,9,5,-1,-1,-1,-1},
{7,6,11,5,4,10,4,2,10,4,0,2,-1,-1,-1,-1},
{3,4,8,3,5,4,3,2,5,10,5,2,11,7,6,-1},
{7,2,3,7,6,2,5,4,9,-1,-1,-1,-1,-1,-1,-1},
{9,5,4,0,8,6,0,6,2,6,8,7,-1,-1,-1,-1},
{3,6,2,3,7,6,1,5,0,5,4,0,-1,-1,-1,-1},
{6,2,8,6,8,7,2,1,8,4,8,5,1,5,8,-1},
{9,5,4,10,1,6,1,7,6,1,3,7,-1,-1,-1,-1},
{1,6,10,1,7,6,1,0,7,8,7,0,9,5,4,-1},
{4,0,10,4,10,5,0,3,10,6,10,7,3,7,10,-1},
{7,6,10,7,10,8,5,4,10,4,8,10,-1,-1,-1,-1},
{6,9,5,6,11,9,11,8,9,-1,-1,-1,-1,-1,-1,-1},
{3,6,11,0,6,3,0,5,6,0,9,5,-1,-1,-1,-1},
{0,11,8,0,5,11,0,1,5,5,6,11,-1,-1,-1,-1},
{6,11,3,6,3,5,5,3,1,-1,-1,-1,-1,-1,-1,-1},
{1,2,10,9,5,11,9,11,8,11,5,6,-1,-1,-1,-1},
{0,11,3,0,6,11,0,9,6,5,6,9,1,2,10,-1},
{11,8,5,11,5,6,8,0,5,10,5,2,0,2,5,-1},
{6,11,3,6,3,5,2,10,3,10,5,3,-1,-1,-1,-1},
{5,8,9,5,2,8,5,6,2,3,8,2,-1,-1,-1,-1},
{9,5,6,9,6,0,0,6,2,-1,-1,-1,-1,-1,-1,-1},
{1,5,8,1,8,0,5,6,8,3,8,2,6,2,8,-1},
{1,5,6,2,1,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,3,6,1,6,10,3,8,6,5,6,9,8,9,6,-1},
{10,1,0,10,0,6,9,5,0,5,6,0,-1,-1,-1,-1},
{0,3,8,5,6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{10,5,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{11,5,10,7,5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{11,5,10,11,7,5,8,3,0,-1,-1,-1,-1,-1,-1,-1},
{5,11,7,5,10,11,1,9,0,-1,-1,-1,-1,-1,-1,-1},
{10,7,5,10,11,7,9,8,1,8,3,1,-1,-1,-1,-1},
{11,1,2,11,7,1,7,5,1,-1,-1,-1,-1,-1,-1,-1},
{0,8,3,1,2,7,1,7,5,7,2,11,-1,-1,-1,-1},
{9,7,5,9,2,7,9,0,2,2,11,7,-1,-1,-1,-1},
{7,5,2,7,2,11,5,9,2,3,2,8,9,8,2,-1},
{2,5,10,2,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1},
{8,2,0,8,5,2,8,7,5,10,2,5,-1,-1,-1,-1},
{9,0,1,5,10,3,5,3,7,3,10,2,-1,-1,-1,-1},
{9,8,2,9,2,1,8,7,2,10,2,5,7,5,2,-1},
{1,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,8,7,0,7,1,1,7,5,-1,-1,-1,-1,-1,-1,-1},
{9,0,3,9,3,5,5,3,7,-1,-1,-1,-1,-1,-1,-1},
{9,8,7,5,9,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{5,8,4,5,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1},
{5,0,4,5,11,0,5,10,11,11,3,0,-1,-1,-1,-1},
{0,1,9,8,4,10,8,10,11,10,4,5,-1,-1,-1,-1},
{10,11,4,10,4,5,11,3,4,9,4,1,3,1,4,-1},
{2,5,1,2,8,5,2,11,8,4,5,8,-1,-1,-1,-1},
{0,4,11,0,11,3,4,5,11,2,11,1,5,1,11,-1},
{0,2,5,0,5,9,2,11,5,4,5,8,11,8,5,-1},
{9,4,5,2,11,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{2,5,10,3,5,2,3,4,5,3,8,4,-1,-1,-1,-1},
{5,10,2,5,2,4,4,2,0,-1,-1,-1,-1,-1,-1,-1},
{3,10,2,3,5,10,3,8,5,4,5,8,0,1,9,-1},
{5,10,2,5,2,4,1,9,2,9,4,2,-1,-1,-1,-1},
{8,4,5,8,5,3,3,5,1,-1,-1,-1,-1,-1,-1,-1},
{0,4,5,1,0,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{8,4,5,8,5,3,9,0,5,0,3,5,-1,-1,-1,-1},
{9,4,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,11,7,4,9,11,9,10,11,-1,-1,-1,-1,-1,-1,-1},
{0,8,3,4,9,7,9,11,7,9,10,11,-1,-1,-1,-1},
{1,10,11,1,11,4,1,4,0,7,4,11,-1,-1,-1,-1},
{3,1,4,3,4,8,1,10,4,7,4,11,10,11,4,-1},
{4,11,7,9,11,4,9,2,11,9,1,2,-1,-1,-1,-1},
{9,7,4,9,11,7,9,1,11,2,11,1,0,8,3,-1},
{11,7,4,11,4,2,2,4,0,-1,-1,-1,-1,-1,-1,-1},
{11,7,4,11,4,2,8,3,4,3,2,4,-1,-1,-1,-1},
{2,9,10,2,7,9,2,3,7,7,4,9,-1,-1,-1,-1},
{9,10,7,9,7,4,10,2,7,8,7,0,2,0,7,-1},
{3,7,10,3,10,2,7,4,10,1,10,0,4,0,10,-1},
{1,10,2,8,7,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,9,1,4,1,7,7,1,3,-1,-1,-1,-1,-1,-1,-1},
{4,9,1,4,1,7,0,8,1,8,7,1,-1,-1,-1,-1},
{4,0,3,7,4,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,8,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{9,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{3,0,9,3,9,11,11,9,10,-1,-1,-1,-1,-1,-1,-1},
{0,1,10,0,10,8,8,10,11,-1,-1,-1,-1,-1,-1,-1},
{3,1,10,11,3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,2,11,1,11,9,9,11,8,-1,-1,-1,-1,-1,-1,-1},
{3,0,9,3,9,11,1,2,9,2,11,9,-1,-1,-1,-1},
{0,2,11,8,0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{3,2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{2,3,8,2,8,10,10,8,9,-1,-1,-1,-1,-1,-1,-1},
{9,10,2,0,9,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{2,3,8,2,8,10,0,1,8,1,10,8,-1,-1,-1,-1},
{1,10,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,3,8,9,1,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,9,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,3,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
};

// ---------------------------------------------------------------------------
// NUM_TRIS_FOR_CUBE: how many triangles this corner-pattern emits.
//   We just count the TRI_TABLE entries before the -1 terminator and divide by
//   3 (3 edge-vertices per triangle). The COUNT pass calls exactly this; the
//   prefix sum of these counts gives each cell its write offset.
// ---------------------------------------------------------------------------
MC_HD inline int num_tris_for_cube(int cube_index) {
    int n = 0;
    // TRI_TABLE rows are -1-terminated; each triangle uses 3 entries.
    while (TRI_TABLE[cube_index][n * 3] != -1) ++n;
    return n;
}

// ---------------------------------------------------------------------------
// classify_cube: build the 8-bit "cube index" for one cell.
//   Bit c is set if corner c is INSIDE the surface, i.e. its sample value is
//   >= iso. (Convention: "inside" = at-or-above the iso-value, e.g. denser than
//   the bone threshold in a CT.) The 8 bits index TRI_TABLE / EDGE_TABLE.
//
//   corner_val[8] are the 8 corner sample values in MC corner order.
// ---------------------------------------------------------------------------
MC_HD inline int classify_cube(const float corner_val[8], float iso) {
    int cube_index = 0;
    // Unrolled-by-loop over 8 corners; branchless bit set keeps the GPU warp
    // from diverging on the comparison.
    for (int c = 0; c < 8; ++c) {
        // (value >= iso) -> 1 contributes bit c. We use >= (not >) so the CPU
        // and GPU agree exactly on the boundary case value==iso.
        cube_index |= (corner_val[c] >= iso ? 1 : 0) << c;
    }
    return cube_index;
}

// ---------------------------------------------------------------------------
// interp_edge: the linearly-interpolated surface vertex on one edge.
//   Edge `e` connects corners a=EDGE_VERTS[e][0], b=EDGE_VERTS[e][1]. The
//   surface (value == iso) crosses the segment a->b at parameter
//       t = (iso - val_a) / (val_b - val_a)
//   and the vertex is p_a + t*(p_b - p_a). This single formula -- run with the
//   SAME float ops on CPU and GPU -- is why the two meshes match. We guard the
//   degenerate val_a==val_b case (parallel to the surface) by falling back to
//   the midpoint, which both sides also do identically.
//
//   pos[8]  : the 8 corner world positions (Vec3), MC corner order.
//   val[8]  : the 8 corner sample values.
// ---------------------------------------------------------------------------
MC_HD inline Vec3 interp_edge(int e, const Vec3 pos[8], const float val[8], float iso) {
    const int a = EDGE_VERTS[e][0];
    const int b = EDGE_VERTS[e][1];
    const float va = val[a];
    const float vb = val[b];
    const float denom = vb - va;
    // t in [0,1]: where along a->b the field equals iso. Branch is identical on
    // both back ends, so no host/device divergence in the result.
    float t = (denom != 0.0f) ? (iso - va) / denom : 0.5f;
    Vec3 r;
    r.x = pos[a].x + t * (pos[b].x - pos[a].x);
    r.y = pos[a].y + t * (pos[b].y - pos[a].y);
    r.z = pos[a].z + t * (pos[b].z - pos[a].z);
    return r;
}

// ---------------------------------------------------------------------------
// gather_corners: load the 8 corner values and world positions for one cell.
//   This is shared so the CPU loop and the GPU kernel address memory the SAME
//   way (no chance of an off-by-one between them). `vol` is the flat sample
//   array; (ci,cj,ck) is the cell's base corner.
// ---------------------------------------------------------------------------
MC_HD inline void gather_corners(const float* vol, const VolDims& d,
                                 int ci, int cj, int ck,
                                 float val[8], Vec3 pos[8]) {
    for (int c = 0; c < 8; ++c) {
        int dx, dy, dz;
        corner_offset(c, dx, dy, dz);
        const int i = ci + dx, j = cj + dy, k = ck + dz;
        val[c] = vol[vol_index(d, i, j, k)];
        // World position of this sample = origin + index*spacing. Printers want
        // physical (mm) coordinates, so we bake spacing/origin in here.
        pos[c].x = d.origin_x + i * d.spacing;
        pos[c].y = d.origin_y + j * d.spacing;
        pos[c].z = d.origin_z + k * d.spacing;
    }
}

// A triangle = 3 world-space vertices. The mesh is just an array of these.
// (STL stores exactly this -- 3 vertices + a face normal per triangle.)
struct Triangle {
    Vec3 v[3];
};
