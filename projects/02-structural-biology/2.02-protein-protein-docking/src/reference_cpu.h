// ===========================================================================
// src/reference_cpu.h  --  Docking model, grid voxelization, brute-force
//                         correlation reference, and the SHARED scoring core
// ---------------------------------------------------------------------------
// Project 2.2 : Protein-Protein Docking  (catalog id 2.2)
//
// WHAT THIS PROJECT COMPUTES
//   Rigid-body protein-protein docking by the classical FFT-correlation method
//   (Katchalski-Katzir 1992; the engine inside ZDOCK and ClusPro). We:
//     1. Voxelize a fixed RECEPTOR and a mobile LIGAND onto the same 3D grid.
//        BOTH proteins become the same real-valued "shape function" on the grid:
//          shape(x):  +1 deep inside the core (buried interior),
//                     a NEGATIVE penalty in a thin surface "skin",
//                     0 in empty space.
//     2. Score EVERY rigid translation t of the ligand by the cross-correlation
//          S(t) = sum_x  R(x) * L(x - t)
//        Large positive S(t) means cores overlap cores (good buried contact)
//        and surfaces meet surfaces, while deep interpenetration (core meeting
//        the other body's penalty skin) is punished -- i.e. SHAPE
//        COMPLEMENTARITY. The best t is the predicted docking translation.
//
// WHY A GPU / cuFFT
//   S(t) for ALL translations t is a 3D cross-correlation. Computing it directly
//   is O(Ng^2) in the number of grid voxels Ng -- hopeless for a real grid of
//   ~10^6 voxels (10^12 multiply-adds per orientation, times thousands of
//   ligand rotations). The Convolution/Correlation Theorem turns it into three
//   FFTs:  S = IFFT( FFT(R) .* conj(FFT(L)) ),  which is O(Ng log Ng). On the GPU
//   we use the cuFFT library for those transforms -- the canonical "use a CUDA
//   library WITHOUT it being a black box" lesson (kernels.cu documents exactly
//   what each cuFFT call computes and the data layout it expects).
//
//   (Per the catalog, full docking also rotates the ligand over thousands of
//   orientations, adds electrostatics, and clusters/refines poses; we describe
//   that in THEORY "Where this sits in the real world". This teaching flagship
//   focuses on the FFT translational search for a SINGLE orientation, which is
//   the reusable computational kernel everything else is built around.)
//
// FILE ROLE
//   This is a PURE C++ header (no CUDA syntax) so it compiles under BOTH the
//   host compiler (for reference_cpu.cpp) AND nvcc (for kernels.cu / main.cu).
//   The HD-macro idiom (PATTERNS.md section 2) lets the voxelization helpers be
//   shared verbatim between CPU and GPU so the two grids are byte-identical and
//   verification is exact.
//
// READ THIS BEFORE: kernels.cuh, reference_cpu.cpp, main.cu.
// ===========================================================================
#pragma once

#include <cstddef>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// HD : the host/device decoration macro (PATTERNS.md section 2).
//   When this header is pulled in by nvcc (__CUDACC__ defined), HD expands to
//   `__host__ __device__` so the helper compiles for BOTH the CPU and the GPU.
//   Under the plain host compiler the decorators do not exist, so HD expands to
//   nothing. The body is identical either way -> CPU and GPU run the SAME math.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// Scoring constants for the Katchalski-Katzir shape grid.
//   CORE_VALUE    : value written into receptor voxels that are buried in the
//                   protein's interior. A ligand voxel landing here is "good
//                   contact" -> it ADDS to the correlation score.
//   SKIN_PENALTY  : value written into a protein's thin surface shell. It is
//                   NEGATIVE so that a voxel of one body overlapping the OTHER
//                   body's surface atoms (a steric CLASH) SUBTRACTS from the
//                   score. The magnitude (clash penalty) is chosen large enough
//                   that any deep overlap outweighs a little extra buried contact
//                   -- exactly how ZDOCK's grid discourages interpenetration.
// BOTH the receptor and the ligand use this SAME two-value model (a positive
// core, a negative skin), so the docking score is a geometric-complementarity
// correlation. These are small integers stored as floats; products and sums of
// small integers are represented EXACTLY in float/double, which keeps CPU vs GPU
// agreement tight (the only divergence is FFT round-off, see THEORY section 5).
// ---------------------------------------------------------------------------
constexpr float CORE_VALUE   = 1.0f;    // buried protein interior  (+ contact)
constexpr float SKIN_PENALTY = -9.0f;   // protein surface shell    (- clash)

// ---------------------------------------------------------------------------
// DockData : everything loaded from a sample file.
//   The grid is a cube of N x N x N voxels with uniform `spacing` Angstrom per
//   voxel. Atom coordinates are in Angstrom. Both proteins are placed on the
//   SAME grid frame so a recovered translation t (in voxels) directly maps the
//   ligand onto the receptor. We store atoms as flat triples:
//   recv[3*i+{0,1,2}] = (x,y,z) of receptor atom i; lig analogously.
// ---------------------------------------------------------------------------
struct DockData {
    int    N        = 0;    // grid edge length in voxels (grid is N*N*N)
    double spacing  = 0.0;  // Angstrom per voxel (grid resolution)
    int    n_recv   = 0;    // number of receptor atoms
    int    n_lig    = 0;    // number of ligand atoms
    std::vector<float> recv;  // [3*n_recv] receptor atom coords (Angstrom)
    std::vector<float> lig;   // [3*n_lig]  ligand   atom coords (Angstrom)
    // The known-answer translation the synthetic sample was built with, in
    // signed VOXELS (tx,ty,tz). The sentinel NO_TRUTH means "not provided" (a
    // real downloaded complex has no pre-known rigid answer). main.cu reports
    // whether the recovered argmax matches it (the science check). We use a
    // far-out sentinel, not -1, because -1 is a legitimate translation value.
    static constexpr int NO_TRUTH = -100000;
    int true_tx = NO_TRUTH, true_ty = NO_TRUTH, true_tz = NO_TRUTH;
};

// ---------------------------------------------------------------------------
// flat3 : map a 3D voxel (x,y,z) to a linear index into an N*N*N grid.
//   Row-major with x fastest: idx = (z*N + y)*N + x. Shared by CPU and GPU so
//   both index the grid identically -- a classic source of CPU/GPU mismatches
//   if they ever disagree, hence one shared helper.
// ---------------------------------------------------------------------------
HD inline int flat3(int x, int y, int z, int N) {
    return (z * N + y) * N + x;
}

// ---------------------------------------------------------------------------
// wrap : reduce an index modulo N into [0, N), the PERIODIC (circular) indexing
//   the FFT uses. The FFT's correlation is inherently circular (it treats the
//   grid as a torus), so our brute-force reference must wrap identically for the
//   two to match exactly. C's % can return negatives, so we add N first.
// ---------------------------------------------------------------------------
HD inline int wrap(int i, int N) {
    int m = i % N;
    return (m < 0) ? m + N : m;
}

// ---------------------------------------------------------------------------
// world_to_voxel : convert an Angstrom coordinate to its voxel index along one
//   axis. The grid's voxel 0 starts at world coordinate `origin`; voxel size is
//   `spacing`. We floor() so a coordinate maps to the voxel that contains it.
//   Returns an int the caller must bounds-check against [0, N).
// ---------------------------------------------------------------------------
HD inline int world_to_voxel(float coord, double origin, double spacing) {
    double v = (static_cast<double>(coord) - origin) / spacing;
    int iv = static_cast<int>(v);
    if (v < 0.0 && static_cast<double>(iv) != v) iv -= 1;  // floor for negatives
    return iv;
}

// ===========================================================================
// Host-only API implemented in reference_cpu.cpp
// ===========================================================================

// Load a DockData from the text sample format (see data/README.md):
//   header:  "<n_recv> <n_lig> <N> <spacing> [<true_tx> <true_ty> <true_tz>]"
//   then n_recv lines "x y z", then n_lig lines "x y z".
// Throws std::runtime_error on a malformed file so demos fail loudly.
DockData load_dock(const std::string& path);

// Voxelize the RECEPTOR onto a real grid g (size N*N*N), writing CORE_VALUE in
// buried interior voxels and SKIN_PENALTY in a one-voxel surface shell. The
// receptor is centered in the grid. This is the CPU twin of build_receptor_grid
// on the GPU; both apply the SAME geometric rule so the grids match exactly.
//   g          : output, resized to N*N*N inside.
//   origin_out : the world coordinate (Angstrom) of voxel 0 along each axis,
//                returned so the ligand uses the identical frame.
void voxelize_receptor(const DockData& d, std::vector<float>& g, double origin_out[3]);

// Voxelize the LIGAND onto a real grid g (size N*N*N) at the grid origin (zero
// translation). Writes LIGAND_VALUE into every occupied voxel. The FFT search
// then slides this grid over all translations t at once.
void voxelize_ligand(const DockData& d, const double origin[3], std::vector<float>& g);

// Brute-force translational correlation S(t) for ALL translations t, computed
// by direct summation -- O(Ng^2), slow but transparently correct. This is the
// trusted reference the cuFFT result is checked against.
//   N    : grid edge length.
//   R, L : the receptor and ligand grids (N*N*N each).
//   score: output, resized to N*N*N; score[flat3(t)] = sum_x R(x) * L(wrap(x-t))
//          with periodic indexing, matching the FFT's circular correlation
//          exactly so CPU and GPU compare apples-to-apples.
void correlate_cpu(int N, const std::vector<float>& R, const std::vector<float>& L,
                   std::vector<float>& score);

// argmax over a score grid: returns the linear index of the largest value and
// writes its (x,y,z) voxel coordinates. Deterministic tie-break = lowest index.
std::size_t argmax_grid(int N, const std::vector<float>& score, int& bx, int& by, int& bz);
