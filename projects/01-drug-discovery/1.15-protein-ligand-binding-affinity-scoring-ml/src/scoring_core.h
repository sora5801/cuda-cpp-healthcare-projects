// ===========================================================================
// src/scoring_core.h  --  The ONE TRUE per-element math, shared by CPU & GPU
// ---------------------------------------------------------------------------
// Project 1.15 : Protein-Ligand Binding Affinity Scoring (ML)
//
// WHY THIS HEADER EXISTS (PATTERNS.md sec.2 -- the HD-macro idiom)
//   The CPU reference (reference_cpu.cpp, compiled by cl.exe) and the GPU kernels
//   (kernels.cu, compiled by nvcc) must produce BYTE-FOR-BYTE identical numbers so
//   verification can be EXACT rather than approximate. The trick: put every
//   per-element formula here ONCE as `__host__ __device__` inline functions. The
//   host compiler sees plain inline functions; nvcc sees device-callable ones.
//   Both call the SAME code, so both compute the same thing. Nothing CUDA-specific
//   (no __global__, no <<<>>>) may appear here, or the host compiler chokes.
//
// WHAT THIS PROJECT COMPUTES (the science is in ../THEORY.md)
//   A machine-learned "scoring function" predicts how tightly a small-molecule
//   LIGAND binds a PROTEIN, given the 3D coordinates of their atoms in a docked
//   pose. Classical scoring uses a physics force field; the ML approach (GNINA,
//   DeepChem AtomicConv, 3D-CNN scorers) instead LEARNS the map from structure to
//   affinity. We implement the canonical 3D-CNN scorer as a small, fully-explicit
//   forward pass so every multiply-add is visible and verifiable:
//
//     atoms --(1) voxelize--> density grid  --(2) conv3d+ReLU--> feature maps
//           --(3) global average pool--> feature vector --(4) dense--> pKd scalar
//
//   "pKd" = -log10(Kd): a higher number means a tighter binder (Kd ~ nanomolar
//   => pKd ~ 9; Kd ~ millimolar => pKd ~ 3). This is exactly the quantity PDBbind
//   tabulates for ~19,000 real complexes.
//
//   The network WEIGHTS here are fixed pseudo-random numbers (a *deterministic*
//   stand-in for a trained model -- we are teaching the GPU INFERENCE pattern, not
//   training). See THEORY "Where this sits in the real world".
//
// THE GPU PATTERN (PATTERNS.md sec.1)
//   Scoring N docked poses are N INDEPENDENT jobs (post-docking rescoring of
//   millions of poses is the real workload). The conv layer itself is the classic
//   per-output-voxel STENCIL/gather. We fuse both: one thread block scores one
//   pose; threads within the block cooperate over the grid voxels.
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// HD: the host/device decorator. Under nvcc (__CUDACC__ defined) these inline
// functions are callable from BOTH host and device; under the plain C++ compiler
// the decorator expands to nothing. This single macro is what lets one formula
// serve the CPU reference and the GPU kernel (PATTERNS.md sec.2).
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// Fixed network / grid geometry. These are COMPILE-TIME constants so loop bounds
// are known to the optimizer (the conv loops fully unroll) and so device buffers
// have fixed sizes. Kept tiny on purpose: this is a teaching forward pass, not a
// production net (a real 3D-CNN scorer uses ~24^3 grids and several conv blocks).
//
//   GRID  : the voxel grid is GRID x GRID x GRID cells.
//   CIN   : input feature channels = atom-type channels (C, N, O, S) x 2 sides
//           (protein / ligand) = 4 x 2 = 8. A channel is "this atom species, on
//           this molecule" -- letting the net learn protein-vs-ligand context.
//   COUT  : number of learned 3D convolution filters (output feature maps).
//   KSZ   : convolution kernel side length (KSZ^3 taps). Odd so it has a center.
//   NTYPES: distinct atom element types we voxelize (C,N,O,S). Others are ignored
//           in this reduced-scope teaching model (THEORY "Limitations").
// ---------------------------------------------------------------------------
constexpr int GRID   = 16;          // 16^3 = 4096 voxels per channel
constexpr int NTYPES = 4;           // C, N, O, S
constexpr int CIN    = NTYPES * 2;  // 8 input channels (4 types x {protein,ligand})
constexpr int COUT   = 8;           // 8 learned conv filters
constexpr int KSZ    = 3;           // 3x3x3 convolution

// Derived sizes (handy, and self-documenting at every use site).
constexpr int VOX_PER_CH = GRID * GRID * GRID;             // 4096
constexpr int GRID_SIZE  = CIN * VOX_PER_CH;               // input grid elements
constexpr int FEAT_SIZE  = COUT * VOX_PER_CH;              // conv output elements
constexpr int WCONV_SIZE = COUT * CIN * KSZ * KSZ * KSZ;   // conv weight count
constexpr int WDENSE_SIZE = COUT;                          // dense weights (1 per map)

// The physical extent of the grid in angstroms. Atom coordinates in [0, BOX_A)
// map linearly onto voxel indices [0, GRID). A real scorer centers the box on
// the binding pocket; our synthetic complexes are generated inside this box.
constexpr double BOX_A   = 16.0;                 // grid spans 16 A
constexpr double VOX_A   = BOX_A / GRID;         // 1.0 A per voxel
constexpr double ATOM_SIGMA = 0.8;               // Gaussian atom radius (A)

// ---------------------------------------------------------------------------
// One atom of a complex. Coordinates are in angstroms; `type` is 0..NTYPES-1
// (C,N,O,S); `is_ligand` selects the protein vs ligand half of the channels.
//   Memory: 3 doubles + 2 ints = 32 bytes. A complex is a contiguous array of
//   these; we voxelize them into the density grid.
// ---------------------------------------------------------------------------
struct Atom {
    double x, y, z;   // position in angstroms, expected within [0, BOX_A)
    int    type;      // element index 0..NTYPES-1  (0=C 1=N 2=O 3=S)
    int    is_ligand; // 1 if this atom belongs to the ligand, 0 if protein
};

// ---------------------------------------------------------------------------
// channel_of: map (atom type, protein/ligand) -> input channel index 0..CIN-1.
//   Layout: ligand channels follow the protein channels, so channel =
//   is_ligand*NTYPES + type. Used identically by the voxelizer on CPU and GPU.
// ---------------------------------------------------------------------------
HD inline int channel_of(int type, int is_ligand) {
    return is_ligand * NTYPES + type;
}

// ---------------------------------------------------------------------------
// grid_index: flatten (channel c, voxel x,y,z) -> a single offset into a
//   [CIN][GRID][GRID][GRID] row-major array. The SAME indexing must be used by
//   the voxelizer (write) and the conv (read) or the two sides disagree.
//   Order: c is outermost, then z, then y, then x (x contiguous).
// ---------------------------------------------------------------------------
HD inline int grid_index(int c, int x, int y, int z) {
    return ((c * GRID + z) * GRID + y) * GRID + x;
}

// ---------------------------------------------------------------------------
// atom_contrib: the Gaussian density an atom at distance `r2` (squared, A^2)
//   deposits into a voxel center. We model each atom as a soft Gaussian blob
//   exp(-r^2 / (2 sigma^2)) rather than a hard sphere so the density grid is
//   smooth and differentiable -- the standard choice in 3D-CNN scorers (GNINA's
//   "atom gridding"). Returns a dimensionless density weight in (0,1].
//
//   We deliberately compute exp() the SAME way on both sides (std::exp / exp are
//   IEEE-correctly-rounded for these libm/CUDA paths at this magnitude), and we
//   cut the tail at r2 > CUTOFF^2 so distant atoms contribute exactly 0 -- making
//   the sum finite-support and the CPU/GPU voxel sums identical term-for-term.
// ---------------------------------------------------------------------------
constexpr double GAUSS_CUTOFF = 3.0;                 // ignore atoms > 3 A from a voxel
constexpr double GAUSS_CUTOFF2 = GAUSS_CUTOFF * GAUSS_CUTOFF;
constexpr double GAUSS_DENOM   = 2.0 * ATOM_SIGMA * ATOM_SIGMA;

HD inline double atom_contrib(double r2) {
    // Hard cutoff first: beyond it the Gaussian is < 1e-3 and we drop it exactly
    // so both implementations sum the identical set of terms (THEORY "verify").
    if (r2 > GAUSS_CUTOFF2) return 0.0;
#ifdef __CUDACC__
    return exp(-r2 / GAUSS_DENOM);   // device exp (double precision)
#else
    return exp(-r2 / GAUSS_DENOM);   // host <cmath> exp
#endif
}

// ---------------------------------------------------------------------------
// relu: the rectified-linear activation max(0,x). The one nonlinearity in this
//   tiny net; it is what makes a CNN more than a single linear map. Trivial, but
//   shared so CPU and GPU clip identically.
// ---------------------------------------------------------------------------
HD inline double relu(double x) { return x > 0.0 ? x : 0.0; }

// ---------------------------------------------------------------------------
// lcg_weight: a DETERMINISTIC pseudo-random "learned" weight in [-1, 1).
//   We have no trained model to ship, so we synthesize fixed weights from a
//   64-bit linear-congruential generator seeded by the weight's flat index. The
//   point: identical on CPU and GPU, reproducible across runs, and spread around
//   zero like real initialized/trained conv weights. (A real project loads
//   trained weights from a file -- see THEORY "Where this sits in the real
//   world".) The LCG constants are the well-known Numerical-Recipes values.
// ---------------------------------------------------------------------------
HD inline double lcg_weight(uint64_t index) {
    // Mix the index so adjacent weights are uncorrelated, then run one LCG step.
    uint64_t s = index * 6364136223846793005ULL + 1442695040888963407ULL;
    s ^= s >> 33;                                   // xorshift finalizer
    s *= 0xff51afd7ed558ccdULL;
    s ^= s >> 33;
    // Take the top 53 bits -> a double in [0,1), then rescale to [-1,1).
    const double u = (double)(s >> 11) * (1.0 / 9007199254740992.0);  // 2^53
    return 2.0 * u - 1.0;
}
