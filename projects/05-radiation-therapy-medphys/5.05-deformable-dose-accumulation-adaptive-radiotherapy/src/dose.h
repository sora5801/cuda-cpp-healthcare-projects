// ===========================================================================
// src/dose.h  --  Shared (host + device) core of deformable DOSE accumulation
// ---------------------------------------------------------------------------
// Project 5.5 : Deformable Dose Accumulation & Adaptive Radiotherapy
//               (reduced-scope 2-D teaching version)
//
// WHAT THIS HEADER OWNS  (the NEW physics of 5.5, on top of Demons DIR)
//   After demons.h has produced a displacement field u(x) that maps the planning
//   frame to today's anatomy, we must move the DOSE the same way and add it up.
//   Two operations, both per-voxel and both here as __host__ __device__ inlines
//   so the CPU reference and the GPU kernel compute byte-identical numbers:
//
//     1. warp_dose  -- DEFORMABLE DOSE WARP ("summation of deformed doses").
//        For each planning-frame voxel x, the dose actually deposited during a
//        fraction lives at the deformed position x+u(x) on today's grid. So we
//        GATHER the delivered-dose map at (x+ux, y+uy) via bilinear interpolation
//        -- exactly the catalog's "custom CUDA trilinear warp for dose mapping"
//        (bilinear = the 2-D case). Warping every fraction back into the common
//        planning frame and summing gives the anatomically-correct TOTAL dose.
//        (The alternative "energy/mass transfer" method, which conserves total
//        energy under compression, is discussed in THEORY §7; interpolation is
//        the standard first method and what most tools default to.)
//
//     2. dvh_bin / DVH accumulation -- DOSE-VOLUME HISTOGRAM.
//        The clinical summary of a dose map is the DVH: what fraction of a
//        structure's volume receives at least dose d. We build it by binning
//        every voxel's dose. Doing that in PARALLEL means many threads add into
//        the same histogram bin -- a reduction. FLOAT atomicAdd would reorder the
//        sums and make stdout non-reproducible (PATTERNS.md §3), so we count in
//        INTEGERS (one count per voxel) with atomicAdd on unsigned ints: integer
//        adds commute, so the histogram is deterministic AND matches the CPU
//        exactly. This is the "CUDA atomic adds for accumulated dose histogram"
//        from the catalog, done the deterministic way.
//
// WHY THE PHYSICS LIVES HERE (the __host__ __device__ idiom, PATTERNS.md §2)
//   DS_HD = __host__ __device__ under nvcc, nothing under the host compiler, so
//   reference_cpu.cpp and kernels.cu run the same formulas. Keep this header free
//   of CUDA-only constructs (no __global__, no <<<>>>) so cl.exe/g++ can include
//   it. dm_bilinear() (the gather) is reused from demons.h -- one warp routine
//   serves both image registration and dose mapping.
//
// READ THIS AFTER: demons.h. Then reference_cpu.cpp, kernels.cu.
// ===========================================================================
#pragma once

#include "demons.h"   // dm_bilinear (the shared bilinear gather) + DM_HD idiom

// DS_HD mirrors DM_HD: __host__ __device__ under nvcc, empty on the host side.
// (Defined separately so this header is self-documenting even read in isolation.)
#ifdef __CUDACC__
#define DS_HD __host__ __device__
#else
#define DS_HD
#endif

// Number of dose bins in the DVH. The demo normalizes dose to [0, DVH_MAX] Gy and
// splits it into this many equal bins. 32 is enough to see the histogram shape at
// a glance in stdout while staying tiny and deterministic.
#define DVH_BINS 32

// The top of the DVH dose axis, in Gray (Gy). Our synthetic prescription peaks at
// ~2 Gy per fraction; a couple of accumulated fractions stay under this ceiling,
// so no dose is clipped. Any voxel dose >= DVH_MAX lands in the last bin.
#define DVH_MAX 6.0

// ---------------------------------------------------------------------------
// warp_dose_at -- one voxel of the deformable dose warp.
//   Returns the delivered dose that maps to planning-frame voxel (x,y): the
//   delivered-dose map `dose` sampled at the deformed location (x+ux, y+uy) via
//   the SHARED bilinear gather. This is a pure per-voxel read -> the GPU version
//   is one thread per output voxel, no atomics, no races (each thread writes a
//   distinct output element). Complexity O(1) per voxel.
//
//   ux,uy : the displacement field from Demons DIR (planning -> today), [ny*nx].
//   dose  : the dose delivered on TODAY's grid (what the linac put down), [ny*nx].
// ---------------------------------------------------------------------------
DS_HD inline double warp_dose_at(const double* dose,
                                 const double* ux, const double* uy,
                                 int x, int y, int nx, int ny) {
    const int i = y * nx + x;                          // this voxel's index
    const double px = (double)x + ux[i];               // deformed x coordinate
    const double py = (double)y + uy[i];               // deformed y coordinate
    return dm_bilinear(dose, px, py, nx, ny);          // gather (clamp-to-edge)
}

// ---------------------------------------------------------------------------
// dvh_bin -- map a dose value (Gy) to a DVH bin index in [0, DVH_BINS-1].
//   Linear binning: bin = floor(d / DVH_MAX * DVH_BINS), clamped to the valid
//   range so that d<0 (never happens for dose, but be safe) and d>=DVH_MAX both
//   fold to an edge bin. Marked DS_HD so the CPU histogram and the GPU histogram
//   assign every voxel to the SAME bin -> the two histograms are bit-identical.
//
//   Determinism note: this returns an INTEGER bin, and the accumulation adds 1
//   (an integer) per voxel. Integer addition is associative and commutative, so
//   the parallel atomic accumulation gives the same counts regardless of thread
//   order -- unlike a float sum (PATTERNS.md §3).
// ---------------------------------------------------------------------------
DS_HD inline int dvh_bin(double d) {
    double t = d / DVH_MAX;                 // fraction of the dose axis, in [0,1]
    if (t < 0.0) t = 0.0;
    int b = (int)(t * (double)DVH_BINS);    // floor into a bin
    if (b < 0)          b = 0;
    if (b >= DVH_BINS)  b = DVH_BINS - 1;   // clamp the top edge into the last bin
    return b;
}
