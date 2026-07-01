// ===========================================================================
// src/mrf_core.h  --  The ONE TRUE per-element math, shared by CPU and GPU
// ---------------------------------------------------------------------------
// Project 4.21 : MR Fingerprinting Reconstruction
//                (see ../THEORY.md and the catalog deep-dive for the "why")
//
// WHY THIS HEADER EXISTS  (PATTERNS.md §2, the "__host__ __device__ core" idiom)
//   MR Fingerprinting (MRF) has three numerical kernels that BOTH the CPU
//   reference (reference_cpu.cpp, compiled by cl.exe) and the GPU code
//   (kernels.cu, compiled by nvcc) must compute *byte-for-byte identically*:
//
//     (1) SIMULATE one dictionary atom -- the signal time course a tissue with
//         relaxation times (T1, T2) would produce under the pseudorandom
//         acquisition. This is the "Bloch simulation" step, reduced here to a
//         closed-form inversion-recovery / variable-flip-angle model so the
//         teaching version needs no ODE integrator (THEORY.md §"The math").
//
//     (2) L2-NORMALIZE a length-T time course to unit energy, so that matching
//         becomes a pure cosine (direction) comparison independent of the
//         unknown proton density / receive gain (a scalar per voxel).
//
//     (3) The MATCH score itself is just an inner product of two unit vectors
//         (a cosine in [-1, 1]); the heavy lifting -- forming ALL voxel×atom
//         inner products at once -- is a matrix-matrix product handed to cuBLAS
//         SGEMM in kernels.cu. This header owns the *scalar* recipe each entry
//         of that product implements, so the library result is not a black box.
//
//   Writing each scalar formula EXACTLY ONCE, here, tagged `__host__ __device__`,
//   and calling it from both sides guarantees the CPU baseline and the GPU
//   kernels produce the same dictionary and the same normalization -- which is
//   what makes the "GPU == CPU" verification in main.cu meaningful.
//
//   Keep this header free of CUDA-only constructs (no __global__, no <<<>>>).
//   It must be includable by the plain host C++ compiler. The MRF_HD macro
//   below evaporates to nothing under cl.exe and becomes `__host__ __device__`
//   under nvcc -- that is the whole trick.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cu  (both include this file).
// ===========================================================================
#pragma once

#include <cmath>     // std::exp, std::sqrt, std::sin, std::cos, std::fabs
#include <cstddef>   // std::size_t

// ---------------------------------------------------------------------------
// MRF_HD : the host/device portability shim.
//   * Under nvcc (__CUDACC__ defined) a function tagged MRF_HD is compiled for
//     BOTH the CPU (host) and the GPU (device), so the identical code object is
//     callable from main.cu's host code and from inside a __global__ kernel.
//   * Under a plain host compiler the decorators do not exist, so MRF_HD must
//     expand to nothing -- the function is then just an ordinary inline.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define MRF_HD __host__ __device__
#else
#define MRF_HD
#endif

namespace mrf {

// ===========================================================================
// SECTION A -- the acquisition schedule and the (T1, T2) grid
// ===========================================================================
//
// An MRF scan plays out T repetitions ("time frames"). At frame t the scanner
// tips the magnetization by a pseudorandom FLIP ANGLE alpha[t] and waits a
// pseudorandom repetition time TR[t]. Different tissues (different T1, T2)
// respond to this exact schedule with a different signal *shape* over the T
// frames -- their "fingerprint". Because the schedule is fixed and shared by
// every voxel, we store it once and hand it to the simulator below.
//
// The DICTIONARY is a grid of candidate tissue parameters: a set of T1 values
// crossed with a set of T2 values (with the physical constraint T2 <= T1). Each
// (T1, T2) pair yields one length-T fingerprint atom. Real MRF dictionaries hold
// 10^5-10^6 atoms; our teaching sample uses a few hundred, which is enough to
// see the method work and small enough to run offline in milliseconds.

// The proton relaxation of a single isochromat during one TR, in the simplified
// model used here. We deliberately avoid a full Bloch ODE integrator; instead we
// use a closed-form recursion that captures the essential physics a learner
// needs (T1 recovery + T2 decay + RF tipping) and, crucially, gives every atom a
// DISTINCT, monotone-in-(T1,T2) shape so matching is well posed. THEORY.md
// §"The math" derives this and names exactly what a production Bloch/EPG
// simulator adds (off-resonance, slice profile, B1+, full complex EPG states).

// bloch_step: advance the longitudinal magnetization Mz through one frame and
//   return the transverse signal that frame samples.
//
//   Physical story of one frame t:
//     1. An RF pulse of flip angle alpha tips longitudinal magnetization into
//        the transverse plane; the SIGNAL we read out is proportional to
//        Mz_before * sin(alpha) (the projected transverse component).
//     2. That transverse component then decays by exp(-TE/T2) before readout
//        (TE, the echo time, is a small fixed fraction of TR here).
//     3. The remaining longitudinal magnetization Mz_before*cos(alpha) recovers
//        toward equilibrium (M0 = 1) over the rest of the TR:
//            Mz_after = 1 - (1 - Mz_before*cos(alpha)) * exp(-TR/T1).
//   The returned pair (signal, Mz_after) lets the caller iterate frame by frame.
//
//   Parameters (all SI-ish, times in milliseconds):
//     mz_before : longitudinal magnetization entering this frame, in [-1, 1]
//     alpha_rad : flip angle for this frame, radians
//     tr_ms     : repetition time for this frame, ms  (> 0)
//     te_ms     : echo time (readout delay) for this frame, ms  (0 < te < tr)
//     t1_ms     : tissue longitudinal relaxation time, ms  (> 0)
//     t2_ms     : tissue transverse   relaxation time, ms  (0 < t2 <= t1)
//   Returns the transverse signal sampled this frame (a real number; in a real
//   scan it is complex, but our real-valued teaching model keeps the algebra
//   transparent). Writes the post-frame Mz through *mz_after.
struct BlochFrame {
    double signal;    // transverse magnitude read out this frame
    double mz_after;  // longitudinal magnetization handed to the next frame
};

MRF_HD inline BlochFrame bloch_step(double mz_before, double alpha_rad,
                                    double tr_ms, double te_ms,
                                    double t1_ms, double t2_ms) {
    BlochFrame f;
    // (1) tip into the transverse plane, (2) let it decay for TE by T2.
    double transverse = mz_before * std::sin(alpha_rad);
    f.signal = transverse * std::exp(-te_ms / t2_ms);
    // (3) the untipped longitudinal part recovers toward M0 = 1 over the TR.
    double mz_untipped = mz_before * std::cos(alpha_rad);
    f.mz_after = 1.0 - (1.0 - mz_untipped) * std::exp(-tr_ms / t1_ms);
    return f;
}

// ===========================================================================
// SECTION B -- simulate one full dictionary atom (a fingerprint)
// ===========================================================================

// simulate_atom: fill out[0..T-1] with the length-T fingerprint of a tissue
//   with relaxation times (t1_ms, t2_ms) under the shared schedule.
//
//   Inputs:
//     alpha : [T] flip angles per frame (radians)
//     tr    : [T] repetition times per frame (ms)
//     te    : [T] echo times per frame (ms)
//     T     : number of frames (time-course length)
//     t1_ms, t2_ms : this atom's relaxation times (ms)
//   Output:
//     out   : [T] the (un-normalized) simulated transverse signal per frame.
//
//   The scan starts from an INVERSION (Mz = -1): an initial 180 degree pulse
//   flips the magnetization negative, which is what gives MRF its strong T1
//   sensitivity early in the train. We then iterate bloch_step frame by frame.
//   This is the loop the CPU reference runs for every atom and the GPU runs in
//   parallel (one thread per atom in the dictionary-build kernel).
MRF_HD inline void simulate_atom(const double* alpha, const double* tr,
                                 const double* te, int T,
                                 double t1_ms, double t2_ms, float* out) {
    double mz = -1.0;                       // start inverted (the 180 deg prep)
    for (int t = 0; t < T; ++t) {
        BlochFrame f = bloch_step(mz, alpha[t], tr[t], te[t], t1_ms, t2_ms);
        out[t] = static_cast<float>(f.signal);
        mz = f.mz_after;                    // carry longitudinal state forward
    }
}

// ===========================================================================
// SECTION C -- L2 normalization (make matching a pure cosine)
// ===========================================================================
//
// A voxel's measured signal is (fingerprint shape) * (unknown scalar): proton
// density times receive gain. We do not know that scalar, and we do not need
// it to identify the TISSUE -- only the SHAPE matters. Normalizing each time
// course to unit L2 norm removes the scalar, so the match score reduces to the
// cosine of the angle between voxel and atom: 1.0 means identical shape. The
// leftover scale (the norm ratio at the winning atom) is the proton-density
// estimate, recovered separately.

// l2_norm: Euclidean norm sqrt(sum v[i]^2) of a length-n vector.
//   Returns 0 for the all-zero vector; callers guard the divide.
MRF_HD inline float l2_norm(const float* v, int n) {
    double s = 0.0;                          // accumulate in double for accuracy
    for (int i = 0; i < n; ++i) s += static_cast<double>(v[i]) * v[i];
    return static_cast<float>(std::sqrt(s));
}

// normalize_inplace: scale v to unit L2 norm (no-op on a zero vector).
//   Both the dictionary atoms and the voxel signals are normalized with THIS
//   exact routine on both CPU and GPU, so the inner products that cuBLAS forms
//   are true cosines and match the CPU baseline to float precision.
MRF_HD inline void normalize_inplace(float* v, int n) {
    float nrm = l2_norm(v, n);
    if (nrm > 0.0f) {
        float inv = 1.0f / nrm;
        for (int i = 0; i < n; ++i) v[i] *= inv;
    }
}

// dot: plain inner product of two length-n float vectors (accumulated in
//   double). This is the SCALAR that each entry of the big cuBLAS SGEMM
//   computes; the CPU reference calls it directly, and kernels.cu explains at
//   the SGEMM call site how the library forms all V*D of them at once.
MRF_HD inline float dot(const float* a, const float* b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += static_cast<double>(a[i]) * b[i];
    return static_cast<float>(s);
}

}  // namespace mrf
