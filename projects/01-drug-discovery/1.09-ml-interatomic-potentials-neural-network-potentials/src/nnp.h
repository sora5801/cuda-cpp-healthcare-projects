// ===========================================================================
// src/nnp.h  --  Shared (host + device) Neural Network Potential core
// ---------------------------------------------------------------------------
// Project 1.9 : ML Interatomic Potentials (Neural Network Potentials)
//
// WHAT THIS FILE IS (the single source of physics for BOTH the CPU and GPU)
//   This header holds the *per-atom* math of a Behler-Parrinello-style neural
//   network potential (NNP), written as `__host__ __device__` inline functions.
//   reference_cpu.cpp (compiled by the host C++ compiler) and kernels.cu
//   (compiled by nvcc) BOTH include it, so the CPU reference and the GPU kernel
//   evaluate byte-for-byte identical arithmetic -> verification is exact, not
//   approximate (PATTERNS.md sec 2, "the shared __host__ __device__ core").
//
//   NNP_HD expands to `__host__ __device__` under nvcc and to nothing under the
//   plain host compiler (which has never heard of those keywords). Keep this
//   header free of CUDA-only constructs (no __global__, no <<<>>>), so cl.exe
//   can compile it too.
//
// THE SCIENCE IN ONE PARAGRAPH (see ../THEORY.md for the full derivation)
//   A neural network potential learns the potential energy surface E(R) of a set
//   of atoms from quantum-chemistry reference data. Behler & Parrinello's key
//   trick (2007): the TOTAL energy is a sum of ATOMIC contributions,
//   E = sum_i E_i, and each E_i is produced by a small neural network whose
//   input is a vector of "atom-centered symmetry functions" (ACSF) -- numbers
//   that describe atom i's local chemical environment in a way that is invariant
//   to translation, rotation, and permutation of identical neighbors. Because
//   each E_i depends only on neighbors within a cutoff radius, the model is
//   short-ranged and embarrassingly parallel: one GPU thread can own one atom.
//
//   THIS TEACHING VERSION is deliberately reduced (CLAUDE.md sec 13): a single
//   atom type, RADIAL (G2) symmetry functions only, and a small fixed MLP with
//   pre-baked weights. The structure (descriptor -> per-atom MLP -> sum) is
//   exactly that of ANI/Behler-Parrinello; THEORY.md explains what production
//   NNPs (ANI, NequIP, MACE) add (angular terms, multiple elements, equivariant
//   message passing, learned weights, analytic forces).
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

#include <cmath>     // std::sqrt, std::exp, std::tanh, std::cos  (host side)

// ---------------------------------------------------------------------------
// NNP_HD: the host/device portability shim (PATTERNS.md sec 2).
//   Under nvcc the macro __CUDACC__ is defined, so the functions below become
//   callable from BOTH host and device. Under cl.exe/g++ it expands to nothing.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define NNP_HD __host__ __device__
#else
#define NNP_HD
#endif

// ---------------------------------------------------------------------------
// FIXED MODEL DIMENSIONS (compile-time constants).
//   Making these compile-time lets the GPU keep the whole model in constant
//   memory and lets the inner loops unroll. They are intentionally small so the
//   demo is fast and the arithmetic is easy to follow by hand.
//
//   N_DESC  : number of radial symmetry functions per atom (the MLP input size).
//             Each one is a Gaussian "shell" peaked at a different distance Rs.
//   N_HID   : neurons in each of the two hidden layers.
//   The network shape is therefore:  N_DESC -> N_HID -> N_HID -> 1.
// ---------------------------------------------------------------------------
constexpr int N_DESC = 8;    // radial descriptors per atom (G2 with 8 shells)
constexpr int N_HID  = 16;   // hidden-layer width (both hidden layers)

// ---------------------------------------------------------------------------
// AtomicNet: the weights/biases of the small per-atom multilayer perceptron.
//   Layout is plain dense matrices in row-major order. In a real NNP these are
//   LEARNED by fitting to DFT/CCSD(T) energies; here they are generated
//   deterministically (scripts/make_synthetic.py mirrors this) so the demo is
//   self-contained and reproducible. The numbers are physically meaningless --
//   what is real and worth studying is the COMPUTATION, not these constants.
//
//   Shapes:
//     w1 : [N_HID][N_DESC]  (hidden layer 1 weights)   b1 : [N_HID]
//     w2 : [N_HID][N_HID]   (hidden layer 2 weights)   b2 : [N_HID]
//     w3 : [N_HID]          (output layer weights)     b3 : scalar
//   Stored flattened (row-major) so the same struct works on host and device.
// ---------------------------------------------------------------------------
struct AtomicNet {
    double w1[N_HID * N_DESC];   // layer 1: maps descriptor -> hidden
    double b1[N_HID];
    double w2[N_HID * N_HID];    // layer 2: hidden -> hidden
    double b2[N_HID];
    double w3[N_HID];            // output: hidden -> scalar atomic energy
    double b3;
};

// ---------------------------------------------------------------------------
// AcsfParams: the radial symmetry-function hyperparameters.
//   A radial (G2) symmetry function for atom i is
//       G2_s(i) = sum_{j != i, r_ij < Rc}  exp(-eta * (r_ij - Rs[s])^2) * fc(r_ij)
//   It answers: "how much neighbor density sits near distance Rs[s] from atom i?"
//   Sweeping Rs over several shells gives a smooth radial fingerprint of the
//   local environment. Rc is the cutoff; eta sets each Gaussian's width.
// ---------------------------------------------------------------------------
struct AcsfParams {
    double Rc;            // cutoff radius (Angstrom): neighbors beyond are ignored
    double eta;           // Gaussian width parameter (1/Angstrom^2)
    double Rs[N_DESC];    // the N_DESC shell centers (Angstrom)
};

// ---------------------------------------------------------------------------
// cutoff_fc: the smooth cosine cutoff function fc(r) (Behler 2011).
//   fc(r) = 0.5 * (cos(pi * r / Rc) + 1)   for r <= Rc,   0 otherwise.
//   WHY: it tapers each neighbor's contribution smoothly to zero AT the cutoff,
//   so an atom drifting across Rc does not cause an energy discontinuity (which
//   would make forces blow up). This is the single most important detail that
//   makes symmetry functions usable in dynamics.
// ---------------------------------------------------------------------------
NNP_HD inline double cutoff_fc(double r, double Rc) {
    if (r >= Rc) return 0.0;                 // outside the cutoff: no contribution
    const double pi = 3.14159265358979323846;
    return 0.5 * (cos(pi * r / Rc) + 1.0);   // 1 at r=0, smoothly -> 0 at r=Rc
}

// ---------------------------------------------------------------------------
// compute_descriptor: fill `desc[N_DESC]` for atom `i` from the neighbor list.
//   Inputs:
//     pos        : flat array of 3*n doubles, atom k at (pos[3k],pos[3k+1],pos[3k+2])
//     n          : number of atoms
//     i          : the atom whose environment we describe
//     p          : the ACSF hyperparameters
//   Output: desc[s] = G2_s(i) for s = 0..N_DESC-1.
//
//   COMPLEXITY: O(n) per atom (it scans every other atom and keeps those inside
//   Rc). A production code uses a spatial cell list to make this O(neighbors);
//   we keep the brute-force scan because it is trivially correct and the demo is
//   tiny -- THEORY.md "real world" explains the cell-list optimization.
//
//   DETERMINISM: neighbors are summed in ascending index order on BOTH host and
//   device (the GPU thread runs this exact loop), so the descriptor is identical
//   bit-for-bit. This is what makes the GPU-vs-CPU check exact.
// ---------------------------------------------------------------------------
NNP_HD inline void compute_descriptor(const double* pos, int n, int i,
                                      const AcsfParams& p, double* desc) {
    // Start every shell accumulator at zero.
    for (int s = 0; s < N_DESC; ++s) desc[s] = 0.0;

    const double xi = pos[3 * i + 0];
    const double yi = pos[3 * i + 1];
    const double zi = pos[3 * i + 2];

    // Scan all other atoms; accumulate those inside the cutoff into every shell.
    for (int j = 0; j < n; ++j) {
        if (j == i) continue;                 // an atom is not its own neighbor
        const double dx = pos[3 * j + 0] - xi;
        const double dy = pos[3 * j + 1] - yi;
        const double dz = pos[3 * j + 2] - zi;
        const double r2 = dx * dx + dy * dy + dz * dz;
        const double r  = sqrt(r2);           // interatomic distance r_ij
        if (r >= p.Rc) continue;              // outside cutoff -> skip
        const double fc = cutoff_fc(r, p.Rc); // smooth taper at the cutoff

        // Add this neighbor's Gaussian contribution to each radial shell.
        for (int s = 0; s < N_DESC; ++s) {
            const double d = r - p.Rs[s];                  // distance from shell center
            desc[s] += exp(-p.eta * d * d) * fc;           // G2 contribution
        }
    }
}

// ---------------------------------------------------------------------------
// atomic_energy_from_desc: run the per-atom MLP on a descriptor vector.
//   E_i = w3 . tanh( w2 . tanh( w1 . desc + b1 ) + b2 ) + b3
//   tanh is the classic smooth, bounded activation used by Behler-Parrinello
//   atomic networks; smoothness matters because forces are derivatives of E.
//
//   We keep activations in local arrays (registers/local memory on the GPU).
//   With N_DESC=8, N_HID=16 the whole forward pass is a handful of FMAs -- tiny,
//   which is the point: the per-atom net is cheap; the win is doing all atoms at
//   once.
// ---------------------------------------------------------------------------
NNP_HD inline double atomic_energy_from_desc(const double* desc, const AtomicNet& net) {
    double h1[N_HID];   // hidden layer 1 activations
    double h2[N_HID];   // hidden layer 2 activations

    // ---- hidden layer 1:  h1 = tanh(W1 * desc + b1) ----
    for (int k = 0; k < N_HID; ++k) {
        double acc = net.b1[k];
        for (int s = 0; s < N_DESC; ++s)
            acc += net.w1[k * N_DESC + s] * desc[s];   // row k of W1 dotted with desc
        h1[k] = tanh(acc);
    }

    // ---- hidden layer 2:  h2 = tanh(W2 * h1 + b2) ----
    for (int k = 0; k < N_HID; ++k) {
        double acc = net.b2[k];
        for (int m = 0; m < N_HID; ++m)
            acc += net.w2[k * N_HID + m] * h1[m];      // row k of W2 dotted with h1
        h2[k] = tanh(acc);
    }

    // ---- output layer:  E_i = w3 . h2 + b3  (linear, scalar) ----
    double e = net.b3;
    for (int k = 0; k < N_HID; ++k)
        e += net.w3[k] * h2[k];
    return e;
}

// ---------------------------------------------------------------------------
// atomic_energy: the full per-atom pipeline (descriptor + MLP) for atom i.
//   This is THE function each GPU thread calls for its own atom, and the one the
//   CPU reference loops over -- the single shared definition that guarantees the
//   two paths agree. Returns E_i (energy units of the model, arbitrary here).
// ---------------------------------------------------------------------------
NNP_HD inline double atomic_energy(const double* pos, int n, int i,
                                   const AcsfParams& p, const AtomicNet& net) {
    double desc[N_DESC];
    compute_descriptor(pos, n, i, p, desc);     // step 1: local-environment fingerprint
    return atomic_energy_from_desc(desc, net);  // step 2: small neural net -> E_i
}
