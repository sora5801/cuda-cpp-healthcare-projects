// ===========================================================================
// src/product_core.h  --  The ONE TRUE per-product physics, shared CPU<->GPU
// ---------------------------------------------------------------------------
// Project 1.18 : Fragment / Combinatorial Library Enumeration
//
// WHY THIS HEADER EXISTS  (PATTERNS.md sec.2: the __host__ __device__ core)
//   A combinatorial library is the Cartesian PRODUCT of several reactant slots:
//   pick one building block (a "synthon") from slot 0, one from slot 1, ... and
//   you get one product molecule. With slot sizes s0,s1,...,s_{R-1} the library
//   has  N = s0*s1*...*s_{R-1}  products -- this is how a few hundred building
//   blocks explode into BILLIONS of compounds (Enamine REAL: >6e9). Every
//   product is INDEPENDENT, so the work is embarrassingly parallel: one GPU
//   thread per product index. That is the whole teaching point of this project.
//
//   For EACH product we must (a) compute its physicochemical properties and
//   (b) decide whether it passes a drug-likeness filter (Lipinski + Veber).
//   To guarantee the CPU reference and the GPU kernel agree BIT-FOR-BIT, the
//   per-product math lives here ONCE as __host__ __device__ inline functions:
//     - the host reference (reference_cpu.cpp) loops them over all products;
//     - the GPU kernel (kernels.cu) calls them from one thread per product.
//   Same source -> same IEEE operations -> exact agreement (see THEORY "verify").
//
// THE SCIENCE SHORTCUT THAT MAKES ENUMERATION TRACTABLE  (additivity)
//   We never assemble the full product molecule here. Instead we use the
//   GROUP-CONTRIBUTION approximation: several key descriptors are (to good
//   accuracy) ADDITIVE over the fragments that make up a molecule --
//     * molecular weight (MW)         : exactly additive minus the atoms lost
//                                       when two fragments bond (we fold the
//                                       "lost mass" into each synthon's value);
//     * Crippen logP (cLogP)          : a SUM of per-atom contributions
//                                       (Wildman & Crippen, 1999) -> additive;
//     * Ertl TPSA (polar surface area): a SUM of per-fragment polar contributions
//                                       (Ertl, 2000) -> additive;
//     * H-bond donors / acceptors     : integer COUNTS -> additive.
//   So a product's descriptor is just the SUM of its synthons' contributions.
//   This is exactly why combinatorial pre-filtering is cheap and GPU-friendly,
//   and it is a real technique (RDKit computes the same descriptors, just on the
//   assembled molecule). The honest caveat -- that true enumeration also needs
//   SMARTS reaction matching to know which synthons can react -- is in THEORY
//   under "Where this sits in the real world".
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh. The data model (how synthons
// are stored) is in reference_cpu.h; this header is only the per-product math.
// ===========================================================================
#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// HD: the host/device decorator macro (PATTERNS.md sec.2 idiom).
//   When this header is compiled by nvcc (which defines __CUDACC__) the inline
//   functions become callable from BOTH host and device. When compiled by the
//   plain host C++ compiler (reference_cpu.cpp), the decorator expands to
//   nothing, so the same functions are ordinary host functions. One source of
//   truth, two compilers, identical math.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// Number of reactant slots in the combinatorial scheme. Three slots (a common
// "A + B + C" three-component reaction, e.g. Ugi/Groebke-Blackburn) is enough
// to show the mixed-radix decode while staying easy to follow. It is a compile-
// time constant so the per-product loops fully unroll on the GPU.
#define N_SLOTS 3

// Number of additive descriptors we accumulate per synthon. Order matters and
// is fixed here so the loader, the kernel, and the reference all agree:
//   [0] MW    -- molecular weight contribution        (g/mol)
//   [1] cLogP -- Crippen logP contribution            (dimensionless)
//   [2] TPSA  -- topological polar surface area        (Angstrom^2)
//   [3] HBD   -- hydrogen-bond DONOR count             (integer, stored as double)
//   [4] HBA   -- hydrogen-bond ACCEPTOR count          (integer, stored as double)
#define N_DESC 5

// Symbolic indices into a descriptor vector -- never use bare 0..4 in logic.
enum DescIndex { D_MW = 0, D_CLOGP = 1, D_TPSA = 2, D_HBD = 3, D_HBA = 4 };

// ---------------------------------------------------------------------------
// Lipinski "Rule of Five" + Veber thresholds for ORAL drug-likeness.
//   A product PASSES if ALL of these hold (these are the canonical cut-offs):
//     MW    <= 500 g/mol          (Lipinski)
//     cLogP <= 5                   (Lipinski)
//     HBD   <= 5                   (Lipinski)
//     HBA   <= 10                  (Lipinski)
//     TPSA  <= 140 Angstrom^2      (Veber -- proxy for oral bioavailability)
//   We expose them as constants so README/THEORY and the code never drift.
//   (RotatableBonds<=10 is the other Veber rule; we omit it here because it is
//   not simply additive across a forming bond -- see THEORY "real world".)
// ---------------------------------------------------------------------------
#define LIMIT_MW     500.0
#define LIMIT_CLOGP    5.0
#define LIMIT_HBD      5.0
#define LIMIT_HBA     10.0
#define LIMIT_TPSA   140.0

// ---------------------------------------------------------------------------
// accumulate_descriptors: sum the per-synthon descriptor vectors of the chosen
// building blocks into the product's descriptor vector.
//   slot_desc : pointer to a flat [N_SLOTS * (max_slot_size) * N_DESC] array...
//               but to keep this core dependency-free we pass the THREE already-
//               resolved synthon descriptor pointers instead (see callers). To
//               avoid pointer juggling in the core, callers hand us the three
//               base pointers via the small struct below.
//   out[N_DESC]: filled with the elementwise sum.
// We implement the sum over a tiny fixed-size struct so the SAME code runs on
// host and device with no STL. See ProductInputs just below.
// ---------------------------------------------------------------------------

// A bundle of pointers to the N_SLOTS chosen synthons' descriptor rows. Each
// row is N_DESC doubles. Using a struct keeps the HD function signature simple
// and identical on both sides.
struct ProductInputs {
    const double* row[N_SLOTS];   // row[k] -> &desc_of_chosen_synthon_in_slot_k[0]
};

// Sum the chosen synthons' descriptors elementwise -> the product descriptor.
//   Why a plain loop: N_SLOTS and N_DESC are compile-time constants, so nvcc
//   unrolls both loops into straight-line FMA-free adds; the host compiler does
//   the same. Identical operation order on both sides => identical doubles.
HD inline void accumulate_descriptors(const ProductInputs& in, double out[N_DESC]) {
    // Initialise the accumulator to zero (no FMA, just additions, so the sum is
    // associative-order-fixed and reproducible).
    for (int d = 0; d < N_DESC; ++d) out[d] = 0.0;
    // Add each slot's contribution. Outer loop over the 3 slots, inner over the
    // 5 descriptors. The fixed bounds let the compiler fully unroll.
    for (int k = 0; k < N_SLOTS; ++k) {
        const double* r = in.row[k];          // this slot's chosen synthon row
        for (int d = 0; d < N_DESC; ++d) {
            out[d] += r[d];                   // additive group contribution
        }
    }
}

// ---------------------------------------------------------------------------
// passes_filter: apply the Lipinski + Veber thresholds to a product descriptor.
//   Returns 1 if the product is "drug-like" (passes ALL rules), else 0.
//   We return an INT (not bool) on purpose: the GPU reduction counts passes
//   with INTEGER atomics, and integer addition is associative, so the total is
//   deterministic and matches the CPU exactly (PATTERNS.md sec.3 rule 2).
//   The comparisons are pure double <= double, identical on host and device.
// ---------------------------------------------------------------------------
HD inline int passes_filter(const double desc[N_DESC]) {
    // Each clause is a hard threshold. Short-circuit && means a product that
    // already failed one rule does not waste comparisons -- but the RESULT is
    // the same regardless of evaluation order (no side effects).
    const int ok =
        (desc[D_MW]    <= LIMIT_MW)    &&   // Lipinski: weight
        (desc[D_CLOGP] <= LIMIT_CLOGP) &&   // Lipinski: lipophilicity
        (desc[D_HBD]   <= LIMIT_HBD)   &&   // Lipinski: H-bond donors
        (desc[D_HBA]   <= LIMIT_HBA)   &&   // Lipinski: H-bond acceptors
        (desc[D_TPSA]  <= LIMIT_TPSA);      // Veber: polar surface area
    return ok ? 1 : 0;
}

// ---------------------------------------------------------------------------
// decode_product_index: turn a flat product index p in [0, N) into the per-slot
// synthon indices (idx[0..N_SLOTS-1]) via a MIXED-RADIX decomposition.
//   This is exactly how an odometer works: slot 0 is the fastest-spinning digit.
//   Given slot sizes s[k], the product index is
//       p = idx[0] + s[0]*(idx[1] + s[1]*(idx[2] + ...))
//   so we peel off one digit at a time with /, %. We do it on BOTH sides so the
//   CPU and GPU visit products in the SAME order -> the "first K passing
//   products" report is identical (determinism).
//   sizes : [N_SLOTS] slot sizes (number of synthons available in each slot).
//   p     : flat product index, 0 <= p < product(sizes).
//   idx   : [N_SLOTS] output, the chosen synthon index within each slot.
// ---------------------------------------------------------------------------
HD inline void decode_product_index(int64_t p, const int sizes[N_SLOTS], int idx[N_SLOTS]) {
    for (int k = 0; k < N_SLOTS; ++k) {
        const int s = sizes[k];            // size of slot k (the radix)
        idx[k] = static_cast<int>(p % s);  // this slot's digit
        p /= s;                            // shift to the next, slower digit
    }
}

#undef HD   // keep the macro local to this header (callers redefine if needed)
