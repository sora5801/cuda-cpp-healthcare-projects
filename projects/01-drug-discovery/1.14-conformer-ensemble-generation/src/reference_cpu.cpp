// ===========================================================================
// src/reference_cpu.cpp  --  Trusted serial baseline: energies + RMSD clustering
// ---------------------------------------------------------------------------
// Project 1.14 : Conformer Ensemble Generation
//
// ROLE
//   (1) enumerate_energies_cpu(): loop over all conformers and call the shared
//       conformer_energy() once each. Obviously correct on purpose -- if this and
//       the GPU kernel (which calls the SAME conformer_energy()) agree, we trust
//       the GPU.
//   (2) rmsd_cluster(): the greedy RMSD pruning that turns hundreds of raw
//       conformers into a small non-redundant ensemble. This is a serial,
//       order-dependent scan, which is why it stays on the CPU (see THEORY).
//
//   Compiled by the host C++ compiler only (no CUDA). The physics it calls lives
//   in conformer.h, where the host compiler sees CONF_HD expand to nothing.
//
// READ THIS AFTER: reference_cpu.h.  Compare against kernels.cu (the GPU twin of
// step (1)).
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>   // std::sort, std::stable_sort
#include <numeric>     // std::iota

// ---------------------------------------------------------------------------
// enumerate_energies_cpu: the serial energy sweep.
//   Complexity: O(N_CONFORMER * N_ATOMS^2) -- the inner cost is the pairwise clash
//   sum. Each conformer is INDEPENDENT (its energy depends only on its own index),
//   which is exactly why this maps perfectly onto one GPU thread per conformer in
//   kernels.cu.
// ---------------------------------------------------------------------------
void enumerate_energies_cpu(std::vector<double>& energy) {
    energy.assign(static_cast<std::size_t>(N_CONFORMER), 0.0);
    for (long c = 0; c < N_CONFORMER; ++c) {
        // conformer_energy() with a null position pointer: we only need the scalar
        // here; clustering rebuilds coordinates on demand below.
        energy[static_cast<std::size_t>(c)] = conformer_energy(c, nullptr);
    }
}

// ---------------------------------------------------------------------------
// rmsd_cluster: greedy "leader" clustering by ascending energy (see header).
//   We deliberately rebuild each candidate's 3D coordinates from its index via
//   conformer_energy(idx, pos) rather than storing all N_CONFORMER coordinate sets
//   -- N_CONFORMER is small here, and recomputation keeps memory flat and the code
//   honest about where coordinates come from. (At library scale you would cache
//   them; that trade-off is discussed in THEORY.)
// ---------------------------------------------------------------------------
std::vector<long> rmsd_cluster(const std::vector<double>& energy,
                               double rmsd_threshold) {
    const long n = static_cast<long>(energy.size());

    // (1) Order conformer indices by ascending energy. Ties are broken by the
    //     lower index via stable_sort over an already-ascending index list, so
    //     the clustering -- and therefore the printed output -- is DETERMINISTIC.
    std::vector<long> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0L);                 // 0,1,2,...,n-1
    std::stable_sort(order.begin(), order.end(),
        [&](long a, long b) {
            return energy[static_cast<std::size_t>(a)]
                 < energy[static_cast<std::size_t>(b)];        // lower energy first
        });

    // (2) Greedy acceptance. `reps` holds the accepted representative indices;
    //     `rep_pos` caches their coordinates so each candidate is compared without
    //     recomputing a representative's geometry every time.
    std::vector<long> reps;
    std::vector<std::array<Vec3, N_ATOMS>> rep_pos;

    for (long idx : order) {
        // Rebuild this candidate's 3D coordinates from its index.
        std::array<Vec3, N_ATOMS> cand;
        conformer_energy(idx, cand.data());

        // Accept unless it is within rmsd_threshold of an existing representative.
        bool is_duplicate = false;
        for (std::size_t k = 0; k < reps.size(); ++k) {
            if (coord_rmsd(cand.data(), rep_pos[k].data()) < rmsd_threshold) {
                is_duplicate = true;   // a lower-energy near-identical shape exists
                break;
            }
        }
        if (!is_duplicate) {
            reps.push_back(idx);
            rep_pos.push_back(cand);
        }
    }
    return reps;   // representative indices, ascending energy
}
