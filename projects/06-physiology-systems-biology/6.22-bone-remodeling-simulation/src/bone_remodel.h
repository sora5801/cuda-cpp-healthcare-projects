// ===========================================================================
// src/bone_remodel.h  --  Shared (host + device) bone-remodeling physics
// ---------------------------------------------------------------------------
// Project 6.22 : Bone Remodeling Simulation   (REDUCED-SCOPE teaching version)
//
// THE MODEL  (see ../THEORY.md for the full science -> math -> algorithm story)
//   Living bone is not static: it constantly rebuilds itself in response to the
//   mechanical loads it carries. Wolff's law (1892), formalized by Frost's
//   "mechanostat" and Huiskes' strain-energy-density (SED) remodeling rule,
//   says each patch of bone senses a mechanical stimulus and:
//       * DEPOSITS  new bone (osteoblasts) where it is OVER-loaded,
//       * RESORBS   bone      (osteoclasts) where it is UNDER-loaded,
//       * stays put inside a "lazy zone" (dead band) around a homeostatic
//         setpoint -- otherwise the tissue would oscillate forever.
//   Iterated over (simulated) months, this feedback carves porous marrow space
//   into oriented trabecular struts that line up with the dominant load path --
//   the beautiful lattice you see inside a vertebra or femoral head.
//
//   THE HONEST SIMPLIFICATION (CLAUDE.md section 13):
//   The research-grade pipeline computes the SED field by solving a large
//   voxel finite-element system K u = f every remodeling step (cuSPARSE
//   assembly + a cuSOLVER/PCG solve over 10^8 voxels). That is a whole solver
//   project on its own. Here we keep the SAME remodeling *biology* but replace
//   the FEM solve with a physically-motivated, cheap PROXY for how mechanical
//   stimulus spreads through the tissue:
//
//       Load sharing as diffusion. A denser (stiffer) voxel carries more load,
//       and load spreads to its neighbours. We model the steady mechanical
//       stimulus S as the solution of a density-weighted smoothing (a discrete
//       diffusion / Jacobi relaxation) of an externally applied load field.
//       This is a 5-point STENCIL -- exactly the GPU pattern (see PATTERNS.md
//       section 1: "grid PDE / nearest-neighbour update -> stencil + ping-pong",
//       exemplified by flagships 6.04 and 14.02). It is NOT a real FEM solve,
//       and THEORY.md section "real world" says precisely how production differs.
//
//   Two per-element formulas are the entire physics, and they live HERE as
//   __host__ __device__ inline functions so the CPU reference (reference_cpu.cpp,
//   host compiler) and the GPU kernels (kernels.cu, nvcc) execute BYTE-FOR-BYTE
//   identical math -- the trick that makes GPU-vs-CPU verification exact rather
//   than approximate (PATTERNS.md section 2). BR_HD expands to __host__ __device__
//   under nvcc and to nothing under the plain host compiler.
//
//   Grid layout: a 2-D voxel grid, nx columns by ny rows, row-major, so the
//   flat index of voxel (x,y) is  y*nx + x. y grows DOWNWARD (row 0 = top,
//   where we push the load in; row ny-1 = the supported base).
// ===========================================================================
#pragma once

#include <cstddef>   // std::size_t

// --- Host/device decorator shim (PATTERNS.md section 2) --------------------
#ifdef __CUDACC__
#define BR_HD __host__ __device__
#else
#define BR_HD
#endif

// ---------------------------------------------------------------------------
// BoneParams : one remodeling "job". All fields are read-only during a run.
//   Units are dimensionless teaching units (documented), NOT SI -- this is a
//   qualitative model of the remodeling FEEDBACK, not a calibrated simulation.
// ---------------------------------------------------------------------------
struct BoneParams {
    int    nx = 0;          // grid columns  (x, across the load)
    int    ny = 0;          // grid rows     (y, along the load; row 0 = loaded top)
    int    remodel_steps=0; // number of remodeling iterations (each ~ "one month")
    int    relax_iters = 0; // Jacobi sweeps per step to (partially) settle the SED field
    double load = 0.0;      // applied mechanical stimulus injected at the loaded footprint (>0)
    int    load_x0 = 0;     // first column of the loaded footprint on the top edge (inclusive)
    int    load_x1 = 0;     // last  column of the loaded footprint on the top edge (inclusive)
    double setpoint = 0.0;  // homeostatic SED target k (the mechanostat's "comfortable" level)
    double lazy = 0.0;      // half-width w of the lazy zone (dead band) around k
    double rate = 0.0;      // remodeling gain: how fast density chases the stimulus
    double rho_min = 0.0;   // minimum density floor (bone never fully vanishes here)
    double rho_init = 0.0;  // initial uniform density everywhere (the "unremodeled" blank)
};

// Flat row-major index of voxel (x,y). Kept in one place so CPU and GPU agree.
BR_HD inline std::size_t bone_idx(int x, int y, int nx) {
    return static_cast<std::size_t>(y) * nx + x;
}

// ---------------------------------------------------------------------------
// bone_relax_point : ONE Jacobi relaxation update of the mechanical-stimulus
//   field S at voxel (x,y). This is the STENCIL that stands in for the FEM solve.
//
//   Intent: stimulus flows from the loaded top edge down toward the support,
//   preferentially through STIFF (dense) material -- stiffer neighbours conduct
//   more load, softer ones less. We model that as a density-weighted average of
//   the four von-Neumann neighbours plus the voxel's own injected load:
//
//       S_new(x,y) = [ load_src(x,y)
//                      + sum_over_neighbours( w_n * S_old(neighbour) ) ]
//                    / ( 1 + sum_over_neighbours( w_n ) )
//
//   where the conductance w_n between (x,y) and a neighbour is the AVERAGE of
//   their densities (so a void voxel, rho~rho_min, barely conducts). Boundary
//   handling: the top row has the external load added as a source; all edges use
//   a zero-flux (Neumann) rule by simply skipping off-grid neighbours. This is a
//   damped, always-stable averaging update (a weighted mean is a convex
//   combination -> it can never blow up), which is why a handful of sweeps per
//   remodeling step is enough for a teaching demo.
//
//   Reads S_old + rho (both read-only this sweep); writes one S_new value.
//   The CPU loops this over all voxels; the GPU runs one thread per voxel. No
//   voxel writes another's S_new, so there are no races and no atomics.
// ---------------------------------------------------------------------------
BR_HD inline double bone_relax_point(int x, int y, int nx, int ny,
                                     double load, int load_x0, int load_x1,
                                     const double* S_old,
                                     const double* rho) {
    const double rho_c = rho[bone_idx(x, y, nx)];   // this voxel's density (stiffness proxy)

    // Accumulate the numerator (sources + weighted neighbour stimulus) and the
    // denominator (1 + total conductance). Starting the denominator at 1.0 is a
    // small self-damping term that keeps the field bounded even in a void.
    double num = 0.0;
    double den = 1.0;

    // External load is injected as a source term only under the loaded FOOTPRINT
    // on the TOP row (y == 0 and load_x0 <= x <= load_x1). Think of a joint or
    // implant pressing on part of the top surface -- a LOCALIZED load, not a
    // uniform pressure. That locality is what makes remodeling carve an oriented
    // strut instead of thickening the whole slab uniformly.
    if (y == 0 && x >= load_x0 && x <= load_x1) num += load;

    // --- Four von-Neumann neighbours: (x-1,y),(x+1,y),(x,y-1),(x,y+1) --------
    // For each in-grid neighbour, conductance = mean density of the pair.
    // dx,dy pairs written out explicitly (no loop) so the math is transparent.
    if (x > 0) {                                    // left neighbour
        const double w = 0.5 * (rho_c + rho[bone_idx(x - 1, y, nx)]);
        num += w * S_old[bone_idx(x - 1, y, nx)];
        den += w;
    }
    if (x < nx - 1) {                               // right neighbour
        const double w = 0.5 * (rho_c + rho[bone_idx(x + 1, y, nx)]);
        num += w * S_old[bone_idx(x + 1, y, nx)];
        den += w;
    }
    if (y > 0) {                                    // upper neighbour (toward load)
        const double w = 0.5 * (rho_c + rho[bone_idx(x, y - 1, nx)]);
        num += w * S_old[bone_idx(x, y - 1, nx)];
        den += w;
    }
    if (y < ny - 1) {                               // lower neighbour (toward support)
        const double w = 0.5 * (rho_c + rho[bone_idx(x, y + 1, nx)]);
        num += w * S_old[bone_idx(x, y + 1, nx)];
        den += w;
    }

    return num / den;   // den >= 1.0 always, so no divide-by-zero
}

// ---------------------------------------------------------------------------
// bone_apply_stimulus : ONE remodeling update of the density rho at voxel (x,y)
//   given the settled mechanical-stimulus field S. THIS is the mechanostat --
//   Frost's dead-band / Huiskes' SED rule -- and it is the biological heart of
//   the whole project.
//
//   The local stimulus we compare against the setpoint is the strain-energy-like
//   quantity  phi = S / rho :  for a fixed load, softer (less dense) bone strains
//   MORE, so it feels a larger stimulus per unit mass -- this is what drives thin
//   struts to thicken and unloaded struts to waste away. (Using S/rho rather than
//   raw S is the standard "remodeling signal per unit bone" choice; see THEORY.)
//
//   Piecewise rule with a lazy zone of half-width `lazy` about `setpoint` k:
//       phi > k + lazy   ->  OVER-loaded  -> FORM  : rho += rate*(phi - (k+lazy))
//       phi < k - lazy   ->  UNDER-loaded -> RESORB: rho -= rate*((k-lazy) - phi)
//       otherwise (lazy) ->  homeostasis  -> rho unchanged
//   Then clamp rho to [rho_min, 1]. The clamp both keeps density physical
//   (a voxel is at most fully mineralized) and prevents total disappearance,
//   which would make S/rho singular.
//
//   Deterministic and independent per voxel: reads S,rho for one voxel, returns
//   the new rho. CPU loops it; GPU gives each voxel a thread.
// ---------------------------------------------------------------------------
BR_HD inline double bone_apply_stimulus(int x, int y, int nx,
                                        double setpoint, double lazy,
                                        double rate, double rho_min,
                                        const double* S,
                                        const double* rho) {
    const std::size_t idx = bone_idx(x, y, nx);
    const double r  = rho[idx];
    // Guard the density floor before dividing (r >= rho_min > 0 by construction,
    // but we make the intent explicit for the reader).
    const double phi = S[idx] / r;             // stimulus per unit bone (SED-like)

    const double hi = setpoint + lazy;         // top of the lazy zone
    const double lo = setpoint - lazy;         // bottom of the lazy zone

    double r_new = r;
    if (phi > hi) {
        // Over-loaded: osteoblasts add bone proportional to the overshoot.
        r_new = r + rate * (phi - hi);
    } else if (phi < lo) {
        // Under-loaded: osteoclasts resorb bone proportional to the shortfall.
        r_new = r - rate * (lo - phi);
    }
    // else: inside the lazy zone -> no net remodeling (mechanostat homeostasis).

    // Clamp to the physical range [rho_min, 1]. Written as two branches (not
    // fmin/fmax) so host and device produce identical results with no library
    // dependence.
    if (r_new < rho_min) r_new = rho_min;
    if (r_new > 1.0)     r_new = 1.0;
    return r_new;
}

// ---------------------------------------------------------------------------
// bone_state : classify a voxel's mechanostat state from phi = S/rho, purely for
//   the deterministic report (a 3-bin histogram: resorbing / homeostatic /
//   forming). Shared so CPU and GPU classify identically. Returns 0,1,2.
// ---------------------------------------------------------------------------
BR_HD inline int bone_state(int x, int y, int nx,
                            double setpoint, double lazy,
                            const double* S, const double* rho) {
    const std::size_t idx = bone_idx(x, y, nx);
    const double phi = S[idx] / rho[idx];
    if (phi < setpoint - lazy) return 0;   // RESORB
    if (phi > setpoint + lazy) return 2;   // FORM
    return 1;                              // homeostatic (lazy zone)
}
