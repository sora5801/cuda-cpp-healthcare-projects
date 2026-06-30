// ===========================================================================
// src/reference_cpu.cpp  --  The trusted serial baseline + backbone loader
// ---------------------------------------------------------------------------
// Project 2.10 : Protein Design / Inverse Folding Inference
//
// ROLE
//   (1) AA_CODES, plus a helper to map a one-letter code back to its index.
//   (2) load_backbone(): parse the tiny text dataset (data/README.md format).
//   (3) design_cpu(): the obviously-correct serial inverse-folding pass the GPU
//       kernel is verified against. No parallelism, no cleverness on purpose --
//       if CPU and GPU agree, we trust the GPU.
//   (4) recovery_percent() / sequence_string(): deterministic reporting helpers.
//
//   Compiled by the host C++ compiler only (no CUDA). Both this file and
//   kernels.cu include inverse_folding.h, so the per-residue SCORE is computed
//   by the identical shared function on both sides -> exact agreement.
//
// READ THIS AFTER: reference_cpu.h, inverse_folding.h. Compare with kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

// Canonical one-letter codes; index k <-> AA_CODES[k]. The order MUST match the
// AA_HYDROPHOBICITY table in inverse_folding.h exactly (index k means the same
// residue in both), namely the standard ProteinMPNN/3-letter-alphabetical order:
//   Ala Arg Asn Asp Cys Gln Glu Gly His Ile Leu Lys Met Phe Pro Ser Thr Trp Tyr Val
// The string literal carries the 20 letters plus the implicit terminating '\0',
// so its length is NUM_AA+1 -- this is the ONE authoritative source of truth.
const char AA_CODES[NUM_AA + 1] = "ARNDCQEGHILKMFPSTWYV";

// aa_index_of: map a one-letter amino-acid code to its 0..19 index, or -1 if the
//   letter is not a standard amino acid. Linear scan over 20 entries is trivial.
//   Used by the loader to parse the native sequence column.
static int aa_index_of(char code) {
    for (int k = 0; k < NUM_AA; ++k)
        if (AA_CODES[k] == code) return k;
    return -1;   // unknown letter -> caller treats as a parse error
}

Backbone load_backbone(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open backbone file: " + path);

    int L = 0;
    if (!(in >> L))
        throw std::runtime_error("bad header (expected residue count <L>) in " + path);
    if (L <= 0) throw std::runtime_error("non-positive residue count in " + path);

    Backbone bb;
    bb.res.reserve(L);
    bb.native.reserve(L);

    // Each residue line is "x y z native_letter": three floats then one letter.
    for (int i = 0; i < L; ++i) {
        BackboneResidue r{};
        std::string aa;
        if (!(in >> r.x >> r.y >> r.z >> aa))
            throw std::runtime_error("unexpected end of data at residue " +
                                     std::to_string(i) + " in " + path);
        if (aa.size() != 1)
            throw std::runtime_error("native residue must be a single letter at residue " +
                                     std::to_string(i) + " in " + path);
        const int idx = aa_index_of(aa[0]);
        if (idx < 0)
            throw std::runtime_error(std::string("unknown amino-acid letter '") + aa[0] +
                                     "' at residue " + std::to_string(i) + " in " + path);
        bb.res.push_back(r);
        bb.native.push_back(idx);
    }
    return bb;
}

void design_cpu(const Backbone& bb, DesignResult& out) {
    const int L = bb.size();
    out.neighbors.assign(L, 0);
    out.designed.assign(L, 0);
    out.score.assign(L, 0);

    // ---- Step 1: burial = neighbor count (all-pairs O(L^2)) ---------------
    // For each residue i, count residues j != i whose Calpha lies within
    // CONTACT_RADIUS. We compare SQUARED distances to avoid a sqrt per pair.
    // This is the serial twin of the neighbor kernel in kernels.cu, and the
    // analog of message-passing over the protein contact graph in a real GNN.
    for (int i = 0; i < L; ++i) {
        const BackboneResidue& ri = bb.res[i];
        int count = 0;
        for (int j = 0; j < L; ++j) {
            if (j == i) continue;                 // a residue is not its own neighbor
            const BackboneResidue& rj = bb.res[j];
            const float dx = ri.x - rj.x;
            const float dy = ri.y - rj.y;
            const float dz = ri.z - rj.z;
            const float d2 = dx * dx + dy * dy + dz * dz;   // squared distance (A^2)
            if (d2 <= CONTACT_RADIUS_SQ) ++count;           // within contact -> neighbor
        }
        out.neighbors[i] = count;
    }

    // ---- Step 2: per-residue argmax over the 20 amino acids ---------------
    // Each position is scored independently (the per-residue logits), so this is
    // embarrassingly parallel -> one GPU thread per residue in kernels.cu.
    for (int i = 0; i < L; ++i) {
        int best_aa = 0;
        // Initialize "best" with amino acid 0 so the tie-break (lowest index)
        // is well-defined and matches the GPU exactly.
        int best_score = score_aa_at_residue(0, out.neighbors[i]);
        for (int aa = 1; aa < NUM_AA; ++aa) {
            const int s = score_aa_at_residue(aa, out.neighbors[i]);
            // STRICT '>' keeps the FIRST (lowest-index) amino acid on a tie, so
            // the choice is deterministic and identical on CPU and GPU.
            if (s > best_score) { best_score = s; best_aa = aa; }
        }
        out.designed[i] = best_aa;
        out.score[i]    = best_score;
    }
}

int recovery_percent(const Backbone& bb, const DesignResult& d) {
    const int L = bb.size();
    if (L == 0) return 0;
    int matches = 0;
    for (int i = 0; i < L; ++i)
        if (d.designed[i] == bb.native[i]) ++matches;   // designed == native?
    // Integer percentage, rounded to nearest: (matches*100 + L/2) / L. Pure
    // integer math so the printed number is reproducible on any platform.
    return (matches * 100 + L / 2) / L;
}

std::string sequence_string(const std::vector<int>& aa_indices) {
    std::string s;
    s.reserve(aa_indices.size());
    for (int idx : aa_indices) s.push_back(AA_CODES[idx]);  // index -> letter
    return s;
}
