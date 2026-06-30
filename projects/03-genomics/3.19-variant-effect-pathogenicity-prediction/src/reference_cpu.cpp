// ===========================================================================
// src/reference_cpu.cpp  --  Data loader, fixed-model init, CPU reference
// ---------------------------------------------------------------------------
// Project 3.19 : Variant Effect / Pathogenicity Prediction
//
// ROLE IN THE PROJECT
//   (1) load_variants()      : parse the tiny text dataset (data/README.md fmt).
//   (2) init_model()         : fill the FIXED, SYNTHETIC CNN weights (the same
//                              bytes the GPU uploads to constant memory).
//   (3) score_variants_cpu() : the trusted serial baseline -- loop the shared
//                              vep_variant_effect() over every variant. No
//                              parallelism, no cleverness: if CPU and GPU agree
//                              we trust the GPU.
//
//   Compiled by the host C++ compiler only (no CUDA). The per-variant math is in
//   vep_model.h (the __host__ __device__ core) and is shared verbatim with the
//   GPU kernel, so the two results match to ~1e-12 (THEORY "How we verify").
//
// READ THIS AFTER: reference_cpu.h, vep_model.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::sin -- used to synthesise deterministic weights
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// load_variants: read the dataset described in data/README.md.
//
// File format (whitespace separated, '#'-prefixed lines and blank lines skipped):
//   line 1 : "<n> <window_width>"          ; width MUST equal VEP_WINDOW
//   next n : "<pos> <REF> <ALT> <WINDOW>"   ; WINDOW = VEP_WINDOW A/C/G/T letters
//                                             the REFERENCE context; its centre
//                                             base must equal <REF>.
// We build the ALT window by copying the REF window and overwriting the centre
// base with <ALT> -- i.e. in-silico mutagenesis of exactly one position. This
// guarantees ref and alt differ at exactly the variant locus and nowhere else,
// which is the whole premise of a delta score.
// ---------------------------------------------------------------------------
VariantSet load_variants(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open variant file: " + path);

    // Read the file line by line so we can skip comments/blanks cleanly, then
    // tokenise each meaningful line with a stringstream.
    auto next_line = [&](std::string& line) -> bool {
        while (std::getline(in, line)) {
            // Trim a leading run of whitespace to detect '#' comments / blanks.
            std::size_t s = line.find_first_not_of(" \t\r\n");
            if (s == std::string::npos) continue;          // blank line
            if (line[s] == '#') continue;                  // comment line
            return true;
        }
        return false;
    };

    std::string line;
    if (!next_line(line)) throw std::runtime_error("empty variant file: " + path);

    int n = 0, width = 0;
    {
        std::istringstream hs(line);
        if (!(hs >> n >> width))
            throw std::runtime_error("bad header (expected '<n> <window_width>') in " + path);
    }
    if (width != VEP_WINDOW)
        throw std::runtime_error("window width mismatch: file has " + std::to_string(width) +
                                 " but this build expects VEP_WINDOW=" + std::to_string(VEP_WINDOW));
    if (n <= 0) throw std::runtime_error("non-positive variant count in " + path);

    VariantSet vs;
    vs.n = n;
    vs.ref.resize(static_cast<std::size_t>(n) * VEP_WINDOW);
    vs.alt.resize(static_cast<std::size_t>(n) * VEP_WINDOW);
    vs.ref_base.resize(n);
    vs.alt_base.resize(n);
    vs.pos.resize(n);

    for (int i = 0; i < n; ++i) {
        if (!next_line(line))
            throw std::runtime_error("unexpected end of data at variant " + std::to_string(i) +
                                     " in " + path);
        std::istringstream ls(line);
        int pos = 0;
        std::string ref_s, alt_s, win_s;
        if (!(ls >> pos >> ref_s >> alt_s >> win_s))
            throw std::runtime_error("malformed variant line " + std::to_string(i + 1) +
                                     " (expected '<pos> <REF> <ALT> <WINDOW>') in " + path);
        if (static_cast<int>(win_s.size()) != VEP_WINDOW)
            throw std::runtime_error("variant " + std::to_string(i) + " window has " +
                                     std::to_string(win_s.size()) + " bases, expected " +
                                     std::to_string(VEP_WINDOW));

        const int8_t rb = base_code(ref_s.empty() ? '?' : ref_s[0]);
        const int8_t ab = base_code(alt_s.empty() ? '?' : alt_s[0]);
        if (rb < 0 || ab < 0)
            throw std::runtime_error("variant " + std::to_string(i) +
                                     " has a non-ACGT ref/alt allele");

        // Decode the reference window letter by letter.
        int8_t* rdst = &vs.ref[static_cast<std::size_t>(i) * VEP_WINDOW];
        int8_t* adst = &vs.alt[static_cast<std::size_t>(i) * VEP_WINDOW];
        for (int p = 0; p < VEP_WINDOW; ++p) {
            const int8_t c = base_code(win_s[p]);
            if (c < 0)
                throw std::runtime_error("variant " + std::to_string(i) +
                                         " window has a non-ACGT base at column " +
                                         std::to_string(p));
            rdst[p] = c;
            adst[p] = c;                       // alt starts as a copy of ref
        }
        // Sanity: the centre of the reference window MUST be the stated ref base.
        if (rdst[VEP_CENTER] != rb)
            throw std::runtime_error("variant " + std::to_string(i) +
                                     ": window centre base disagrees with REF allele");
        // In-silico mutagenesis: flip ONLY the centre base in the alt window.
        adst[VEP_CENTER] = ab;

        vs.ref_base[i] = rb;
        vs.alt_base[i] = ab;
        vs.pos[i]      = pos;
    }
    return vs;
}

// ---------------------------------------------------------------------------
// init_model: synthesise the FIXED model weights deterministically.
//
//   We do NOT random-initialise at runtime (that would make the demo's output
//   change run to run and break the byte-identical-stdout rule). Instead every
//   weight is a closed-form function of its indices via sin() -- a cheap, fully
//   reproducible "pseudo-random" generator that produces the SAME weights on
//   every machine and every run. Then we hand-engineer two effects on top so the
//   network is not just noise:
//
//     * Filter 0 is tuned to fire on the 5-mer  C A G C T  (codes 1 0 2 1 3):
//       a planted "deleterious motif". Its dense weight is strongly POSITIVE, so
//       creating this motif raises the pathogenic score (positive delta).
//     * Filter 1 is tuned to fire on  T A T A T  (codes 3 0 3 0 3): a planted
//       "benign/protective motif" with a NEGATIVE dense weight, so creating it
//       lowers the score (negative delta).
//
//   These plants make the ranking interpretable: a variant that builds the
//   deleterious motif at the centre should top the "most pathogenic" list, which
//   is exactly the kind of sanity check PATTERNS.md sec 6 recommends. The values
//   are arbitrary teaching constants with NO biological meaning (CLAUDE.md sec 8).
// ---------------------------------------------------------------------------
void init_model(VepModel& m) {
    // (a) Base "texture": a deterministic small weight for every conv entry.
    //     sin() of a mixed index gives values in [-1,1]; scaling by 0.30 keeps
    //     pre-activations modest so the sigmoid stays in its informative range.
    for (int k = 0; k < VEP_KERNELS; ++k) {
        for (int c = 0; c < VEP_BASES; ++c) {
            for (int w = 0; w < VEP_KWIDTH; ++w) {
                const double idx = 1.0 + k * 31 + c * 7 + w * 3;     // distinct per entry
                m.conv_w[k][c][w] = 0.30 * std::sin(idx * 1.2345);
            }
        }
        m.conv_b[k] = -0.50;                  // negative bias: filters fire only
                                              // when a real match overcomes it.
        // Dense weights: also deterministic, modest magnitude. Overwritten below
        // for the two "designed" filters.
        m.dense_w[k] = 0.40 * std::sin(2.0 + k * 5.0);
    }
    m.dense_b = -0.20;                        // small global offset on the logit

    // (b) Plant the DELETERIOUS motif into filter 0 (5-mer C A G C T).
    //     For each position w we set a strong +1.0 weight on the channel that
    //     matches the motif base and a -0.5 weight on the other three, so the
    //     filter scores high only when the exact 5-mer is present.
    const int del_motif[VEP_KWIDTH] = {1, 0, 2, 1, 3};   // C A G C T
    for (int w = 0; w < VEP_KWIDTH; ++w) {
        for (int c = 0; c < VEP_BASES; ++c)
            m.conv_w[0][c][w] = (c == del_motif[w]) ? 1.0 : -0.5;
    }
    m.conv_b[0]  = -3.0;     // high threshold: needs a near-perfect 5-mer match
    m.dense_w[0] = 2.5;      // strong POSITIVE -> creating the motif is "pathogenic"

    // (c) Plant the PROTECTIVE motif into filter 1 (5-mer T A T A T).
    const int ben_motif[VEP_KWIDTH] = {3, 0, 3, 0, 3};   // T A T A T
    for (int w = 0; w < VEP_KWIDTH; ++w) {
        for (int c = 0; c < VEP_BASES; ++c)
            m.conv_w[1][c][w] = (c == ben_motif[w]) ? 1.0 : -0.5;
    }
    m.conv_b[1]  = -3.0;
    m.dense_w[1] = -2.5;     // strong NEGATIVE -> creating the motif is "benign"
}

// ---------------------------------------------------------------------------
// score_variants_cpu: the trusted serial reference. One readable loop over all
// variants, each calling the SAME vep_variant_effect() the GPU kernel uses.
//   Complexity: O(n * 2 * K * (L-W+1) * W) FMAs -- linear in the variant count.
//   This is the baseline whose wall time (timed in main.cu) we compare with the
//   GPU kernel, and the values the GPU result is verified against.
// ---------------------------------------------------------------------------
void score_variants_cpu(const VepModel& m, const VariantSet& vs,
                        std::vector<double>& effect) {
    effect.assign(static_cast<std::size_t>(vs.n), 0.0);
    for (int i = 0; i < vs.n; ++i) {
        const int8_t* ref_win = &vs.ref[static_cast<std::size_t>(i) * VEP_WINDOW];
        const int8_t* alt_win = &vs.alt[static_cast<std::size_t>(i) * VEP_WINDOW];
        // The delta score: score(alt) - score(ref). Pure, side-effect-free, and
        // identical to the device path -> exact agreement up to FP rounding.
        effect[static_cast<std::size_t>(i)] = vep_variant_effect(m, ref_win, alt_win);
    }
}
