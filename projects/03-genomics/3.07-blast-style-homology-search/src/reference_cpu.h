// ===========================================================================
// src/reference_cpu.h  --  Data model, BLOSUM62, loader + CPU reference
// ---------------------------------------------------------------------------
// Project 3.7 : BLAST-Style Homology Search
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler, so the shared DATA
//   MODEL (the encoded sequence database, the BLOSUM62 matrix, the FASTA loader)
//   and the CPU reference prototype live here -- no CUDA syntax. kernels.cuh
//   includes this header too, to reuse the SAME database struct and constants,
//   so the host and device sides can never drift apart. The per-residue scoring
//   math itself lives one level deeper, in blast_core.h (shared __host__
//   __device__), which this header includes.
//
// THE PIPELINE (see ../THEORY.md for the full derivation)
//   1. Load a query protein + a small protein database (FASTA), encode each
//      residue to a 0..23 index.
//   2. Build a k-mer index of the QUERY: a hash from each length-k word in the
//      query to the query position(s) where it occurs. (k=4 here.) This is done
//      ONCE, on the host, and shared by CPU and GPU.
//   3. For each DB sequence, slide a length-k window; whenever a DB k-mer is in
//      the query index, that (qpos,dpos) pair is a SEED. Run gapless X-drop
//      extension (blast_core.h) from each seed and keep the best HSP score.
//   4. Rank DB sequences by best HSP score -> the homology hits.
//
//   The GPU parallelises steps 3-4: one thread per DB sequence. Steps 1-2 are
//   cheap host setup shared by both paths (so CPU and GPU see identical seeds).
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh. blast_core.h is the scoring.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "blast_core.h"   // N_ALPHA, IDX_X, SeqView, encode_residue, gapless_xdrop

// ---------------------------------------------------------------------------
// Algorithm constants. COMPILE-TIME so the kernel can size fixed buffers and
// the loader/index agree with the device. Defaults follow protein BLAST.
// ---------------------------------------------------------------------------
constexpr int SEED_K = 4;    // k-mer (word) length for seeding (BLASTP uses 3; 4 is a
                             // touch more specific -> fewer spurious seeds on a tiny DB).
constexpr int X_DROP = 12;   // X-drop threshold for ungapped extension (BLOSUM units).
                             // Larger => extensions run longer before giving up.

// ---------------------------------------------------------------------------
// blosum62(): return a pointer to the canonical 24x24 BLOSUM62 matrix, flat,
// row-major, in the ALPHA order of blast_core.h. Defined in reference_cpu.cpp.
//   The matrix is the de-facto standard protein substitution score table (log-
//   odds of substitution among homologous proteins at ~62% identity). Integer
//   entries are what make CPU==GPU exact. The GPU copies these same 576 bytes
//   into constant memory (kernels.cu).
// ---------------------------------------------------------------------------
const int8_t* blosum62();

// ---------------------------------------------------------------------------
// SequenceDB: the loaded problem.
//   The query and N database sequences, all encoded to 0..23 residue indices.
//   The DB is stored as ONE concatenated buffer `db_res` plus per-sequence
//   (offset,length) arrays, so the GPU receives a single coalesced device array
//   instead of N little ones. `names` keeps the FASTA headers for reporting.
// ---------------------------------------------------------------------------
struct SequenceDB {
    // The query sequence.
    std::vector<int8_t>      query;      // [query_len] encoded residues
    std::string              query_name; // FASTA header of the query

    // The database, concatenated.
    int                      n = 0;      // number of DB sequences
    std::vector<int8_t>      db_res;     // all DB residues, end to end
    std::vector<int>         db_off;     // [n] start offset of seq i in db_res
    std::vector<int>         db_len;     // [n] length of seq i
    std::vector<std::string> names;      // [n] FASTA header of each DB sequence

    // Convenience: a SeqView onto DB sequence i (used by host and device code).
    SeqView db_view(int i) const {
        return SeqView{ db_res.data() + db_off[i], db_len[i] };
    }
    SeqView query_view() const {
        return SeqView{ query.data(), static_cast<int>(query.size()) };
    }
};

// ---------------------------------------------------------------------------
// QueryIndex: the query's k-mer hash, built once and shared by CPU and GPU.
//   `table` maps a packed k-mer code -> list of query positions holding it.
//   We pack a length-k word of 0..23 indices into a single int by base-24
//   digits (code = ((r0*24 + r1)*24 + r2)*24 + r3 for k=4); this is a perfect,
//   collision-free encoding because each residue < 24.
//
//   For the GPU we ALSO export a flat, device-friendly form (see kernels.cu):
//   a sorted array of (code, qpos) pairs the kernel binary-searches. The host
//   reference uses the hash map directly. Both enumerate the SAME seeds.
// ---------------------------------------------------------------------------
struct QueryIndex {
    std::unordered_map<int, std::vector<int>> table;  // kmer code -> query positions
};

// pack_kmer: fold k encoded residues at seq[pos..pos+k) into one base-24 int.
//   Returns -1 if any residue is the 'X' fallback or the window runs past the
//   end (BLAST does not seed on ambiguous residues; this also keeps the code
//   space tight). Marked HD (__host__ __device__, see blast_core.h) so the SAME
//   packing runs on the host loader/index AND inside the GPU kernel -- both
//   sides therefore produce identical k-mer codes (and hence identical seeds).
HD inline int pack_kmer(const int8_t* seq, int len, int pos, int k) {
    if (pos + k > len) return -1;
    int code = 0;
    for (int t = 0; t < k; ++t) {
        int r = seq[pos + t];
        if (r == IDX_X) return -1;          // skip ambiguous 'X' residues
        code = code * N_ALPHA + r;          // base-24 digit shift
    }
    return code;
}

// build_query_index: scan the query once and record every (k-mer -> position).
QueryIndex build_query_index(const std::vector<int8_t>& query, int k);

// load_fasta: parse a FASTA file into a SequenceDB. The FIRST record is the
//   query; every subsequent record is a database sequence. Residues are encoded
//   on load. Throws std::runtime_error on a missing/empty/malformed file.
SequenceDB load_fasta(const std::string& path);

// ---------------------------------------------------------------------------
// blast_cpu: THE trusted reference. For each DB sequence i, enumerate its seeds
// against the query index and run gapless X-drop extension (blast_core.h),
// returning the best HSP score per sequence in `best_score[i]`. Deliberately
// simple and obviously correct -- the GPU kernel is verified against this.
//   query_idx : the prebuilt query k-mer index (seeds come from here).
//   best_score: resized to db.n; best HSP score for each DB sequence.
// ---------------------------------------------------------------------------
void blast_cpu(const SequenceDB& db, const QueryIndex& query_idx,
               std::vector<int>& best_score);
