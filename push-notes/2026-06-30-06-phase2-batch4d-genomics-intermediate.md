# Push 2026-06-30 #06 -- phase2 batch4d genomics intermediate

> Push-note (CLAUDE.md section 7.1). Fourth domain-3 batch: 6 more Intermediate genomics
> projects, each worker-built and independently lead-verified.

## 1. Summary

Six more **domain-3 (genomics) Intermediate** projects are complete, taking the collection to
**100 -> 106 / 301 (35.2%)** and domain 3 to **25/30** — only 5 projects from finishing the
domain. This batch is the "ML + long-read + alignment-DP variants" cluster: a protein
language model, variant-effect prediction, HiFi assembly overlap, structural-variant calling,
splice-aware alignment, and methylation calling. Notably it shows **four different flavors of
Smith-Waterman**: banded (3.21 SV), spliced/intron-aware (3.23), event-alignment (3.24), and
seed-and-chain overlap (3.20) — a reminder that one DP recurrence underlies much of genomics.
Each was built in its own folder by one worker and re-verified by the lead.

## 2. What changed

Six new fully-implemented projects under `projects/03-genomics/`:

- [`3.18` Protein Language Model Inference](../projects/03-genomics/3.18-protein-language-model-inference)
- [`3.19` Variant Effect / Pathogenicity Prediction](../projects/03-genomics/3.19-variant-effect-pathogenicity-prediction)
- [`3.20` Long-Read HiFi Assembly Overlap & Polishing](../projects/03-genomics/3.20-long-read-hifi-assembly-overlap-polishing)
- [`3.21` Structural Variant (SV) Calling](../projects/03-genomics/3.21-structural-variant-sv-calling)
- [`3.23` Splice-Aware RNA Alignment](../projects/03-genomics/3.23-splice-aware-rna-alignment)
- [`3.24` Methylation / Modified-Base Calling](../projects/03-genomics/3.24-methylation-modified-base-calling)

`docs/STATUS.md` -> these 6 marked **done** (106/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **3.18 Protein Language Model** — a single-block **multi-head self-attention** forward pass
  (the ESM-2 core): embeddings -> Q/K/V -> softmax(QK^T/sqrt(d))*V per head -> output
  projection, one block per (head, query-row) with a shared-memory parallel-reduction softmax
  (shared `attention_math.h`, CPU==GPU ~1.5e-8). The third attention kernel in the collection
  (cf. 2.01, 2.14), now for protein sequences.
- **3.19 Variant Effect Prediction** — batched **in-silico mutagenesis**: a fixed-weight 1-D
  CNN (conv/ReLU/global-max-pool/dense/sigmoid) scores one-hot DNA windows, one thread per
  variant (weights in constant memory), delta = score(ALT) - score(REF). CPU==GPU ~2.8e-17.
  The CADD/Enformer-style scoring shape.
- **3.20 HiFi Assembly Overlap** — minimiser **seed-and-chain** all-vs-all overlap, one thread
  per read pair, shared core (canonical k-mer hashing + integer both-strand chaining DP) ->
  exact CPU==GPU (66/66 pairs); recovers the 11 true neighbour overlaps. The hifiasm/minimap2
  overlap stage.
- **3.21 Structural Variant Calling** — one thread per split read refines its breakpoint via
  **banded Smith-Waterman** (shared `sv.h`), then votes into an integer atomic histogram that
  is clustered into calls. GPU histogram + calls match CPU exactly; the planted deletion is
  recovered.
- **3.23 Splice-Aware RNA Alignment** — a **spliced Smith-Waterman** with a canonical-GT-AG-
  scored intron (`N`) move, one block per read, shared recurrence -> exact CPU==GPU (scores,
  endpoints, all DP cells). The STAR/HISAT2 idea as one DP kernel.
- **3.24 Methylation Calling** — an f5c-style **banded event-alignment** DP (Viterbi
  match/stay/skip in log space) + canonical-vs-5mC pore-model **log-likelihood-ratio** scoring,
  one thread per (read, site), both pore models in constant memory. Shared `meth_core.h` ->
  exact CPU==GPU; recovers 12/12 planted sites.

All six are clearly-labeled **reduced-scope teaching versions** (synthetic data, labeled
synthetic), with production tools (ESM-2/ProtTrans, CADD/AlphaMissense, hifiasm, Manta/Sniffles,
STAR/HISAT2, nanopolish/f5c) named in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/03-genomics/3.24-methylation-modified-base-calling   # (or any of the six)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

No new CUDA libraries this batch — all custom kernels (attention softmax, 1-D CNN, banded /
spliced / event-alignment DP, atomic histograms).

## 5. What to study here

Reading path: **3.19** (1-D CNN scoring) -> **3.18** (multi-head attention) -> the four
Smith-Waterman variants: **3.20** (seed-and-chain) -> **3.21** (banded) -> **3.23** (spliced)
-> **3.24** (event-alignment in log space). Read them after the flagship 3.01 to see how one
recurrence specializes. Exercise: in **3.23**, change the intron penalty and watch the spliced
alignment prefer/avoid introns; in **3.18**, add a second attention block and confirm
determinism holds.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 6 project folders changed; no shared/root file; no artifacts.
- ✅ **Clean rebuild** (`/t:Rebuild`, fat arch list) of all 6 in both `Release|x64` and
  `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (12/12 builds).
- ✅ All 6 **demos PASS**: GPU==CPU (HiFi/SV/splice exact; protein-LM 1.5e-8; VEP 2.8e-17;
  methylation 1e-3).
- ✅ `verify_project.py` -> **DONE** for all 6 (comment ratios **0.80–1.13**).
- **Workflow:** 6 agents, ~1.10M agent tokens, 468 tool uses (relaunched after a window reset;
  first attempt was killed mid-run by the usage limit).
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All six are **reduced-scope teaching versions**: a single attention block (not a trained
  ESM-2), fixed CNN weights, an overlap stage (no consensus polishing), deletion-only SV, a
  small pore model. Labeled synthetic; production scale described in each THEORY.md.

## 8. Next push preview

The **last 5 domain-3 Intermediates** (`3.26` BAM sort/dedup, `3.27` suffix array/BWT/FM-index,
`3.28` profile HMM Viterbi/Forward, `3.29` motif finding, `3.30` pangenome construction) —
completing **domain 3 (30/30)** — then on to **domain 4 (medical imaging, 33 projects)**. Same
workflow, lead-verified.
