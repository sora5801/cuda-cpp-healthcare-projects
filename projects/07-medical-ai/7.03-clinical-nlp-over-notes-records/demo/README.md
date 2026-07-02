# Demo — 7.3 Clinical NLP over Notes & Records

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed synthetic sample `data/sample/notes_sample.txt`.
3. **Verify** the GPU result (one transformer self-attention encoder block,
   computed with cuBLAS batched DGEMM + a hand-written softmax kernel) against the
   plain-C++ CPU reference (`reference_cpu.cpp`), printing a clear `PASS`/`FAIL`.
4. **Time** each GPU stage (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric CPU/GPU error (which vary run to
  run), so it is shown but never diffed.

## Expected result (stdout)

```
7.3 -- Clinical NLP over Notes & Records
batch: B=4 notes, S=8 tokens, D=8 dim, H=2 heads (dh=4) [SYNTHETIC]
per-note attention (head 0): pronoun 'he' attends most to ->
  note 0: 'he'@5 -> 'patient'@1  (weight=0.1972)
  note 1: 'he'@5 -> 'patient'@1  (weight=0.1883)
  note 2: 'he'@5 -> 'patient'@1  (weight=0.2304)
  note 3: 'he'@4 -> 'patient'@1  (weight=0.1885)
coreference link 'he'->'patient' recovered in 4 / 4 notes
per-note [CLS] summary (head 0 attention entropy; output L2 norm):
  note 0: entropy=1.8949 nats   ||CLS_out||=0.7638
  note 1: entropy=1.9367 nats   ||CLS_out||=0.8929
  note 2: entropy=1.9316 nats   ||CLS_out||=0.4693
  note 3: entropy=1.7473 nats   ||CLS_out||=0.8754
RESULT: PASS (GPU attention matches CPU within tol)
```

## How to read it

- The **coreference line** is the headline result: the synthetic batch plants a
  link (the pronoun `he` shares the embedding of `patient`), and a correct
  self-attention block makes `he` attend most strongly to `patient`. All 4 notes
  recover it. (Note the `patient` key sits at position 1 in every note, right
  after `[CLS]`.)
- The **attention weight** (~0.19) is only modestly above uniform (1/8 = 0.125)
  because our fabricated projection weights are near-identity, so attention is
  fairly diffuse — but `patient` is reliably the *argmax*. High **entropy**
  (near `log 8 ≈ 2.08` nats) reflects that diffuseness; this is honest, not a bug.
- The **stderr `[verify]` lines** show the worst CPU-vs-GPU difference (~1e-16),
  far under the documented `1e-11` tolerance — the GPU's batched DGEMM and the
  CPU's serial loops agree to machine precision.

## Note

This is a clearly-labeled **reduced-scope teaching version** of clinical NLP: it
implements the *attention mechanism* at the heart of a clinical transformer, not a
trained model. See the project `README.md` "Limitations" and `THEORY.md`
"real world" sections.
