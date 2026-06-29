# Commenting Standard — the canonical rubric

> This is the full, binding rubric referenced by `CLAUDE.md` §6. The owner asked for **"as much comment as
> possible, explaining what each function does, what each variable is for, what the logic and thought
> process is, how everything ties together."** Take it literally. **Over-comment on purpose.** A file with
> sparse comments is considered *unfinished*, no matter how well it runs.

The single test: **a stranger who knows C++ but is new to CUDA and new to the domain could learn the
concept from your file alone.** If they couldn't, add more comments.

---

## 1. The eight rules (binding)

1. **File header block** at the top of every source file: what this file is, its role in the project, the
   key idea, inputs/outputs, and a "read this after / before" pointer to sibling files. Reference the
   catalog **ID** so the file is traceable to the deep-dive entry.
2. **Every function** gets a doc-comment block: purpose, each parameter (with **units, ranges, ownership**),
   return value, side effects, complexity, and *why it exists*. For **kernels**, additionally document the
   **launch configuration** (grid/block dims and the reasoning), the memory spaces touched, whether atomics
   or shared memory are used, and the **thread-to-data mapping**.
3. **Every non-trivial variable** gets an inline note on first use: what it represents, its units, and why
   it has the type/size it does. Especially flag indices, strides, padded sizes, and device pointers.
4. **Narrate the thought process.** Before a block of logic, state the *intent* and the alternative you
   rejected. Comments answer **why**, not just restate the code.
5. **Tie it together.** Where a function hands off to another, say so ("result feeds `backproject()` in
   kernels.cu"). Cross-reference README/THEORY sections by name.
6. **Explain library calls.** Any cuBLAS/cuFFT/cuRAND/cuSOLVER/Thrust call gets 2–4 lines: what it computes
   mathematically, why we use it instead of hand-rolling, and the shape/layout of its inputs/outputs.
   **No black boxes.**
7. **CUDA error checking is always visible and explained.** Wrap API calls in `CUDA_CHECK(...)` and check
   launches with `CUDA_CHECK_LAST(...)` (both defined and commented in `src/util/cuda_check.cuh`).
8. **No commented-out dead code.** Comments teach; they don't store graveyards. Delete dead code; explain
   decisions in prose.

## 2. Density target

Aim for a comment-to-code ratio a stranger could learn from — **often ≥ 1:1 by line in kernel files**.
`tools/verify_project.py` enforces a floor (default **0.40** non-trivial comment lines per code line across
`src/`), counting only comments with real words (rulers like `// -----` don't count). **The floor is a
safety net, not the goal.** The validated template (`docs/PROJECT_TEMPLATE/`) measures ~1.0 — match it.

Run the meter:

```bash
python tools/verify_project.py <project-path>      # prints "comments: src ratio X.XX >= 0.40 ..."
```

---

## 3. Worked examples

These are excerpted from the **validated** `docs/PROJECT_TEMPLATE/` (its SAXPY placeholder builds, runs, and
passes verification). Use them as the bar.

### 3.1 A kernel (file header + launch doc + thread mapping + ragged-block guard)

```cpp
// kernels.cu  --  The GPU kernel and its host wrapper (placeholder: SAXPY)
// Project 1.12 (see ../THEORY.md and the catalog deep-dive for the "why").

// saxpy_kernel: one thread computes one output element.
//   Launch config (set in saxpy_gpu):
//     grid  = ceil(n / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: i = blockIdx.x * blockDim.x + threadIdx.x.
//   Memory: reads x[i], y[i] from global memory, writes out[i]; no shared
//   memory or atomics needed because elements are fully independent.
__global__ void saxpy_kernel(int n, float a,
                             const float* __restrict__ x,   // [n] device input
                             const float* __restrict__ y,   // [n] device input
                             float* __restrict__ out) {     // [n] device output
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's element
    // GUARD THE RAGGED LAST BLOCK: n is rarely a multiple of the block size, so
    // the final block has threads with i >= n; they must do nothing or they
    // read/write out of bounds (an illegal-address crash).
    if (i < n) {
        out[i] = a * x[i] + y[i];   // the work, run in parallel across all i
    }
}
```

What makes this pass: a **file header** naming the project/ID, a **launch-config** block, an explicit
**thread-to-data mapping**, **units/space** on every parameter, and a **why** comment on the guard (not just
"if i < n").

### 3.2 A host wrapper (the five canonical CUDA steps, each narrated)

```cpp
// saxpy_gpu: host wrapper. The five canonical steps of a CUDA computation:
//   (1) allocate device memory  (2) copy inputs host->device
//   (3) launch the kernel        (4) copy result device->host
//   (5) free device memory
// We time ONLY step (3) with CUDA events so the reported figure is kernel cost,
// not PCIe transfer cost (discussed separately in THEORY).
void saxpy_gpu(int n, float a, const std::vector<float>& x, /* host inputs */
               const std::vector<float>& y, std::vector<float>& out,
               float* kernel_ms /* out-param: GPU-measured kernel time */) {
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    float *d_x = nullptr;                       // d_ = DEVICE pointer (CLAUDE §12)
    CUDA_CHECK(cudaMalloc(&d_x, bytes));        // can fail: out of device memory
    // ... (H2D copies, ceil-division grid, launch, CUDA_CHECK_LAST, D2H, frees)
}
```

What makes this pass: the `d_`/`h_` pointer convention is **stated**, the **out-parameter** is labeled, the
**ceiling division** for the grid is explained where it occurs, and the timing decision (kernel vs. copies)
is justified.

### 3.3 A `.vcxproj` comment (yes, MSBuild XML supports comments)

```xml
<CudaCompile>
  <!-- Fat binary: real SASS for Turing(75)/Ampere(86)/Ada(89) + PTX(89) so a
       JIT can target newer cards. Narrow for faster local builds (BUILD_GUIDE). -->
  <CodeGeneration>compute_75,sm_75;compute_86,sm_86;compute_89,sm_89;compute_89,compute_89</CodeGeneration>
</CudaCompile>
```

> **XML comment trap (learned the hard way):** an XML comment **cannot contain a double hyphen `--`**. Use
> ` : `, `->`, or a single `-` as a separator inside `<!-- ... -->`, never `--`. This bites project headers
> like `Project 1.12 -- Name` (write `Project 1.12 : Name`).

If a project links a library, **comment why each is linked**, e.g.:

```xml
<!-- cuFFT: 3D forward/inverse FFT for the PME reciprocal-space sum (THEORY §4).
     Hand-rolling a batched 3D FFT at this performance is out of scope; we treat
     it as a known building block and explain what it computes, not how. -->
<AdditionalDependencies>cufft.lib;cudart_static.lib;%(AdditionalDependencies)</AdditionalDependencies>
```

### 3.4 A CPU reference (deliberately obvious)

```cpp
// out[i] = a * x[i] + y[i], computed serially on the CPU.
//   Complexity: O(n) time, O(1) extra space. This is the baseline whose wall
//   time (timed in main.cu) we compare with the GPU kernel, AND the ground
//   truth the GPU result is asserted against within tolerance.
void saxpy_cpu(int n, float a, const std::vector<float>& x,
               const std::vector<float>& y, std::vector<float>& out) {
    out.assign(static_cast<std::size_t>(n), 0.0f);
    for (int i = 0; i < n; ++i)
        out[i] = a * x[i] + y[i];   // each output depends only on its own inputs
                                    // -> WHY this parallelizes perfectly on a GPU
}
```

What makes this pass: it states **why the reference exists** (baseline + ground truth) and connects the
serial structure to the GPU mapping ("each output depends only on its own inputs").

---

## 4. Anti-patterns (these fail review)

| Anti-pattern | Fix |
|---|---|
| `i++; // increment i` | Restates code. Say *why*: `// advance to the next neighbor in the cell list` |
| A kernel with no launch-config comment | Add grid/block dims **and the reasoning** |
| `cufftExecC2C(...)` with no explanation | 2–4 lines: what it computes, why, input/output layout |
| Commented-out old code left "just in case" | Delete it; explain the decision in prose |
| `float* p;` in device code with no note | `float* d_p; // [n*FP_WORDS] device, row-major fingerprints` |
| `--` inside an XML comment | Use ` : ` / `->` / single `-` |

## 5. Markdown surfaces count too

`README.md`, `THEORY.md`, `data/README.md`, and `demo/README.md` are teaching surfaces. `THEORY.md` must
carry the science → math → algorithm → **GPU mapping** → numerics → verification → real-world arc (§4.2). A
project whose code is well-commented but whose `THEORY.md` is thin is **not done**.
