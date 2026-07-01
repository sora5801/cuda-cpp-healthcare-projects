# Build Guide — install the toolchain and build any project

> Canonical, copy-paste-friendly guide for building any project in this repo on **Windows + NVIDIA**. The
> required deliverable is the **Visual Studio solution**; an optional `CMakeLists.txt` is provided for
> Linux/macOS/CI. If a build step ever changes, update this file in the same push (`CLAUDE.md` §5).

## 0. Verified configuration

This repo's build standard was validated end-to-end on:

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 2080 (8 GB), **compute capability `sm_75`** (Turing) |
| Driver | 591.86 (advertises CUDA 13.1; CUDA 13.x minor-version compatibility covers the 13.3 toolkit) |
| CUDA Toolkit | **13.3** (`nvcc` V13.3.33) |
| Visual Studio | **2026 Community**, MSVC 14.51, **`v145`** platform toolset |
| MSBuild | `…\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe` |

> **Why 13.3 + VS 2026, not 12.x + VS 2022?** The original contract specified 12.x/2022; the owner's machine
> has 13.3/2026 and that was ratified as the standard (`CLAUDE.md` §5). CUDA 13 dropped Maxwell/Pascal/Volta,
> so **`sm_75` (Turing) is the architecture floor** — which is exactly the bottom of our arch list.

## 1. Install prerequisites

1. **Visual Studio 2026 Community** with the **"Desktop development with C++"** workload (this provides
   `cl.exe`, MSBuild, and the Windows SDK).
2. **CUDA Toolkit 13.3** — during install, keep **"Visual Studio Integration"** checked. This drops the
   `CUDA 13.3.props/.targets/.xml` build customization into VS so `.cu` files compile. Verify it landed:

   ```powershell
   Get-ChildItem "C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Microsoft\VC\v180\BuildCustomizations\" -Filter "CUDA*"
   # expect: CUDA 13.3.props, CUDA 13.3.targets, CUDA 13.3.Version.props, CUDA 13.3.xml
   ```

3. Confirm the tools are on PATH:

   ```powershell
   nvcc --version          # -> release 13.3
   nvidia-smi              # -> your GPU + driver
   ```

## 2. Detect your GPU's compute capability

Two ways:

```powershell
# A) one-liner (newer drivers):
nvidia-smi --query-gpu=name,compute_cap --format=csv

# B) what nvcc can emit (sanity-check your card is in range):
nvcc --list-gpu-code     # must list your card's sm_XX (e.g. sm_75)
```

Map: RTX 20xx → `sm_75` · RTX 30xx → `sm_86` · RTX 40xx → `sm_89`.

## 3. (Optional) narrow the architecture list for faster local builds

Every `.vcxproj` ships a **fat** code-generation list so binaries run on most cards:

```
compute_75,sm_75;compute_86,sm_86;compute_89,sm_89;compute_89,compute_89   (last = PTX for JIT)
```

Compiling three architectures takes ~3× longer. For quick local iteration, set it to **just your card**
(e.g. `sm_75`): in VS, **Project → Properties → CUDA C/C++ → Device → Code Generation**, or override on the
MSBuild CLI:

```powershell
msbuild build\<slug>.sln /p:Configuration=Release /p:Platform=x64 `
  /p:CodeGeneration="compute_75,sm_75"
```

> Keep the committed `.vcxproj` on the **fat** list (portability for other learners); narrow only locally.

## 4. Build a project

### 4a. In Visual Studio (the canonical path)

1. Open `projects/<domain>/<id>-<slug>/build/<slug>.sln`.
2. Set the configuration dropdown to **`Release`** and platform to **`x64`**.
3. **Build → Build Solution** (`Ctrl+Shift+B`).
4. The executable lands in `build/x64/Release/<slug>.exe`.

Use **`Debug|x64`** to step through kernels in **Nsight** (`-G` device debug is enabled there).

### 4b. From the command line

From a **Developer PowerShell for VS 2026** (so `msbuild` is on PATH):

```powershell
cd projects\<domain>\<id>-<slug>
msbuild build\<slug>.sln /p:Configuration=Release /p:Platform=x64 /m
```

Or call MSBuild by full path from any shell:

```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe" `
  build\<slug>.sln /p:Configuration=Release /p:Platform=x64 /m /v:minimal
```

## 5. Run the demo

```powershell
.\demo\run_demo.ps1
```

It builds if needed, runs on `data/sample/`, prints the deterministic result (diffed against
`demo/expected_output.txt`), shows the **GPU-vs-CPU agreement** check, and prints a timing line. A green
`PASS` means the GPU result matches the CPU reference within tolerance.

## 6. Verify structure & comment density

```powershell
python tools\verify_project.py projects\<domain>\<id>-<slug>   # one project
python tools\verify_project.py --all                          # whole repo sweep
```

A fresh skeleton reports **NOT DONE: scaffold TODOs remain** — that's expected until the project is built out.

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `MSB4025: An XML comment cannot contain '--'` | A `--` inside a `<!-- -->` comment in the `.vcxproj`. Replace with ` : ` / `->` / single `-`. |
| `error MSB4019: The imported project "…CUDA 13.3.props" was not found` | CUDA VS integration not installed (or a different CUDA version). Reinstall the toolkit with VS integration, or change the two `CUDA 13.3` import filenames in the `.vcxproj` to your version. |
| `msbuild` not recognized | Use the **Developer PowerShell for VS 2026**, or call MSBuild by full path (§4b). |
| `nvcc fatal: Unsupported gpu architecture 'compute_50'` | CUDA 13 dropped pre-Turing archs. Keep the arch list at `sm_75`+ only. |
| Runtime error about CUDA driver version | Driver older than the toolkit's minor. Within CUDA 13.x this is usually fine (minor-version compatibility); update the driver if a kernel refuses to launch. |
| Demo `FAIL` but build OK | The program's stdout drifted from `demo/expected_output.txt`. Regenerate expected output, or check you didn't print varying data (timings) to **stdout** (they belong on **stderr**). |

## 7b. Linking a CUDA library (cuFFT, cuSOLVER, cuBLAS, …)

Some projects use a CUDA library (e.g. `8.03` cuFFT, `2.06` cuSOLVER). Add the
`.lib` to **both** the Debug and Release `<Link>` sections of the `.vcxproj`:

```xml
<AdditionalDependencies>cufft.lib;cudart_static.lib;%(AdditionalDependencies)</AdditionalDependencies>
<!-- cuSOLVER also needs cuBLAS + cuSPARSE: -->
<AdditionalDependencies>cusolver.lib;cublas.lib;cusparse.lib;cudart_static.lib;%(AdditionalDependencies)</AdditionalDependencies>
```

and to `CMakeLists.txt` for the optional Linux build:

```cmake
find_package(CUDAToolkit REQUIRED)
target_link_libraries(<slug> PRIVATE CUDA::cufft)   # or CUDA::cusolver CUDA::cublas CUDA::cusparse
```

The library headers (`<cufft.h>`, `<cusolverDn.h>`, …) and `.lib` paths come from
the CUDA build customization automatically — no manual include/lib paths needed.
See **[docs/PATTERNS.md](PATTERNS.md) §5** for the "no black box" documentation rule.

## 7c. Using Thrust / CUB (header-only)

Thrust (`thrust::sort_by_key`, `reduce_by_key`, …) and CUB ship **with the toolkit** — no
extra `.lib` to link (only the usual `cudart_static.lib`). But the heavy template headers
need a couple of `.vcxproj` knobs to compile cleanly under MSVC + `nvcc` (first used by
`3.26` GPU BAM sort/dedup):

- Host compiler: add **`/Zc:preprocessor`** (conformant preprocessor) to `<ClCompile>`'s
  `AdditionalOptions`, and make sure C++17 is on (`<LanguageStandard>stdcpp17</LanguageStandard>`).
- Device compiler: pass **`-std=c++17`** to `nvcc` (CUDA C/C++ → Command Line / `AdditionalOptions`).
- In **Debug only** (`-G` device debug), Thrust emits two benign diagnostics; silence them with
  **`-diag-suppress 20011,20014`** so the "zero new warnings" gate stays meaningful. Do **not**
  add these to Release (they are unnecessary there).

All four switches are plain `.vcxproj` edits (no path hardcoding). For the optional CMake build,
Thrust/CUB are found via `find_package(CUDAToolkit)` and need no extra `target_link_libraries`.

## 7d. Using cuSPARSE (sparse linear algebra)

cuSPARSE (`cusparseSpMV`, sparse solves, format conversions) links like the other libraries —
add `cusparse.lib` to **both** `<Link>` sections (§7b) and `CUDA::cusparse` in CMake. One extra
gotcha (first hit by `5.02` fluence-map optimization): parts of the modern **generic** API are
marked deprecated in `<cusparse.h>`, which raises MSVC **C4996** warnings and would fail the
"zero new warnings" gate. Define **`DISABLE_CUSPARSE_DEPRECATED`** (Project → C/C++ →
Preprocessor, or `-D` on the `nvcc` command line) to silence exactly those deprecation notices
without hiding real warnings. Comment the define in the `.vcxproj` so the reason is visible.

## 8. CI note

A GitHub Actions workflow can **compile** changed projects (hosted runners have the toolkit) but **cannot run
kernels** — hosted runners have **no NVIDIA GPU**. Running/demoing is a **local** step. Never let a green
"build" badge imply the kernels executed in CI (`CLAUDE.md` §9).
