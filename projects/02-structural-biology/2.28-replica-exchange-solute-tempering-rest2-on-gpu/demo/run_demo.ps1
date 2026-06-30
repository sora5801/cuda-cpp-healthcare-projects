# ===========================================================================
# demo/run_demo.ps1  --  One command: build (if needed) + run + verify
# ---------------------------------------------------------------------------
# Project 2.28 -- Replica Exchange Solute Tempering (REST2) on GPU   (template skeleton)
#
# WHAT IT DOES (CLAUDE.md §6.3)
#   1. Ensure the Release exe exists (build it with MSBuild if missing).
#   2. Run it on data/sample/, capturing stdout (deterministic) separately
#      from stderr (timing / varies).
#   3. Diff stdout against demo/expected_output.txt (line endings normalized).
#   4. Echo the stderr timing for the learner; report PASS/FAIL and exit code.
#
# Usage:  ./demo/run_demo.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$Demo        = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $Demo
$Slug        = "replica-exchange-solute-tempering-rest2-on-gpu"
$Sln         = Join-Path $ProjectRoot "build\$Slug.sln"
$Exe         = Join-Path $ProjectRoot "build\x64\Release\$Slug.exe"
$Sample      = Join-Path $ProjectRoot "data\sample\rest2_config.txt"
$Expected    = Join-Path $Demo "expected_output.txt"

function Find-MSBuild {
    $cmd = Get-Command msbuild -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $p = & $vswhere -latest -requires Microsoft.Component.MSBuild `
                 -find "MSBuild\**\Bin\MSBuild.exe" | Select-Object -First 1
        if ($p) { return $p }
    }
    return $null
}

# --- 1. Build if the executable is missing --------------------------------
if (-not (Test-Path $Exe)) {
    Write-Host "[run_demo] Release exe not found; building $Slug ..."
    $msbuild = Find-MSBuild
    if (-not $msbuild) {
        Write-Error "MSBuild not found. Open build\$Slug.sln in Visual Studio 2026 and build Release|x64, or run from a Developer PowerShell."
        exit 2
    }
    & $msbuild $Sln /p:Configuration=Release /p:Platform=x64 /m /nologo /v:minimal
    if ($LASTEXITCODE -ne 0) { Write-Error "Build failed."; exit $LASTEXITCODE }
}

# --- 2. Run, capturing stdout and stderr separately -----------------------
# We use Start-Process with file redirection (NOT "& $Exe 2> file"): under
# Windows PowerShell 5.1 with $ErrorActionPreference='Stop', a native program
# writing ANYTHING to stderr is surfaced as a terminating NativeCommandError --
# even though our program only writes timing/info there. Start-Process keeps the
# child's stderr out of PowerShell's error stream, so the demo stays robust.
# [System.IO.Path]::GetTempFileName() works on every PowerShell edition.
Write-Host "[run_demo] Running $Slug on the committed sample ..."
$outFile = [System.IO.Path]::GetTempFileName()
$errFile = [System.IO.Path]::GetTempFileName()
$proc = Start-Process -FilePath $Exe -ArgumentList "`"$Sample`"" -NoNewWindow -Wait -PassThru `
          -RedirectStandardOutput $outFile -RedirectStandardError $errFile

# Read both streams. Get-Content -Raw returns $null for an empty file; coercing
# to [string] FIRST avoids the "$null -replace ... -> @()" array trap that then
# fails on .TrimEnd(). We also retry the stdout read briefly in case the OS has
# not finished flushing the child's redirected handle yet.
$stdoutRaw = ""
for ($try = 0; $try -lt 20; $try++) {
    $stdoutRaw = [string](Get-Content $outFile -Raw)
    if ($stdoutRaw.Trim().Length -gt 0) { break }
    Start-Sleep -Milliseconds 25
}
$stderr = ([string](Get-Content $errFile -Raw)) -replace "`r",""
Remove-Item $outFile, $errFile -ErrorAction SilentlyContinue

# Normalize stdout (strip CR) to a clean LF string.
$stdout = $stdoutRaw -replace "`r",""
$exitCode = $proc.ExitCode

# --- 3. Compare stdout to expected ----------------------------------------
$expected = ([string](Get-Content $Expected -Raw)) -replace "`r",""
$actualLines   = $stdout.TrimEnd("`n").Split("`n")
$expectedLines = $expected.TrimEnd("`n").Split("`n")

# --- 4. Report ------------------------------------------------------------
Write-Host "---- program output (stdout) ----"
Write-Host $stdout.TrimEnd()
Write-Host "---- timing / detail (stderr) ----"
Write-Host $stderr.TrimEnd()
Write-Host "----------------------------------"

$match = ($actualLines.Count -eq $expectedLines.Count)
if ($match) {
    for ($i = 0; $i -lt $expectedLines.Count; $i++) {
        if ($actualLines[$i].TrimEnd() -ne $expectedLines[$i].TrimEnd()) { $match = $false; break }
    }
}

if ($match -and $exitCode -eq 0) {
    Write-Host "[run_demo] PASS: output matches expected_output.txt and GPU==CPU." -ForegroundColor Green
    exit 0
} else {
    Write-Host "[run_demo] FAIL: output did not match expected_output.txt (or exit != 0)." -ForegroundColor Red
    exit 1
}
