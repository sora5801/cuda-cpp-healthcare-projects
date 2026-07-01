# ===========================================================================
# demo/run_demo.ps1  --  One command: build (if needed) + run + verify
# ---------------------------------------------------------------------------
# Project 6.6 : Neuronal Network Simulation (Biophysical)
#   stdout (deterministic network summary) is diffed against expected_output.txt;
#   stderr (timing / activity trace) is shown but not diffed.
# ===========================================================================
$ErrorActionPreference = "Stop"
$Demo        = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $Demo
$Slug        = "neuronal-network-simulation-biophysical"
$Sln         = Join-Path $ProjectRoot "build\$Slug.sln"
$Exe         = Join-Path $ProjectRoot "build\x64\Release\$Slug.exe"
$Sample      = Join-Path $ProjectRoot "data\sample\network.txt"
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

Write-Host "[run_demo] Running $Slug on the committed sample ..."
$outFile = New-TemporaryFile
$errFile = New-TemporaryFile
$proc = Start-Process -FilePath $Exe -ArgumentList "`"$Sample`"" -NoNewWindow -Wait -PassThru `
          -RedirectStandardOutput $outFile -RedirectStandardError $errFile
$stdout = (Get-Content $outFile -Raw) -replace "`r",""
$stderr = (Get-Content $errFile -Raw) -replace "`r",""
Remove-Item $outFile, $errFile -ErrorAction SilentlyContinue

$expected = (Get-Content $Expected -Raw) -replace "`r",""
$actualLines   = $stdout.TrimEnd("`n").Split("`n")
$expectedLines = $expected.TrimEnd("`n").Split("`n")

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

if ($match -and $proc.ExitCode -eq 0) {
    Write-Host "[run_demo] PASS: output matches expected_output.txt and GPU==CPU." -ForegroundColor Green
    exit 0
} else {
    Write-Host "[run_demo] FAIL: output did not match expected_output.txt (or exit != 0)." -ForegroundColor Red
    exit 1
}
