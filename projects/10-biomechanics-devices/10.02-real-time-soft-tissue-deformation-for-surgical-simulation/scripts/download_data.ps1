# ===========================================================================
# scripts/download_data.ps1  --  Realistic organ-mesh pointers (Windows)
# ---------------------------------------------------------------------------
# Project 10.02 : Real-Time Soft-Tissue Deformation. Nothing to download.
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 10.02 -- Real-Time Soft-Tissue Deformation (PBD)"
Write-Host ""
Write-Host "There is no file to download: the mesh is built from the parameters in"
Write-Host "data/sample/cloth_params.txt."
Write-Host ""
Write-Host "For REAL organ meshes + GPU PBD frameworks:"
Write-Host "  SOFA  : https://github.com/sofa-framework/sofa  (physics + haptics)"
Write-Host "  iMSTK : https://github.com/Kitware/iMSTK        (CUDA deformation)"
Write-Host "  FleX  : https://github.com/NVIDIAGameWorks/FleX (GPU PBD particles)"
Write-Host "  Patient meshes: segment CT/MRI (e.g., 3D Slicer) into tetra/surface meshes."
Write-Host ""
Write-Host "Bigger mesh (no download):"
Write-Host "  python scripts/make_synthetic.py --R 128 --C 128 --steps 600"
Write-Host ""
Write-Host "Target data dir: $ProjectRoot\data"
