# Build voxel_fdtd.dll (MinGW g++ + OpenMP)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$gpp = $null
$candidates = @(
    "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin\g++.exe",
    "C:\mingw64\bin\g++.exe",
    "C:\WinLibs\mingw64\bin\g++.exe"
)
Get-ChildItem "C:\Users\set\AppData\Local\Microsoft\WinGet\Packages" -Recurse -Filter "g++.exe" -ErrorAction SilentlyContinue |
    Select-Object -First 3 | ForEach-Object { $candidates += $_.FullName }
foreach ($c in $candidates) {
    if ($c -and (Test-Path $c)) { $gpp = $c; break }
}
if (-not $gpp) {
    $cmd = Get-Command g++.exe -ErrorAction SilentlyContinue
    if ($cmd) { $gpp = $cmd.Source }
}
if (-not $gpp) { throw "g++ not found. Install WinLibs MinGW." }

$bin = Split-Path $gpp
$env:PATH = "$bin;" + $env:PATH
Write-Host "Using $gpp"

& $gpp -O3 -fopenmp -std=c++17 -shared -o voxel_fdtd.dll voxel_fdtd.cpp
if ($LASTEXITCODE -ne 0) { throw "compile failed" }
Write-Host "built $here\voxel_fdtd.dll"
Copy-Item -Force "$bin\libgomp-1.dll" . -ErrorAction SilentlyContinue
Copy-Item -Force "$bin\libgcc_s_seh-1.dll" . -ErrorAction SilentlyContinue
Copy-Item -Force "$bin\libstdc++-6.dll" . -ErrorAction SilentlyContinue
Copy-Item -Force "$bin\libwinpthread-1.dll" . -ErrorAction SilentlyContinue
Write-Host "ok"
