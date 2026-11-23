param(
    [string]$Python = "$PSScriptRoot\.venv\Scripts\python.exe",
    [string]$Cxx = $env:CXX,
    [switch]$VerifyRepeat
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
$oldPath = $env:Path
try {
    if (!(Test-Path $Python)) { throw 'Create a Python 3.12 venv and install requirements-win.lock first.' }
    New-Item -ItemType Directory -Force build | Out-Null
    if (!$Cxx) {
        foreach ($c in 'cl.exe','g++.exe','clang++.exe') {
            $t = Get-Command $c -ErrorAction SilentlyContinue
            if ($t) { $Cxx = $t.Source; break }
        }
    }
    if (!$Cxx) { throw 'Use an x64 MSVC Developer PowerShell or set CXX to an x64 MinGW g++.exe.' }
    $compiler = (Get-Command $Cxx -ErrorAction Stop).Source
    $env:Path = "$(Split-Path $compiler);$oldPath"
    $msvc = (Split-Path $compiler -Leaf) -eq 'cl.exe'
    $passes = if ($VerifyRepeat) { 2 } else { 1 }
    for ($i = 0; $i -lt $passes; $i++) {
        if ($msvc) {
            & $compiler /nologo /EHsc /O2 /std:c++14 /Fobuild/main.obj /Febuild/cone-fdk.exe src/main.cpp shell32.lib /link /Brepro
        } else {
            & $compiler -O3 -std=c++14 -static '-Wl,--no-insert-timestamp' src/main.cpp -lshell32 -o build/cone-fdk.exe
        }
        if ($LASTEXITCODE) { throw 'Native C++ build failed' }
        & $Python scripts/ptx.py
        if ($LASTEXITCODE) { throw 'CUDA C++ compilation failed' }
        if ($VerifyRepeat) {
            $exe = (Get-FileHash -Algorithm SHA256 build/cone-fdk.exe).Hash.ToLowerInvariant()
            $ptx = (Get-FileHash -Algorithm SHA256 build/kernels.ptx).Hash.ToLowerInvariant()
            if ($i -eq 0) { $firstExe = $exe; $firstPtx = $ptx }
            elseif ($exe -ne $firstExe -or $ptx -ne $firstPtx) { throw 'Rebuilds are not byte-identical' }
        }
    }
    if ($VerifyRepeat) {
        $tool = if ($msvc) { 'MSVC ' + (Get-Item $compiler).VersionInfo.FileVersion } else { (& $compiler --version)[0] }
        $nvrtc = & $Python -c "from importlib.metadata import version; print(version('nvidia-cuda-nvrtc-cu12'))"
        if ($LASTEXITCODE) { throw 'Could not read the NVRTC version' }
        $report = [ordered]@{compiler=$tool; nvrtc=$nvrtc; rebuilds=2; executable_sha256=$exe; ptx_sha256=$ptx; identical=$true}
        New-Item -ItemType Directory -Force reports | Out-Null
        [IO.File]::WriteAllText("$PSScriptRoot\reports\build-repro.json", ($report | ConvertTo-Json) + "`n", [Text.UTF8Encoding]::new($false))
    }
} finally {
    $env:Path = $oldPath
    Pop-Location
}
