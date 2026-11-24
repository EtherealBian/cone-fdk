param([string]$Cxx=$env:CXX,[string]$Python="$PSScriptRoot\..\.venv\Scripts\python.exe")
$ErrorActionPreference='Stop'
Push-Location "$PSScriptRoot\.."
$old=$env:Path
try {
    if (!$Cxx) { foreach($c in 'cl.exe','g++.exe','clang++.exe'){ $t=Get-Command $c -ErrorAction SilentlyContinue; if($t){$Cxx=$t.Source;break} } }
    if (!$Cxx) { throw 'Set CXX or pass -Cxx with an x64 C++ compiler.' }
    $compiler=(Get-Command $Cxx -ErrorAction Stop).Source
    $env:Path="$(Split-Path $compiler);$old"
    New-Item -ItemType Directory -Force build/pores | Out-Null
    if((Split-Path $compiler -Leaf) -eq 'cl.exe') { & $compiler /nologo /EHsc /O2 /std:c++14 /Fobuild/pores/main.obj /Febuild/pores/cone-pores.exe pores/main.cpp shell32.lib /link /Brepro }
    else { & $compiler -O3 -std=c++14 -static '-Wl,--no-insert-timestamp' pores/main.cpp -lshell32 -o build/pores/cone-pores.exe }
    if($LASTEXITCODE){throw 'Native pore analysis build failed'}
    & $Python scripts/ptx.py pores/kernels.cu build/pores/pores.ptx
    if($LASTEXITCODE){throw 'Pore PTX compilation failed'}
    & $Python scripts/ptx.py pores/nlm.cu build/pores/nlm.ptx
    if($LASTEXITCODE){throw 'NLM compilation failed'}
    & $Python scripts/ptx.py pores/patches.cu build/pores/patches.ptx
    if($LASTEXITCODE){throw 'Patch NLM compilation failed'}
    & $Python scripts/ptx.py pores/multiscale.cu build/pores/multiscale.ptx
    if($LASTEXITCODE){throw 'Multiscale NLM compilation failed'}
    & $Python scripts/ptx.py pores/adaptive.cu build/pores/adaptive.ptx
    if($LASTEXITCODE){throw 'Adaptive NLM compilation failed'}
    & $Python scripts/ptx.py pores/rankpatch.cu build/pores/rankpatch.ptx
    if($LASTEXITCODE){throw 'Rank patch compilation failed'}
    & $Python scripts/ptx.py pores/transfer_faces.cu build/pores/transfer_faces.ptx
    if($LASTEXITCODE){throw 'Transfer boundary compilation failed'}
    & $Python scripts/ptx.py pores/spatial.cu build/pores/spatial.ptx
    if($LASTEXITCODE){throw 'Spatial convolution compilation failed'}
    & $Python scripts/ptx.py pores/patch_stats.cu build/pores/patch-stats.ptx
    if($LASTEXITCODE){throw 'Patch statistic compilation failed'}
    & $Python scripts/ptx.py pores/latent.cu build/pores/latent.ptx
    if($LASTEXITCODE){throw 'Latent decision compilation failed'}
    & $Python scripts/ptx.py pores/search_support.cu build/pores/search-support.ptx
    if($LASTEXITCODE){throw 'Search support compilation failed'}
    & $Python scripts/ptx.py pores/inverse_weight.cu build/pores/inverse-weight.ptx
    if($LASTEXITCODE){throw 'Inverse patch weight compilation failed'}
} finally {$env:Path=$old;Pop-Location}
