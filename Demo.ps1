param([string]$Python="$PSScriptRoot\.venv\Scripts\python.exe")
$ErrorActionPreference='Stop'
Push-Location $PSScriptRoot
try {
  & $Python scripts/accept.py walnut
  if ($LASTEXITCODE) { throw 'Walnut reconstruction or numerical reference gate failed' }
  & $Python scripts/export_volume.py data/walnut-corrected-256.f32 --n 256 --fov 40 --out reports
  if ($LASTEXITCODE) { throw 'Slice viewer / volume export failed' }
  Write-Host "Offline viewer: $PSScriptRoot\reports\walnut-viewer.html"
} finally { Pop-Location }
