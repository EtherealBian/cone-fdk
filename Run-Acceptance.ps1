param([switch]$CoreOnly, [switch]$SkipBuild, [string]$Python="$PSScriptRoot\.venv\Scripts\python.exe", [int]$DownloadJobs=6)
$ErrorActionPreference='Stop'
function Run-Python { & $Python @args; if ($LASTEXITCODE -ne 0) {throw ('Stage failed: ' + ($args -join ' '))} }
Push-Location $PSScriptRoot
try {
  if (!(Test-Path $Python)) {throw 'Install requirements-win.lock in a Python 3.12 x64 venv first.'}
  if ($env:PYTHONOPTIMIZE) {throw 'Unset PYTHONOPTIMIZE: acceptance checks require assertions.'}
  if (!$SkipBuild) { & "$PSScriptRoot\Build.ps1" -Python $Python }
  New-Item -ItemType Directory -Force reports,data | Out-Null
  Run-Python -m pytest tests -q --junitxml=reports/native-tests.xml
  Run-Python scripts/accept.py all
  Run-Python scripts/validate_plane_phantom.py
  Run-Python scripts/validate_stack.py phantom
  Run-Python scripts/validate_cylinder.py phantom
  Run-Python scripts/validate_wall.py
  Run-Python scripts/validate_bore.py
  Run-Python scripts/validate_loss.py
  Run-Python scripts/validate_axis.py
  Run-Python scripts/validate_remount.py
  Run-Python scripts/validate_witness.py --edge-review --region-review
  Run-Python scripts/export_volume.py data/walnut-corrected-256.f32 --n 256 --fov 40 --out reports
  Run-Python scripts/benchmark.py
  if (!$CoreOnly) {
    Run-Python scripts/stress.py
    Run-Python scripts/fetch.py --jobs $DownloadJobs
    Run-Python scripts/full_walnut.py
    Run-Python scripts/validate_roi.py
    Run-Python scripts/validate_roi_pack.py
    Run-Python scripts/validate_planner.py
    Run-Python scripts/validate_planes.py
    Run-Python scripts/validate_stack.py walnut
    Run-Python scripts/validate_cylinder.py walnut
    Run-Python scripts/report.py
    Write-Host "Complete: $PSScriptRoot\reports\RESULTS.md"
  } else {Write-Host 'Core gates passed. Oversize and 720-view scans were not rerun.'}
} finally {Pop-Location}
