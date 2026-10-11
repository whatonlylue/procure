# Stage a self-contained Tesseract 5 (Windows) into src-tauri/resources/tesseract/.
# NOTE: this path is only exercised in CI (windows-latest); it has not been
# tested on a local Windows machine.
#
# Installs the UB-Mannheim build via Chocolatey if tesseract.exe is not found,
# then copies tesseract.exe + every DLL next to it, plus the pinned,
# sha256-verified tessdata_fast eng.traineddata.
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

$TessdataCommit = '87416418657359cb625c412a48b6e1d6d41c29bd'
$TessdataSha256 = '7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2'
$TessdataUrl = "https://github.com/tesseract-ocr/tessdata_fast/raw/$TessdataCommit/eng.traineddata"

$InstallDir = 'C:\Program Files\Tesseract-OCR'
if (-not (Test-Path "$InstallDir\tesseract.exe")) {
    choco install tesseract -y --no-progress
    if ($LASTEXITCODE -ne 0) { throw 'choco install tesseract failed' }
}
if (-not (Test-Path "$InstallDir\tesseract.exe")) { throw "tesseract.exe not found in $InstallDir" }

$Dest = 'src-tauri/resources/tesseract'
if (Test-Path $Dest) { Remove-Item -Recurse -Force $Dest }
New-Item -ItemType Directory -Force -Path "$Dest/tessdata", "$Dest/LICENSES" | Out-Null

Copy-Item "$InstallDir\tesseract.exe" $Dest
Copy-Item "$InstallDir\*.dll" $Dest

$Eng = Join-Path $Dest 'tessdata/eng.traineddata'
Invoke-WebRequest -Uri $TessdataUrl -OutFile $Eng
$Actual = (Get-FileHash -Algorithm SHA256 $Eng).Hash.ToLower()
if ($Actual -ne $TessdataSha256) { throw "eng.traineddata sha256 mismatch: $Actual != $TessdataSha256" }

# Licenses: whatever the installer ships (e.g. doc\ or LICENSE files).
Get-ChildItem $InstallDir -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^(LICENSE|COPYING|NOTICE)' } |
    ForEach-Object { Copy-Item $_.FullName (Join-Path "$Dest/LICENSES" ($_.Directory.Name + '-' + $_.Name)) }
@'
Bundles Tesseract OCR (Apache-2.0) and Leptonica (BSD-2-Clause) plus the DLLs
shipped by the UB-Mannheim Windows build (https://github.com/UB-Mannheim/tesseract).
Language data: tessdata_fast eng.traineddata (Apache-2.0).
'@ | Set-Content (Join-Path "$Dest/LICENSES" 'README.txt')

# Smoke test: the staged binary must run on its own.
$env:TESSDATA_PREFIX = (Resolve-Path "$Dest/tessdata").Path
& "$Dest/tesseract.exe" --version
if ($LASTEXITCODE -ne 0) { throw 'staged tesseract.exe failed to run' }
$size = [math]::Round(((Get-ChildItem $Dest -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), 1)
Write-Host "staged $Dest ($size MB)"
