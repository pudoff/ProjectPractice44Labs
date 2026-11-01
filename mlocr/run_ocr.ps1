param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$InputFile,

    [ValidateRange(1, 100000)]
    [int]$Page = 1,

    [ValidateRange(0, 600)]
    [int]$Dpi = 0,

    [string]$Output,

    [string]$CvJson,

    [string]$ModelsRoot = $env:DRAWING_OCR_MODEL_DIR,

    [string]$PythonPath,

    [switch]$AllPages
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = $PythonPath
if (-not $Python) { $Python = Join-Path $RepoRoot 'backend\.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf) -and -not $PythonPath) {
    $Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
}
if (-not $ModelsRoot) { $ModelsRoot = Join-Path $RepoRoot 'models\paddle' }
$DetModel = Join-Path $ModelsRoot 'PP-OCRv5_mobile_det_infer'
$RecModel = Join-Path $ModelsRoot 'eslav_PP-OCRv5_mobile_rec_infer'

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "OCR environment not found: $Python`nCreate it using the instructions in mlocr/README.md."
}

$ResolvedInput = (Resolve-Path -LiteralPath $InputFile).Path
$Extension = [IO.Path]::GetExtension($ResolvedInput).ToLowerInvariant()
if ($Extension -notin @('.pdf', '.png', '.jpg', '.jpeg')) {
    throw "Supported formats are PDF, PNG and JPEG; received extension $Extension."
}

foreach ($ModelDir in @($DetModel, $RecModel)) {
    if (-not (Test-Path -LiteralPath $ModelDir -PathType Container)) {
        throw "Local model weights not found: $ModelDir`nSee mlocr/models/MANIFEST.json."
    }
}

if ($AllPages -and $PSBoundParameters.ContainsKey('Page')) {
    throw 'Use either -AllPages or -Page, not both.'
}

if (-not $Output) {
    $Output = Join-Path $PSScriptRoot "results\$([IO.Path]::GetFileNameWithoutExtension($ResolvedInput)).ocr.json"
}
if (-not [IO.Path]::IsPathRooted($Output)) {
    $Output = Join-Path $RepoRoot $Output
}
$OutputDir = Split-Path -Parent $Output
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$EffectiveDpi = 300
if ($Extension -eq '.pdf') {
    if ($Dpi -gt 0) {
        if ($Dpi -lt 72) {
            throw 'Manual DPI must be between 72 and 600. Use 0 for automatic selection.'
        }
        $EffectiveDpi = $Dpi
    }
    else {
        $PageMetrics = & $Python -c 'import pymupdf,sys; d=pymupdf.open(sys.argv[1]); pages=list(d); print(max(p.rect.width*p.rect.height for p in pages), max(max(p.rect.width,p.rect.height) for p in pages))' $ResolvedInput
        if ($LASTEXITCODE -ne 0) {
            throw 'Could not read PDF page dimensions for automatic DPI selection.'
        }
        $MetricParts = (($PageMetrics | Select-Object -Last 1).ToString() -split '\s+')
        $PageAreaPt2 = [double]$MetricParts[0]
        $MaxSidePt = [double]$MetricParts[1]
        if ($PageAreaPt2 -le 0) {
            throw 'The PDF has no pages with valid dimensions.'
        }
        # Keep below PaddleOCR's 4000-pixel side limit and target <=40 MP.
        $DpiByPixelCount = [Math]::Sqrt(40000000 * 72 * 72 / $PageAreaPt2)
        $DpiBySideLimit = 3800 * 72 / $MaxSidePt
        $EffectiveDpi = [Math]::Min(300, [Math]::Max(72, [Math]::Floor([Math]::Min($DpiByPixelCount, $DpiBySideLimit))))
    }
}

$Arguments = @(
    '-m', 'mlocr.src.cli', $ResolvedInput,
    '--dpi', [string]$EffectiveDpi,
    '--det-model-dir', $DetModel,
    '--rec-model-dir', $RecModel,
    '--output', $Output
)
if (-not $AllPages) {
    $Arguments += @('--page', [string]$Page)
}
if ($CvJson) {
    $ResolvedCvJson = (Resolve-Path -LiteralPath $CvJson).Path
    $Arguments += @('--cv-json', $ResolvedCvJson)
}

Write-Host "Файл: $ResolvedInput"
if ($Extension -eq '.pdf') {
    Write-Host "Page: $(if ($AllPages) { 'all' } else { $Page }); DPI: $EffectiveDpi"
}
else {
    Write-Host 'Input is an image; DPI does not apply.'
}
Write-Host "Output: $Output"

Push-Location $RepoRoot
try {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "OCR failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
