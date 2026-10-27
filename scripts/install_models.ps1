# install_models.ps1 — Windows PowerShell model installer for stt
param (
    [switch]$All,
    [string]$Dest
)

$ErrorActionPreference = "Stop"

if (-not $Dest) {
    if ($env:STT_MODELS) {
        $Dest = $env:STT_MODELS
    } elseif ($env:LOCALAPPDATA) {
        $Dest = Join-Path $env:LOCALAPPDATA "stt\models"
    } else {
        $Dest = Join-Path $HOME "AppData\Local\stt\models"
    }
}

if (-not (Test-Path $Dest)) {
    New-Item -ItemType Directory -Force -Path $Dest | Out-Null
}

Write-Host "Target models directory: $Dest"

$ASR = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
$ENH = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models"

function Install-ModelTarball([string]$url, [string]$filename) {
    $folderName = $filename -replace "\.tar\.bz2$", ""
    $finalPath = Join-Path $Dest $folderName
    if ((Test-Path $finalPath) -and (Get-ChildItem $finalPath -Filter "*.onnx")) {
        Write-Host "Already installed: $folderName"
        return
    }

    $tarPath = Join-Path $Dest $filename
    Write-Host "Downloading $filename..."
    Invoke-WebRequest -Uri "$url/$filename" -OutFile $tarPath

    Write-Host "Extracting $filename..."
    # Use tar.exe which is built into modern Windows 10/11
    if (Get-Command tar.exe -ErrorAction SilentlyContinue) {
        tar.exe -xjf $tarPath -C $Dest
    } else {
        # Fallback to python extraction if available
        python -c "import tarfile; tarfile.open(r'$tarPath').extractall(r'$Dest')"
    }
    Remove-Item -Force $tarPath
    Write-Host "OK: $folderName"
}

function Install-RawFile([string]$url, [string]$filename) {
    $finalPath = Join-Path $Dest $filename
    if ((Test-Path $finalPath) -and ((Get-Item $finalPath).Length -gt 0)) {
        Write-Host "Already installed: $filename"
        return
    }
    Write-Host "Downloading $filename..."
    Invoke-WebRequest -Uri "$url/$filename" -OutFile $finalPath
    Write-Host "OK: $filename"
}

# 1. Offline Parakeet ASR
Install-ModelTarball $ASR "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"
# 2. Streaming Nemotron ASR
Install-ModelTarball $ASR "sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25.tar.bz2"
# 3. Silero VAD
Install-RawFile $ASR "silero_vad.onnx"
# 4. Default DPDFNet8 Denoiser
Install-RawFile $ENH "dpdfnet8.onnx"

if ($All) {
    Install-ModelTarball $ASR "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06.tar.bz2"
    foreach ($extra in @("dpdfnet2.onnx", "dpdfnet4.onnx", "gtcrn_simple.onnx")) {
        Install-RawFile $ENH $extra
    }
}

Write-Host "All models installed successfully into $Dest"
