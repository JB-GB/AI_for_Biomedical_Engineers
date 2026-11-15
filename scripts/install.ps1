# RAUSNet: install everything with one command (Windows, PowerShell 5.1 or 7).
#
#   irm https://raw.githubusercontent.com/JB-GB/AI_for_Biomedical_Engineers/PIA_devl/scripts/install.ps1 | iex
#
# Inside a clone:  powershell -ExecutionPolicy Bypass -File scripts\install.ps1 [-Cpu] [-AppOnly] [-Dir D:\RAUSNet]
# Options also work as environment variables (needed with "irm | iex"):
#   RAUSNET_CPU=1      PyTorch without CUDA          RAUSNET_APP_ONLY=1  desktop app only (no PyTorch)
#   RAUSNET_DIR=path   where to clone (default ~\RAUSNet)    RAUSNET_BRANCH=name (default PIA_devl)
#   RAUSNET_NO_SHORTCUT=1  do not create the desktop shortcut
#
# Steps: find Python 3.11+ and Git -> clone or update the repository -> scripts/setup_env.py
# (virtual environment, PyTorch for the GPU if nvidia-smi is found, all extras) -> tests a model load
# -> desktop shortcut "RAUSNet". Nothing outside the folder and the shortcut is changed.
param(
    [switch]$Cpu,
    [switch]$AppOnly,
    [string]$Dir = $(if ($env:RAUSNET_DIR) { $env:RAUSNET_DIR } else { Join-Path $HOME "RAUSNet" }),
    [string]$Branch = $(if ($env:RAUSNET_BRANCH) { $env:RAUSNET_BRANCH } else { "PIA_devl" })
)
$ErrorActionPreference = "Stop"
$Repo = "https://github.com/JB-GB/AI_for_Biomedical_Engineers.git"
if ($env:RAUSNET_CPU -eq "1") { $Cpu = $true }
if ($env:RAUSNET_APP_ONLY -eq "1") { $AppOnly = $true }

function Step($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }
# throw, not exit: under "irm | iex" exit would close the user's PowerShell window
function Fail($text) { Write-Host "`nERROR: $text" -ForegroundColor Red; throw "RAUSNet install failed: $text" }

# 1. Python 3.11+ (the py launcher first, then python on PATH)
Step "Looking for Python 3.11 or newer"
$Python = $null
foreach ($candidate in @(@("py", "-3.11"), @("py", "-3.12"), @("py", "-3.13"), @("python"))) {
    try {
        $exe = $candidate[0]; $rest = @($candidate | Select-Object -Skip 1)
        $ver = & $exe @rest -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and [version]$ver -ge [version]"3.11") { $Python = $candidate; break }
    } catch { }
}
if (-not $Python) {
    Fail "Python 3.11+ not found. Install it (for example: winget install -e --id Python.Python.3.11), then run this command again."
}
Write-Host "Python: $($Python -join ' ') ($ver)"

# 2. Repository: use the clone this script lives in, otherwise clone or update one
Step "Getting the code"
$Root = $null
if ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot "..\pyproject.toml"))) {
    $Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    Write-Host "Using this clone: $Root"
} else {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Fail "Git not found. Install it (for example: winget install -e --id Git.Git), then run this command again."
    }
    if (Test-Path (Join-Path $Dir ".git")) {
        git -C $Dir fetch --quiet origin $Branch; if ($LASTEXITCODE) { Fail "git fetch failed" }
        git -C $Dir checkout --quiet $Branch; if ($LASTEXITCODE) { Fail "git checkout failed" }
        git -C $Dir pull --quiet --ff-only origin $Branch; if ($LASTEXITCODE) { Fail "git pull failed (local changes?)" }
        Write-Host "Updated $Dir"
    } else {
        git clone --quiet --branch $Branch $Repo $Dir; if ($LASTEXITCODE) { Fail "git clone failed" }
        Write-Host "Cloned into $Dir"
    }
    $Root = (Resolve-Path $Dir).Path
}

# 3. Environment and packages
$setupArgs = @()
if ($AppOnly) {
    $setupArgs += "--app-only"; Step "Installing the desktop app (no PyTorch)"
} elseif ($Cpu -or -not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    $setupArgs += "--cpu"; Step "Installing everything (PyTorch for CPU: no NVIDIA GPU found)"
} else {
    Step "Installing everything (PyTorch with CUDA for the NVIDIA GPU)"
}
Push-Location $Root
try {
    $exe = $Python[0]; $rest = @($Python | Select-Object -Skip 1)
    & $exe @rest scripts/setup_env.py @setupArgs
    if ($LASTEXITCODE) { Fail "setup failed (see the messages above)" }

    # 4. Check that the bundled model loads through the API (no window)
    Step "Checking the bundled model"
    $venvPy = Join-Path $Root ".venv\Scripts\python.exe"
    & $venvPy -c "from rausnet.serve.api import resolve_model_dir; from rausnet.serve.inference import Predictor; p = Predictor(resolve_model_dir('release/workspace/models')); print('model', p.info['version'], 'ready')"
    if ($LASTEXITCODE) { Fail "the model could not be loaded" }
} finally { Pop-Location }

# 5. Desktop shortcut
$app = Join-Path $Root ".venv\Scripts\rausnet-desktop.exe"
if ($env:RAUSNET_NO_SHORTCUT -eq "1") {
    Write-Host "Desktop shortcut skipped"
} else {
    try {
        $lnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "RAUSNet.lnk"
        $shell = New-Object -ComObject WScript.Shell
        $sc = $shell.CreateShortcut($lnk); $sc.TargetPath = $app; $sc.WorkingDirectory = $Root
        $sc.Description = "RAUSNet - synovitis grading (research use only)"; $sc.Save()
        Write-Host "Shortcut: $lnk"
    } catch { Write-Host "No desktop shortcut created ($($_.Exception.Message))" }
}

Write-Host "`nRAUSNet is installed in $Root" -ForegroundColor Green
Write-Host "Start the app:  & `"$app`"   (or the RAUSNet shortcut; --lang en for English)"
Write-Host "Research use only: the bundled model is trained on synthetic images."
