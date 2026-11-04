# Keep every cache, temp file and download inside this workspace (nothing on C:).
# Usage (PowerShell, from anywhere):  . "d:\Machine Learning\lexsiai hackathon\scripts\env.ps1"
# Dot-source it in every new shell; the settings only live in the current process.

$ws = Split-Path -Parent $PSScriptRoot
$cache = Join-Path $ws ".cache"

$dirs = @{
    PIP_CACHE_DIR       = "pip"
    UV_CACHE_DIR        = "uv"
    XDG_CACHE_HOME      = "xdg"
    HF_HOME             = "huggingface"
    TORCH_HOME          = "torch"
    WANDB_DIR           = "wandb"
    WANDB_CACHE_DIR     = "wandb-cache"
    WANDB_CONFIG_DIR    = "wandb-config"
    MPLCONFIGDIR        = "matplotlib"
    PYTHONPYCACHEPREFIX = "pycache"
    TMP                 = "tmp"
    TEMP                = "tmp"
    TMPDIR              = "tmp"
}
foreach ($k in $dirs.Keys) {
    $p = Join-Path $cache $dirs[$k]
    New-Item -ItemType Directory -Force -Path $p | Out-Null
    Set-Item -Path "Env:$k" -Value $p
}

# Fail fast if a tool still points at C: (informational check, does not modify anything).
$env:PYTHONUTF8 = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:PATH = (Join-Path $ws ".venv\Scripts") + ";" + $env:PATH
Write-Output "workspace env active: caches -> $cache"
