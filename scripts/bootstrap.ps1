# One-command setup + first data pull for Windows PowerShell (5.1 and 7+ both work).
#
#   powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -FredApiKey YOUR_KEY
#
# Re-running is safe: dependency sync and ingestion are both idempotent, and
# ingestion is incremental (it resumes from the lake watermark).
param(
    [string]$FredApiKey = "",
    [string]$Start = "2016-01-01",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "FAILED: $msg" -ForegroundColor Red; exit 1 }

# --- repo root (script lives in scripts/) ---
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))

# --- uv ---
Step "Checking for uv"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    # Prefer winget: the astral.sh web installer's direct download stalls on some
    # Windows networks (AV/proxy interference); winget's CDN is far more reliable.
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Step "Installing uv via winget"
        winget install --id astral-sh.uv -e --accept-source-agreements --accept-package-agreements --disable-interactivity
    }
    else {
        Step "Installing uv (astral.sh official installer - winget not found)"
        # PowerShell 5.1 defaults to TLS 1.0, which the download endpoint rejects
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    }
    # both installers target dirs that only land on PATH in NEW shells - patch this one
    $env:Path = "$env:USERPROFILE\.local\bin;$env:LOCALAPPDATA\Microsoft\WinGet\Links;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Fail "uv was installed but is not on PATH yet. Close this window, open a NEW PowerShell, and re-run this script."
    }
}
uv --version

# --- FRED key (persisted to the user environment, never written to the repo) ---
if ($FredApiKey -ne "") {
    Step "Storing FRED_API_KEY in your user environment (persists across sessions)"
    [Environment]::SetEnvironmentVariable("FRED_API_KEY", $FredApiKey, "User")
    $env:FRED_API_KEY = $FredApiKey
}
elseif (-not $env:FRED_API_KEY) {
    $key = Read-Host "Enter your FRED API key (Enter to skip; macro then falls back to no-vintage CSV with an audit warning)"
    if ($key) {
        [Environment]::SetEnvironmentVariable("FRED_API_KEY", $key, "User")
        $env:FRED_API_KEY = $key
    }
}

# --- dependencies (uv also fetches a suitable Python 3.11/3.12 automatically) ---
Step "Installing dependencies (uv sync)"
uv sync
if ($LASTEXITCODE -ne 0) { Fail "uv sync failed - check network / proxy and re-run" }

# --- verification gate before touching any data ---
if (-not $SkipTests) {
    Step "Running the test suite (a few minutes; the PIT corruption harness is in here)"
    uv run pytest -q
    if ($LASTEXITCODE -ne 0) { Fail "tests failed - do not ingest on a broken tree" }
}

# --- universe, then Stage-1 ingest ---
Step "Building universe (instrument master + S&P 500 PIT membership from Wikipedia)"
uv run python scripts/ingest.py --dataset universe
if ($LASTEXITCODE -ne 0) { Fail "universe ingest failed (usually transient network - re-run this script)" }

Step "Ingesting all Stage-1 datasets from $Start (long first run; incremental afterwards)"
uv run python scripts/ingest.py --dataset all --start $Start
if ($LASTEXITCODE -ne 0) {
    Write-Host "Some datasets reported errors. Ingestion is incremental and idempotent - re-running this script only re-pulls what is missing." -ForegroundColor Yellow
}

# --- factor gate preview ---
Step "Factor IC report (gate preview - nothing is recorded yet)"
uv run python scripts/build_factors.py --ic-report

Write-Host @"

Bootstrap complete. Next:
  uv run python scripts/build_factors.py --ic-report --apply   # record gate verdicts into configs\factors.yaml
  uv run python scripts/run_backtest.py                        # walk-forward backtest -> reports\

Daily refresh (Task Scheduler-friendly):
  uv run python scripts/ingest.py --dataset all
"@ -ForegroundColor Green
