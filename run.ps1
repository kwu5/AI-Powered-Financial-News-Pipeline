# run.ps1 — daily-use launcher for FinNews-RAG.
#
# Two phases (see IMPLEMENTATION_PLAN.md / src/pipeline.py):
#   1. Refresh: fetch + index today's news and regenerate the briefing
#      (python main.py --mode run-once).
#   2. Serve:   launch the Streamlit Q&A app over the freshly-built index.
#
# Refresh runs BEFORE Streamlit so the app's cached VectorStore sees the new
# articles (app.py caches it via @st.cache_resource for the session).
#
# Usage:
#   .\run.ps1              # refresh news, then open the app
#   .\run.ps1 -NoRefresh   # skip the fetch (no RSS pull, no OpenAI cost); just query the existing index

param(
    [switch]$NoRefresh
)

$ErrorActionPreference = "Stop"

# Run from the repo root regardless of where the script is invoked from.
Set-Location -Path $PSScriptRoot

# Activate the project venv.
$activate = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    Write-Host "ERROR: venv not found at .venv\Scripts\Activate.ps1" -ForegroundColor Red
    Write-Host "Create it first:  python -m venv .venv  &&  pip install -r requirements.txt"
    exit 1
}
& $activate

# Phase 1 — refresh today's news + index (skippable).
if (-not $NoRefresh) {
    Write-Host "`n=== Refreshing news (fetch + index + briefing) ===" -ForegroundColor Cyan
    python main.py --mode run-once
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nRefresh failed (exit $LASTEXITCODE)." -ForegroundColor Yellow
        $ans = Read-Host "Open the app anyway on the existing index? [y/N]"
        if ($ans -ne "y") { exit $LASTEXITCODE }
    }
} else {
    Write-Host "`n=== Skipping refresh (-NoRefresh): querying existing index ===" -ForegroundColor Cyan
}

# Phase 2 — launch the Streamlit Q&A app (blocks until you close it with Ctrl+C).
Write-Host "`n=== Launching Streamlit (Ctrl+C to stop) ===" -ForegroundColor Cyan
streamlit run app.py
