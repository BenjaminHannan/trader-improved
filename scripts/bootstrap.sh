#!/usr/bin/env bash
# One-command setup + first data pull (macOS / Linux). Windows: scripts/bootstrap.ps1.
#
#   FRED_API_KEY=yourkey bash scripts/bootstrap.sh [start-date]
#
# Re-running is safe: sync and ingestion are idempotent, ingestion is incremental.
set -euo pipefail
cd "$(dirname "$0")/.."
START="${1:-2016-01-01}"

if ! command -v uv >/dev/null 2>&1; then
    echo "=== Installing uv ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ -z "${FRED_API_KEY:-}" ]; then
    echo "NOTE: FRED_API_KEY not set - macro falls back to no-vintage CSV with an audit warning."
fi

echo "=== uv sync ===";            uv sync
echo "=== test suite ===";         uv run pytest -q
echo "=== universe ===";           uv run python scripts/ingest.py --dataset universe
echo "=== Stage-1 ingest ===";     uv run python scripts/ingest.py --dataset all --start "$START" || \
    echo "Some datasets reported errors - re-run; ingestion is incremental and idempotent."
echo "=== factor IC report ===";   uv run python scripts/build_factors.py --ic-report

cat <<'EOF'

Bootstrap complete. Next:
  uv run python scripts/build_factors.py --ic-report --apply
  uv run python scripts/run_backtest.py
EOF
