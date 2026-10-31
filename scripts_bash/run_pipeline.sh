#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

usage () {
  cat << EOF
  Usage: $(basename "$0") [OVERRIDES...]

  Run the full NYC Taxi pipeline: ETL into PostgreSQL, then training and evaluation.

  Steps:
    1. Check the environment (.env file, CLI tools).
    2. ETL: load and clean the raw trips (nyc-taxi-etl).
    3. Training and evaluation on the test split (nyc-taxi-train).

  Arguments:
    OVERRIDES     Optional OmegaConf overrides passed to training, in key=value form.

  Options:
    -h, --help    Show this help and exit.

  Examples:
    $(basename "$0")
    $(basename "$0") training.max_epochs=20
    $(basename "$0") data.target=trip_duration_minutes data.target_unit=min

  Requirements:
    - A .env file in the repository root (see .env.example).
    - A running PostgreSQL instance, for example: docker compose up -d postgres
    - The package installed: pip install -e ".[dev]"
EOF
}

SKIP_DOWNLOAD=false
OVERRIDES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h| --help)
            usage
            exit 0
            ;;
        --skip-downloads)
        SKIP_DOWNLOAD=true
        shift
        ;;
    *=*)
      OVERRIDES+=("$1")
      shift
      ;;
    *)
    echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

log(){
  echo "[$(date +%H:%M:%S)] $*"
}



if [[ ! -f .env ]]; then
  log "ERROR: .env not found. Copy .env.example to .env and fill it in." >&2
  exit 1
fi

for cmd in nyc-taxi-etl nyc-taxi-train; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    log "ERROR: '$cmd' not found. Run: pip install -e \".[dev]\"" >&2
        exit 1
    fi
done

CURRENT_STEP="setup"

on_error() {
  log "ERROR: step '$CURRENT_STEP' failed at line $1" >&2
}
trap 'on_error $LINENO' ERR

START=$SECONDS

CURRENT_STEP="ETL"
if [[ "$SKIP_DOWNLOAD" == true ]]; then
    log "ETL skipped (--skip-etl)"
else
    log "ETL started"
    nyc-taxi-etl
fi

CURRENT_STEP="training"
log "Training started with overrides: ${OVERRIDES[*]:-none}"
nyc-taxi-train "${OVERRIDES[@]}"

ELAPSED=$((SECONDS - START))
log "Pipeline finished in ${ELAPSED}s."
log "Checkpoints and metrics: see the checkpoint directory in configs/train.yaml."
