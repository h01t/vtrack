#!/bin/bash
# Compatibility wrapper for the unified vtrack train-remote CLI.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

exec "$ROOT_DIR/.venv/bin/vtrack" train-remote "$@"
