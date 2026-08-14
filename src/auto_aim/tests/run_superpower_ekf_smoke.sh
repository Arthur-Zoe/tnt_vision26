#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EIGEN_DIR="${EIGEN3_INCLUDE_DIR:-/usr/include/eigen3}"
if [[ ! -f "$EIGEN_DIR/Eigen/Dense" ]]; then
  echo "Eigen not found at $EIGEN_DIR; set EIGEN3_INCLUDE_DIR." >&2
  exit 2
fi
OUT="${TMPDIR:-/tmp}/superpower_ekf_smoke"
g++ -std=c++17 -O0 -Wall -Wextra -Wpedantic -Werror \
  -I"$ROOT/include" -I"$EIGEN_DIR" \
  "$ROOT/src/EKF/SuperPowerEKF.cpp" \
  "$ROOT/src/EKF/SuperPowerTarget.cpp" \
  "$ROOT/src/EKF/SuperPowerTracker.cpp" \
  "$ROOT/tests/superpower_ekf_smoke.cpp" \
  -o "$OUT"
"$OUT"
