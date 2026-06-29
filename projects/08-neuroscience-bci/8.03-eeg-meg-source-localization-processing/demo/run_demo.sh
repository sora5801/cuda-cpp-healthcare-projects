#!/usr/bin/env bash
# ===========================================================================
# demo/run_demo.sh  --  One command: build (if needed) + run + verify (Linux)
# ---------------------------------------------------------------------------
# Project 8.3 -- EEG/MEG Source Localization & Processing   (template skeleton)
#
# Uses the OPTIONAL CMake build (the required deliverable is the VS solution).
# Mirrors run_demo.ps1: deterministic stdout is diffed against
# demo/expected_output.txt; stderr (timing) is shown but not diffed.
#
# Usage:  ./demo/run_demo.sh
# ===========================================================================
set -euo pipefail
DEMO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$DEMO")"
SLUG="eeg-meg-source-localization-processing"
SAMPLE="$ROOT/data/sample/saxpy_sample.txt"
EXPECTED="$DEMO/expected_output.txt"
BUILD="$ROOT/build/cmake"
EXE="$BUILD/$SLUG"

# --- 1. Build with CMake if the exe is missing ----------------------------
if [[ ! -x "$EXE" ]]; then
  echo "[run_demo] building $SLUG with CMake ..."
  cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$BUILD" --config Release >/dev/null
fi

# --- 2. Run, capturing stdout and stderr separately -----------------------
echo "[run_demo] running $SLUG on the committed sample ..."
STDOUT="$("$EXE" "$SAMPLE" 2>/tmp/${SLUG}_stderr.txt)" || true
STDERR="$(cat /tmp/${SLUG}_stderr.txt)"; rm -f /tmp/${SLUG}_stderr.txt

echo "---- program output (stdout) ----"; echo "$STDOUT"
echo "---- timing / detail (stderr) ----"; echo "$STDERR"
echo "----------------------------------"

# --- 3. Diff stdout vs expected (normalize trailing whitespace) -----------
if diff <(printf '%s\n' "$STDOUT" | sed 's/[[:space:]]*$//') \
        <(sed 's/[[:space:]]*$//' "$EXPECTED") >/dev/null; then
  echo "[run_demo] PASS: output matches expected_output.txt and GPU==CPU."
  exit 0
else
  echo "[run_demo] FAIL: output did not match expected_output.txt."
  exit 1
fi
