#!/usr/bin/env bash
# ===========================================================================
# demo/run_demo.sh  --  One command: build (if needed) + run + verify (Linux)
# Project 12.01 : Mass-Spectrometry Proteomics Search   (CMake path)
# ===========================================================================
set -euo pipefail
DEMO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$DEMO")"
SLUG="mass-spectrometry-proteomics-search"
SAMPLE="$ROOT/data/sample/spectra_sample.txt"
EXPECTED="$DEMO/expected_output.txt"
BUILD="$ROOT/build/cmake"
EXE="$BUILD/$SLUG"

if [[ ! -x "$EXE" ]]; then
  echo "[run_demo] building $SLUG with CMake ..."
  cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$BUILD" --config Release >/dev/null
fi

echo "[run_demo] running $SLUG on the committed sample ..."
STDOUT="$("$EXE" "$SAMPLE" 2>/tmp/${SLUG}_stderr.txt)" || true
STDERR="$(cat /tmp/${SLUG}_stderr.txt)"; rm -f /tmp/${SLUG}_stderr.txt

echo "---- program output (stdout) ----"; echo "$STDOUT"
echo "---- timing / detail (stderr) ----"; echo "$STDERR"
echo "----------------------------------"

if diff <(printf '%s\n' "$STDOUT" | sed 's/[[:space:]]*$//') \
        <(sed 's/[[:space:]]*$//' "$EXPECTED") >/dev/null; then
  echo "[run_demo] PASS: output matches expected_output.txt and GPU==CPU."
  exit 0
else
  echo "[run_demo] FAIL: output did not match expected_output.txt."
  exit 1
fi
