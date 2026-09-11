#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -n "${TRANSCRIBE_TEST_PYTHON:-}" ]]; then
    TEST_PYTHON="$TRANSCRIBE_TEST_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    TEST_PYTHON="$ROOT/.venv/bin/python"
else
    TEST_PYTHON=python3
fi

bash -n install.sh transcribe-media scripts/gx10-runtime.sh scripts/check.sh
shellcheck -x install.sh transcribe-media scripts/gx10-runtime.sh scripts/check.sh
"$TEST_PYTHON" -m compileall -q transcribe_media.py transcribe_media_app tests
"$TEST_PYTHON" -m unittest discover -s tests -v
