#!/usr/bin/env bash
set -euo pipefail

# Prefer explicit 3.12 on Apple Silicon, then 3.11+, then python3 fallback.
pick_python() {
  for cand in \
    /opt/homebrew/bin/python3.12 \
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
    /opt/homebrew/bin/python3.11 \
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
    python3; do
    if command -v "$cand" >/dev/null 2>&1; then
      echo "$cand"
      return 0
    fi
  done
  echo "❌ Python 3.11+ not found. Install one from python.org or Homebrew." >&2
  exit 1
}

# Force ARM64 shell (no Rosetta)
if [ "$(uname -m)" != "arm64" ]; then
  exec arch -arm64 zsh -lc "$0"
fi

PY="$(pick_python)"

# Create venv if missing using the chosen interpreter
if [ ! -d ".venv" ]; then
  echo "🔧 Creating virtual environment with: $PY"
  "$PY" -m venv .venv
fi

# Activate venv
# shellcheck disable=SC1091
source .venv/bin/activate

# Verify version >= 3.11 and arch = arm64
python - <<'PYCHK'
import sys, platform
maj, min = sys.version_info[:2]
assert (maj, min) >= (3, 11), f"Need Python 3.11+, found {sys.version.split()[0]}"
assert platform.machine() == "arm64", f"Need arm64 shell, found {platform.machine()}"
PYCHK

# Install deps if requirements.txt exists
if [ -f requirements.txt ]; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
fi

echo "🚀 Launching Streamlit with $(python -V) @ $(python -c 'import sys;print(sys.executable)')"
exec python -m streamlit run app.py
