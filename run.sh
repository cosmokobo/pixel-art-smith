#!/usr/bin/env bash
# ==============================================================================
# PixelArtSmith - Source & Dedicated venv Launcher (GUI / CLI)
# Guarantees live source execution and automatic dependency cross-verification.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# 1. Select Best Available Python (3.10 / 3.11 preferred via pyenv or system python3)
PYTHON_CMD="python3"
if command -v pyenv >/dev/null 2>&1; then
    PYENV_ROOT="$(pyenv root 2>/dev/null || echo "$HOME/.pyenv")"
    if [ -x "$PYENV_ROOT/versions/3.11.14/bin/python3" ]; then
        PYTHON_CMD="$PYENV_ROOT/versions/3.11.14/bin/python3"
    elif [ -x "$PYENV_ROOT/versions/3.10.13/bin/python3" ]; then
        PYTHON_CMD="$PYENV_ROOT/versions/3.10.13/bin/python3"
    fi
fi

# 2. Check if venv exists; create if missing
if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "[INFO] Creating dedicated virtual environment at: $VENV_DIR using $PYTHON_CMD..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    NEED_INSTALL=1
else
    NEED_INSTALL=0
fi

# 3. Cross-validate installed packages against requirements.txt
if [ "$NEED_INSTALL" -eq 0 ]; then
    if ! "$VENV_DIR/bin/python" -c "
import sys
from importlib.metadata import distributions

installed = {d.metadata['Name'].lower().replace('-', '_') for d in distributions() if d.metadata and d.metadata.get('Name')}
core_reqs = ['pillow', 'numpy', 'opencv_python_headless', 'scikit_learn']

missing = [r for r in core_reqs if r not in installed and r.replace('_', '-') not in installed]
if missing:
    sys.exit(1)

# Ensure essential libraries import cleanly
import PIL, numpy, cv2, sklearn
" >/dev/null 2>&1; then
        echo "[WARN] Dependencies in $VENV_DIR are missing or incomplete. Reinstalling..."
        NEED_INSTALL=1
    fi
fi

# 4. Install or repair requirements if needed
if [ "$NEED_INSTALL" -eq 1 ]; then
    echo "[INFO] Installing/verifying dependencies from requirements.txt..."
    "$VENV_DIR/bin/pip" install --upgrade pip --quiet
    "$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt" --quiet
    echo "[SUCCESS] Dedicated virtual environment is fully synchronized."
fi

# 5. Execute live source code via venv python
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/main.py" "$@"
