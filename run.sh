#!/usr/bin/env bash
# ==============================================================================
# PixelArtSmith - Direct Launcher (GUI / CLI)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Setting up virtual environment..."
    PYTHON_CMD="python3"
    if command -v pyenv >/dev/null 2>&1; then
        PYENV_ROOT="$(pyenv root 2>/dev/null || echo "$HOME/.pyenv")"
        if [ -x "$PYENV_ROOT/versions/3.11.14/bin/python3" ]; then
            PYTHON_CMD="$PYENV_ROOT/versions/3.11.14/bin/python3"
        fi
    fi
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip --quiet
    pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
else
    source "$VENV_DIR/bin/activate"
fi

python "$SCRIPT_DIR/main.py" "$@"
