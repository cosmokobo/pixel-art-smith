#!/usr/bin/env bash
# ==============================================================================
# PixelArtSmith - Standalone Binary Builder (PyInstaller)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [ "$REPO_ROOT" != "/" ] && [ ! -f "$REPO_ROOT/AGENTS.md" ] && [ ! -d "$REPO_ROOT/.git" ] && [ ! -f "$REPO_ROOT/.git" ]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done

if [ -f "$REPO_ROOT/AGENTS.md" ] || [ -f "$REPO_ROOT/.gitmodules" ]; then
    BUILD_ROOT="$REPO_ROOT/build/pixel-art-smith"
else
    BUILD_ROOT="$SCRIPT_DIR/dist"
fi

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

log_info "Building PixelArtSmith -> $BUILD_ROOT"

VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_CMD="python3"

# Find python 3.10/3.11 if available via pyenv
if command -v pyenv >/dev/null 2>&1; then
    PYENV_ROOT="$(pyenv root 2>/dev/null || echo "$HOME/.pyenv")"
    if [ -x "$PYENV_ROOT/versions/3.11.14/bin/python3" ]; then
        PYTHON_CMD="$PYENV_ROOT/versions/3.11.14/bin/python3"
    elif [ -x "$PYENV_ROOT/versions/3.10.13/bin/python3" ]; then
        PYTHON_CMD="$PYENV_ROOT/versions/3.10.13/bin/python3"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    log_info "Creating virtual environment at $VENV_DIR using $PYTHON_CMD..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

log_info "Installing/verifying dependencies..."
pip install --upgrade pip --quiet
pip install -r "$SCRIPT_DIR/requirements.txt" --quiet

mkdir -p "$BUILD_ROOT"

log_info "Running PyInstaller..."
pyinstaller \
    --name "pixel-art-smith" \
    --onefile \
    --clean \
    --noconfirm \
    --distpath "$BUILD_ROOT" \
    --workpath "$SCRIPT_DIR/build_temp" \
    --specpath "$SCRIPT_DIR" \
    --copy-metadata "pymatting" \
    --copy-metadata "rembg" \
    --copy-metadata "onnxruntime" \
    --copy-metadata "tqdm" \
    --copy-metadata "jsonschema" \
    --hidden-import "pixel_art_smith" \
    --hidden-import "pixel_art_smith.core" \
    --hidden-import "pixel_art_smith.cli" \
    --hidden-import "pixel_art_smith.gui" \
    --hidden-import "PIL" \
    --hidden-import "PIL.Image" \
    --hidden-import "cv2" \
    --hidden-import "numpy" \
    --hidden-import "sklearn" \
    --hidden-import "rembg" \
    --hidden-import "customtkinter" \
    "$SCRIPT_DIR/main.py"

# Clean temporary build files
rm -rf "$SCRIPT_DIR/build_temp" "$SCRIPT_DIR/pixel-art-smith.spec"

log_success "PixelArtSmith built successfully to: $BUILD_ROOT/pixel-art-smith"
