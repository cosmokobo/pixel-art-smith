"""Tests for PaletteQuantizer, hex conversions, and CIELAB color matching."""

import numpy as np
from PIL import Image

from pixel_art_smith.core.palette import (
    PALETTES,
    PaletteQuantizer,
    hex_to_rgb,
    rgb_to_hex,
)


def test_hex_rgb_conversions():
    assert hex_to_rgb("#ff0000") == (255, 0, 0)
    assert hex_to_rgb("00ff00") == (0, 255, 0)
    assert rgb_to_hex((0, 0, 255)) == "#0000ff"


def test_palette_quantizer_basic():
    # Setup custom 3-color palette (pure red, pure green, pure blue)
    quantizer = PaletteQuantizer(custom_colors=["#ff0000", "#00ff00", "#0000ff"])

    # Create image with near-red pixel
    arr = np.zeros((2, 2, 4), dtype=np.uint8)
    arr[0, 0] = [250, 10, 10, 255]  # Near red
    arr[0, 1] = [10, 245, 10, 255]  # Near green
    arr[1, 0] = [0, 0, 0, 0]  # Transparent

    img = Image.fromarray(arr, "RGBA")
    quantized = quantizer.quantize(img)
    q_arr = np.array(quantized)

    # Near red snapped to pure red
    assert list(q_arr[0, 0, :3]) == [255, 0, 0]
    # Near green snapped to pure green
    assert list(q_arr[0, 1, :3]) == [0, 255, 0]
    # Alpha remains 0
    assert q_arr[1, 0, 3] == 0


def test_snapper_and_curated_palettes():
    assert "sweetie-16" in PALETTES
    assert len(PALETTES["sweetie-16"]) == 16

    quantizer = PaletteQuantizer(palette_name="sweetie-16")
    colors = quantizer.get_colors_hex()
    assert len(colors) == 16
