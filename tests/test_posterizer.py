"""Tests for PixelPosterizer: KMeans semantic palette extraction and chroma-weighted quantization."""

import numpy as np
from PIL import Image

from pixel_art_smith.core.posterizer import PixelPosterizer


def test_extract_semantic_palette():
    # 32x32 character image
    arr = np.zeros((32, 32, 4), dtype=np.uint8)
    arr[5:25, 5:25] = [180, 40, 40, 255]

    img = Image.fromarray(arr, "RGBA")
    palette = PixelPosterizer.extract_semantic_palette(img, max_colors=8)

    assert isinstance(palette, np.ndarray)
    assert len(palette) <= 8
    # 0 is dedicated black outline, 1 is white
    assert np.all(palette[0] == [0, 0, 0])
    assert np.all(palette[1] == [236, 236, 236])


def test_process_snapper_pipeline():
    arr = np.zeros((20, 20, 3), dtype=np.uint8)
    arr[:10, :] = [240, 50, 50]
    arr[10:, :] = [50, 240, 50]

    img = Image.fromarray(arr, "RGB")
    quant_img, palette_hex = PixelPosterizer.process_snapper_pipeline(img, max_colors=16)

    assert quant_img.size == (20, 20)
    assert len(palette_hex) <= 16
    assert any(h.startswith("#") for h in palette_hex)
