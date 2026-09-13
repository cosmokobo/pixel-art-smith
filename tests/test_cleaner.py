"""Tests for PixelCleaner: 8-connectivity orphan removal and transparency halo cleanup."""

import numpy as np
from PIL import Image

from pixel_art_smith.core.cleaner import PixelCleaner


def test_remove_orphan_pixels():
    # 10x10 transparent image
    arr = np.zeros((10, 10, 4), dtype=np.uint8)

    # 1. Connected diagonal line (should NOT be removed)
    arr[2, 2] = [255, 0, 0, 255]
    arr[3, 3] = [255, 0, 0, 255]

    # 2. Truly isolated 1x1 orphan pixel (SHOULD be removed)
    arr[7, 7] = [0, 255, 0, 255]

    img = Image.fromarray(arr, "RGBA")
    cleaned = PixelCleaner.remove_orphan_pixels(img)
    clean_arr = np.array(cleaned)

    # Connected diagonal pixels preserved
    assert clean_arr[2, 2, 3] == 255
    assert clean_arr[3, 3, 3] == 255

    # Orphan pixel cleaned to transparent
    assert clean_arr[7, 7, 3] == 0


def test_cleanup_transparency_halos():
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[0, 0] = [100, 100, 100, 50]  # Semi-transparent halo (< 128)
    arr[1, 1] = [100, 100, 100, 200]  # Solid pixel (>= 128)

    img = Image.fromarray(arr, "RGBA")
    cleaned = PixelCleaner.cleanup_transparency_halos(img)
    clean_arr = np.array(cleaned)

    assert clean_arr[0, 0, 3] == 0
    assert clean_arr[1, 1, 3] == 255
