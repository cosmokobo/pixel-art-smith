"""Tests for BackgroundRemover: 4-connected perimeter floodfill and cavity preservation."""

import numpy as np
from PIL import Image

from pixel_art_smith.core.bg_remover import BackgroundRemover


def test_segment_background_outer_floodfill():
    # 20x20 image with white background (255, 255, 255)
    arr = np.full((20, 20, 3), 255, dtype=np.uint8)

    # Red character in middle (x: 5..15, y: 5..15)
    arr[5:15, 5:15] = [200, 30, 30]

    # White shirt inside character interior (x: 8..12, y: 8..12)
    # This internal white MUST NOT be eaten by outer floodfill!
    arr[8:12, 8:12] = [255, 255, 255]

    bg_mask, fg_mask, _resolved = BackgroundRemover.segment_background_with_cavity_resolution(
        arr, resolve_cavities=False
    )

    # Border must be detected as background
    assert bg_mask[0, 0] is True or bg_mask[0, 0] == 1
    assert bg_mask[19, 19] is True or bg_mask[19, 19] == 1

    # Red body must be foreground
    assert fg_mask[6, 6] is True or fg_mask[6, 6] == 1

    # Internal white shirt must be PRESERVED as foreground!
    assert fg_mask[10, 10] is True or fg_mask[10, 10] == 1


def test_remove_background_quantized():
    arr = np.full((16, 16, 3), 255, dtype=np.uint8)
    arr[4:12, 4:12] = [0, 150, 255]

    img = Image.fromarray(arr, "RGB")
    rgba_out = BackgroundRemover.remove_background_quantized(img)

    assert rgba_out.mode == "RGBA"
    out_arr = np.array(rgba_out)

    # Background transparent
    assert out_arr[0, 0, 3] == 0
    # Character opaque
    assert out_arr[8, 8, 3] == 255


def test_segment_background_enclosed_cavity_ring():
    # 32x32 image with white background
    arr = np.full((32, 32, 3), 255, dtype=np.uint8)

    # Golden ring (outer radius 12, inner radius 6) centered at (16, 16)
    for y in range(32):
        for x in range(32):
            dist_sq = (y - 16) ** 2 + (x - 16) ** 2
            if 36 <= dist_sq <= 144:  # ring band
                arr[y, x] = [210, 170, 40]

    # Without cavity resolution: center is treated as foreground
    bg_mask_no, fg_mask_no, res_no = BackgroundRemover.segment_background_with_cavity_resolution(
        arr, resolve_cavities=False
    )
    assert res_no == 0
    assert fg_mask_no[16, 16] is True or fg_mask_no[16, 16] == 1

    # With cavity resolution: center is resolved as background
    bg_mask_yes, fg_mask_yes, res_yes = BackgroundRemover.segment_background_with_cavity_resolution(
        arr, resolve_cavities=True, max_cavity_area=None
    )
    assert res_yes > 0
    assert bg_mask_yes[16, 16] is True or bg_mask_yes[16, 16] == 1
    assert fg_mask_yes[16, 16] is False or fg_mask_yes[16, 16] == 0
    # Ring band remains foreground
    assert fg_mask_yes[16, 26] is True or fg_mask_yes[16, 26] == 1


def test_segment_background_large_cavity_unconstrained():
    # 64x64 image with large cavity (area > 100)
    arr = np.full((64, 64, 3), 255, dtype=np.uint8)
    arr[10:54, 10:54] = [50, 150, 200]  # outer box
    arr[20:44, 20:44] = [255, 255, 255]  # inner cavity (24x24 = 576 pixels)

    # When max_cavity_area=40, cavity is rejected
    _, _, res_capped = BackgroundRemover.segment_background_with_cavity_resolution(
        arr, resolve_cavities=True, max_cavity_area=40
    )
    assert res_capped == 0

    # When max_cavity_area=None, cavity is resolved
    bg_mask, fg_mask, res_unconstrained = BackgroundRemover.segment_background_with_cavity_resolution(
        arr, resolve_cavities=True, max_cavity_area=None
    )
    assert res_unconstrained == 576
    assert bg_mask[32, 32] is True or bg_mask[32, 32] == 1
