"""Tests for GridDetector: pitch estimation, downsampling, and nearest-neighbor upscaling."""

import numpy as np
from PIL import Image

from pixel_art_smith.core.grid_detector import GridDetector


def test_estimate_pixel_pitch():
    # Create an 8x8 block checkerboard pattern (128x128 total)
    arr = np.zeros((128, 128), dtype=np.uint8)
    for y in range(0, 128, 16):
        for x in range(0, 128, 16):
            arr[y : y + 8, x : x + 8] = 255
            arr[y + 8 : y + 16, x + 8 : x + 16] = 255

    img = Image.fromarray(arr, "L")
    pitch = GridDetector.estimate_pixel_pitch(img, min_pitch=4, max_pitch=16)
    assert pitch in (4, 8)


def test_core_subblock_downsample_rgb():
    # 64x64 image with 8px blocks -> 8x8 result
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:32, :32] = [255, 0, 0]  # Red block
    arr[32:, 32:] = [0, 255, 0]  # Green block

    img = Image.fromarray(arr, "RGB")
    downsampled = GridDetector.core_subblock_downsample(img, pitch=8)

    assert downsampled.size == (8, 8)
    down_arr = np.array(downsampled)
    # Check top-left is red
    assert np.all(down_arr[1, 1] == [255, 0, 0])
    # Check bottom-right is green
    assert np.all(down_arr[6, 6] == [0, 255, 0])


def test_core_subblock_downsample_rgba():
    arr = np.zeros((32, 32, 4), dtype=np.uint8)
    arr[:16, :16] = [0, 0, 255, 255]  # Blue opaque

    img = Image.fromarray(arr, "RGBA")
    downsampled = GridDetector.core_subblock_downsample(img, pitch=8)

    assert downsampled.size == (4, 4)
    down_arr = np.array(downsampled)
    assert down_arr[0, 0, 3] == 255
    assert down_arr[3, 3, 3] == 0


def test_upscale_nearest():
    img = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
    scaled_4x = GridDetector.upscale_nearest(img, scale=4)
    assert scaled_4x.size == (64, 64)

    # 1x scale returns exact same
    scaled_1x = GridDetector.upscale_nearest(img, scale=1)
    assert scaled_1x.size == (16, 16)
