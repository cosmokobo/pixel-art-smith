"""Test fixtures and synthetic image generators for PixelArtSmith QA."""

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def synthetic_sprite_frame():
    """Generate a single 32x32 RGBA sprite character frame with transparent background."""
    arr = np.zeros((32, 32, 4), dtype=np.uint8)

    # Character body in center (x: 10..22, y: 8..28)
    arr[8:28, 10:22] = [200, 50, 50, 255]  # Red armor
    arr[12:18, 12:20] = [255, 220, 180, 255]  # Face skin
    arr[14:16, 14:15] = [20, 20, 20, 255]  # Eye left
    arr[14:16, 17:18] = [20, 20, 20, 255]  # Eye right
    arr[6:12, 10:22] = [255, 215, 0, 255]  # Blonde hair

    # Black outlines
    arr[7:29, 9] = [10, 10, 10, 255]
    arr[7:29, 22] = [10, 10, 10, 255]
    arr[7, 9:23] = [10, 10, 10, 255]
    arr[28, 9:23] = [10, 10, 10, 255]

    return Image.fromarray(arr, "RGBA")


@pytest.fixture
def synthetic_4x4_sprite_sheet(synthetic_sprite_frame):
    """Generate a 4x4 matrix sprite sheet (128x128) with 16 character frames."""
    cell_w, cell_h = 32, 32
    sheet = Image.new("RGBA", (cell_w * 4, cell_h * 4), (0, 0, 0, 0))
    frame = synthetic_sprite_frame

    for r in range(4):
        for c in range(4):
            # Introduce slight motion offset per frame
            offset_x = (c % 2) * 2
            sheet.paste(frame, (c * cell_w + offset_x, r * cell_h), frame)

    return sheet


@pytest.fixture
def synthetic_upscaled_raw_sheet(synthetic_4x4_sprite_sheet):
    """Generate an upscaled raw sprite sheet (e.g. 8x pixel pitch, 1024x1024) with white background."""
    pitch = 8
    w, h = synthetic_4x4_sprite_sheet.size
    upscaled = synthetic_4x4_sprite_sheet.resize((w * pitch, h * pitch), resample=Image.Resampling.NEAREST)

    # Convert transparent areas to solid white background like raw Stable Diffusion output
    bg = Image.new("RGB", upscaled.size, (255, 255, 255))
    bg.paste(upscaled, mask=upscaled.getchannel("A"))
    return bg
