"""Tests for SpritePacker: ground alignment, optimal cell resolution, and sheet packing."""

from PIL import Image

from pixel_art_smith.core.packer import SpritePacker
from pixel_art_smith.core.sprite_isolator import FrameItem


def test_standardize_frame():
    # 10x20 sprite placed in 32x32 cell
    sprite = Image.new("RGBA", (10, 20), (255, 0, 0, 255))
    standardized = SpritePacker.standardize_frame(sprite, cell_size=(32, 32), bottom_margin=1)

    assert standardized.size == (32, 32)
    # Character centered horizontally: offset_x = (32 - 10) // 2 = 11
    # Bottom aligned: offset_y = 32 - 20 - 1 = 11
    pixel = standardized.getpixel((11, 11))
    assert isinstance(pixel, tuple) and pixel[3] == 255


def test_resolve_cell_size():
    dummy_item = FrameItem(Image.new("RGBA", (20, 28)), (0, 0, 20, 28))
    matrix = [[dummy_item]]

    assert SpritePacker.resolve_cell_size(matrix, "fixed-32") == (32, 32)
    assert SpritePacker.resolve_cell_size(matrix, "fixed-48") == (48, 48)
    assert SpritePacker.resolve_cell_size(matrix, "fixed-24x32") == (24, 32)


def test_pack_matrix_sheet():
    items = []
    for r in range(4):
        row = []
        for c in range(4):
            img = Image.new("RGBA", (16, 24), (100 + r * 20, 50 + c * 20, 200, 255))
            row.append(FrameItem(img, (c * 32, r * 32, 16, 24), row=r, col=c))
        items.append(row)

    sheet, meta, grid = SpritePacker.pack_matrix_sheet(
        matrix=items,
        cell_size=(32, 32),
        scale=1,
        grid_mode="fixed-32",
    )

    assert sheet.size == (128, 128)
    assert meta["sprite_sheet"]["width"] == 128
    assert meta["sprite_sheet"]["height"] == 128
    assert meta["sprite_sheet"]["grid_layout"]["rows"] == 4
    assert meta["sprite_sheet"]["grid_layout"]["columns"] == 4
    assert len(grid) == 4
    assert len(grid[0]) == 4
