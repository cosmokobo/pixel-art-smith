"""Tests for SpriteIsolator: layout detection, boundary cut penalty, and frame slicing."""

from PIL import Image

from pixel_art_smith.core.sprite_isolator import SpriteIsolator


def test_detect_matrix_layout_4x4(synthetic_4x4_sprite_sheet):
    mode, rows, cols = SpriteIsolator.detect_matrix_layout(synthetic_4x4_sprite_sheet)
    assert mode == "sheet"
    assert rows == 4
    assert cols == 4


def test_detect_matrix_layout_single_canvas(synthetic_sprite_frame):
    # Only 1 character in the top-left of a 128x128 canvas
    canvas = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    canvas.paste(synthetic_sprite_frame, (10, 10))

    mode, rows, cols = SpriteIsolator.detect_matrix_layout(canvas)
    assert mode == "canvas"
    assert rows == 1
    assert cols == 1


def test_isolate_grid_cells(synthetic_4x4_sprite_sheet):
    isolator = SpriteIsolator(min_area=8, padding=1)
    matrix = isolator.isolate_grid_cells(synthetic_4x4_sprite_sheet, n_rows=4, n_cols=4)

    assert len(matrix) == 4
    for row in matrix:
        assert len(row) == 4
        for item in row:
            assert item.image.width > 0
            assert item.image.height > 0
            assert item.bbox[2] > 0
            assert item.bbox[3] > 0
