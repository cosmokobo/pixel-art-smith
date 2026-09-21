"""Tests for CLI Runner: cell size parsing, frame export, and end-to-end processing pipeline."""

from PIL import Image

from pixel_art_smith.cli.runner import (
    export_frame_grid_by_motion,
    main_cli,
    parse_cell_size,
    process_single_image,
)


def test_parse_cell_size():
    assert parse_cell_size(None) is None
    assert parse_cell_size("auto") is None
    assert parse_cell_size("32") == (32, 32)
    assert parse_cell_size("32x48") == (32, 48)


def test_export_frame_grid_by_motion(tmp_path):
    std_grid = [
        [Image.new("RGBA", (16, 16), (255, 0, 0, 255)) for _ in range(4)]
        for _ in range(4)
    ]
    out_dir = tmp_path / "frames"
    exported = export_frame_grid_by_motion(std_grid, out_dir, stem="knight")

    # 16 frames x 4 outputs (subfolder, stem-tagged, direction-tagged, index-tagged) = 64 files
    assert len(exported) == 64
    assert (out_dir / "motion_00_down" / "frame_00.png").is_file()
    assert (out_dir / "knight_motion_00_down_frame_00.png").is_file()


def test_process_single_image_e2e(tmp_path, synthetic_upscaled_raw_sheet):
    raw_img_path = tmp_path / "hero.png"
    synthetic_upscaled_raw_sheet.save(raw_img_path)

    out_dir = tmp_path / "out"
    result = process_single_image(
        input_path=raw_img_path,
        output_dir=out_dir,
        scale=4,
        palette_name="snapper-16",
        max_colors=16,
        grid_mode="fixed-32",
        export_1x=True,
        export_frames=True,
        export_gifs=True,
    )

    assert result["success"] is True

    # Check 1x outputs
    assert (out_dir / "1x" / "hero_pixel_sheet.png").is_file()
    assert (out_dir / "1x" / "hero_metadata.json").is_file()
    assert (out_dir / "1x" / "hero_frames").is_dir()
    assert (out_dir / "1x" / "hero_gifs").is_dir()

    # Check 4x outputs
    assert (out_dir / "4x" / "hero_pixel_sheet.png").is_file()
    assert (out_dir / "4x" / "hero_metadata.json").is_file()
    assert (out_dir / "4x" / "hero_frames").is_dir()
    assert (out_dir / "4x" / "hero_gifs").is_dir()


def test_main_cli_help(capsys):
    import pytest

    with pytest.raises(SystemExit) as exc_info:
        main_cli(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "PixelArtSmith" in captured.out
    assert "Item Mode" in captured.out


def test_main_cli_dash_help(capsys):
    import pytest

    with pytest.raises(SystemExit) as exc_info:
        main_cli(["-help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "PixelArtSmith" in captured.out
    assert "Item Mode" in captured.out


def test_main_cli_short_h(capsys):
    import pytest

    with pytest.raises(SystemExit) as exc_info:
        main_cli(["-h"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "PixelArtSmith" in captured.out


def test_main_cli_agent_guide(capsys):
    import json

    ret = main_cli(["--agent-guide"])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["name"] == "PixelArtSmith"
    assert "character" in data["modes"]
    assert "item" in data["modes"]
    assert "--item" in data["flags"]


def test_process_single_image_item_mode(tmp_path):
    import numpy as np

    # Synthetic single item image: 128x128 with pitch 8 -> 16x16 grid
    # Foreground sword in center, white background (255, 255, 255)
    img_arr = np.full((128, 128, 3), 255, dtype=np.uint8)
    # Draw sword blade: rows 40..90, cols 60..68
    img_arr[40:90, 60:68] = [60, 120, 220]
    # Draw sword guard: rows 90..98, cols 48..80
    img_arr[90:98, 48:80] = [220, 180, 50]
    # Draw sword hilt: rows 98..112, cols 60..68
    img_arr[98:112, 60:68] = [120, 70, 30]

    item_img = Image.fromarray(img_arr, "RGB")
    item_path = tmp_path / "sword.png"
    item_img.save(item_path)

    out_dir = tmp_path / "out_item"
    result = process_single_image(
        input_path=item_path,
        output_dir=out_dir,
        scale=4,
        palette_name="snapper-16",
        max_colors=16,
        mode="item",
        export_sheet=False,
        export_frames=False,
        export_gifs=False,
        export_1x=True,
    )

    assert result["success"] is True
    assert result["mode"] == "item"
    assert result["rows"] == 1
    assert result["total_frames"] == 1
    assert "PASS" in result["audit_metric"].verdict

    # Check 1x deliverables: clean single PNG, no sheets/frames/gifs
    assert (out_dir / "1x" / "sword.png").is_file()
    assert (out_dir / "1x" / "sword_metadata.json").is_file()
    assert not (out_dir / "1x" / "sword_pixel_sheet.png").exists()
    assert not (out_dir / "1x" / "sword_frames").exists()
    assert not (out_dir / "1x" / "sword_gifs").exists()

    # Check 4x deliverables: clean single PNG, no sheets/frames/gifs
    assert (out_dir / "4x" / "sword.png").is_file()
    assert (out_dir / "4x" / "sword_metadata.json").is_file()
    assert not (out_dir / "4x" / "sword_pixel_sheet.png").exists()
    assert not (out_dir / "4x" / "sword_frames").exists()
    assert not (out_dir / "4x" / "sword_gifs").exists()


def test_main_cli_item_mode_e2e(tmp_path):
    import numpy as np

    img_arr = np.full((128, 128, 3), 255, dtype=np.uint8)
    img_arr[40:90, 60:68] = [60, 120, 220]
    img_arr[90:98, 48:80] = [220, 180, 50]

    item_img = Image.fromarray(img_arr, "RGB")
    item_path = tmp_path / "shield.png"
    item_img.save(item_path)

    out_dir = tmp_path / "out_shield"
    ret = main_cli([str(item_path), "-o", str(out_dir), "--item", "-s", "4"])
    assert ret == 0

    # Verify files created by CLI in item mode
    assert (out_dir / "1x" / "shield.png").is_file()
    assert (out_dir / "4x" / "shield.png").is_file()
    assert (out_dir / "1x" / "shield_metadata.json").is_file()
    assert (out_dir / "4x" / "shield_metadata.json").is_file()
    assert not (out_dir / "1x" / "shield_pixel_sheet.png").exists()
    assert not (out_dir / "1x" / "shield_gifs").exists()
    assert not (out_dir / "1x" / "shield_frames").exists()
    assert (out_dir / "result.md").is_file()

