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
