from pathlib import Path

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
    assert "--no-gifs" in captured.out
    assert "--static" in captured.out


def test_main_cli_dash_help(capsys):
    import pytest

    with pytest.raises(SystemExit) as exc_info:
        main_cli(["-help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "PixelArtSmith" in captured.out
    assert "--no-gifs" in captured.out


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
    assert "multi_motion_sheet" in data["asset_types"]
    assert "single_frame_static" in data["asset_types"]
    assert "--no-gifs / --export-gifs" in data["flags"]
    assert "--static" in data["flags"]


def test_process_single_image_static_asset(tmp_path):
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
        grid_mode="canvas",
        export_sheet=False,
        export_frames=False,
        export_gifs=False,
        export_1x=True,
    )

    assert result["success"] is True
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


def test_main_cli_static_e2e(tmp_path):
    import numpy as np

    img_arr = np.full((128, 128, 3), 255, dtype=np.uint8)
    img_arr[40:90, 60:68] = [60, 120, 220]
    img_arr[90:98, 48:80] = [220, 180, 50]

    item_img = Image.fromarray(img_arr, "RGB")
    item_path = tmp_path / "shield.png"
    item_img.save(item_path)

    out_dir = tmp_path / "out_shield"
    ret = main_cli([str(item_path), "-o", str(out_dir), "--static", "-s", "4"])
    assert ret == 0

    # Verify files created by CLI in static mode
    assert (out_dir / "1x" / "shield.png").is_file()
    assert (out_dir / "4x" / "shield.png").is_file()
    assert (out_dir / "1x" / "shield_metadata.json").is_file()
    assert (out_dir / "4x" / "shield_metadata.json").is_file()
    assert not (out_dir / "1x" / "shield_pixel_sheet.png").exists()
    assert not (out_dir / "1x" / "shield_gifs").exists()
    assert not (out_dir / "1x" / "shield_frames").exists()
    assert (out_dir / "result.md").is_file()


def test_main_cli_granular_flags_e2e(tmp_path):
    import numpy as np

    img_arr = np.full((128, 128, 3), 255, dtype=np.uint8)
    img_arr[40:90, 60:68] = [60, 120, 220]
    img_arr[90:98, 48:80] = [220, 180, 50]

    item_img = Image.fromarray(img_arr, "RGB")
    item_path = tmp_path / "potion.png"
    item_img.save(item_path)

    out_dir = tmp_path / "out_potion"
    ret = main_cli([str(item_path), "-o", str(out_dir), "--no-gifs", "--no-frames", "--no-sheet", "-s", "4"])
    assert ret == 0

    # Verify deliverable flags suppress gifs, frames, and sheet suffix
    assert (out_dir / "1x" / "potion.png").is_file()
    assert (out_dir / "4x" / "potion.png").is_file()
    assert not (out_dir / "1x" / "potion_pixel_sheet.png").exists()
    assert not (out_dir / "1x" / "potion_gifs").exists()
    assert not (out_dir / "1x" / "potion_frames").exists()
    assert (out_dir / "result.md").is_file()


def test_main_cli_cavity_flags_e2e(tmp_path):
    import numpy as np

    # 32x32 hollow ring image
    img_arr = np.full((32, 32, 3), 255, dtype=np.uint8)
    for y in range(32):
        for x in range(32):
            dist_sq = (y - 16) ** 2 + (x - 16) ** 2
            if 36 <= dist_sq <= 144:
                img_arr[y, x] = [210, 170, 40]

    ring_img = Image.fromarray(img_arr, "RGB")
    ring_path = tmp_path / "ring.png"
    ring_img.save(ring_path)

    # 1. Default static mode -> auto-resolves cavities
    out_dir_default = tmp_path / "out_default"
    ret = main_cli([str(ring_path), "-o", str(out_dir_default), "--static", "-P", "1", "-s", "4"])
    assert ret == 0
    res_img = Image.open(out_dir_default / "1x" / "ring.png")
    res_arr = np.array(res_img)
    # Center hole must be transparent
    assert res_arr[16, 16, 3] == 0

    # 2. Explicit --no-cavities -> cavity remains opaque
    out_dir_no = tmp_path / "out_no_cavities"
    ret = main_cli([str(ring_path), "-o", str(out_dir_no), "--static", "--no-cavities", "-P", "1", "-s", "4"])
    assert ret == 0
    res_no_img = Image.open(out_dir_no / "1x" / "ring.png")
    res_no_arr = np.array(res_no_img)
    # Center hole must be opaque
    assert res_no_arr[16, 16, 3] == 255


def test_real_test_assets_cavity_transparency(tmp_path):
    import cv2
    import numpy as np

    test_assets_dir = Path("/Users/kojeomstudio/pixel-art-smith-test")
    if not test_assets_dir.exists():
        return

    out_dir = tmp_path / "out_real_test_assets"
    ret = main_cli([str(test_assets_dir), "-o", str(out_dir), "--static", "-P", "0", "-s", "4"])
    assert ret == 0

    expected_assets = [
        "accessory_16_jade_bangle.png",
        "accessory_17_charm_bracelet.png",
        "accessory_18_crystal_pendant.png",
        "accessory_20_diamond_ring.png",
        "weapon_08_crossbow.png",
    ]

    for asset_name in expected_assets:
        out_file = out_dir / "1x" / asset_name
        assert out_file.is_file(), f"Missing output for {asset_name}"
        img = Image.open(out_file)
        arr = np.array(img)
        alpha = arr[:, :, 3]
        num_labels, _, _, _ = cv2.connectedComponentsWithStats((alpha > 0).astype(np.uint8) * 255, connectivity=8)
        # Foreground must be a single connected component
        assert num_labels - 1 == 1, f"{asset_name} foreground should be single connected component, got {num_labels - 1}"
        # Total transparent pixels must include both perimeter and internal cavity
        assert np.sum(alpha == 0) > 600, f"{asset_name} should have transparent pixels including cavity"


