"""Tests for QualityAuditor: metrics extraction and Markdown audit report generation."""

from PIL import Image

from pixel_art_smith.core.auditor import QualityAuditor


def test_audit_single_and_generate_markdown(tmp_path):
    raw_img = Image.new("RGB", (128, 128), (255, 255, 255))
    sheet_img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    # Add sufficient opaque pixels (> 1000) to represent valid sprite content
    for y in range(20, 60):
        for x in range(20, 60):
            sheet_img.putpixel((x, y), (200, 50, 50, 255))

    meta = {
        "schema_version": "2.0",
        "sprite_sheet": {
            "format": "RGBA8888",
            "width": 128,
            "height": 128,
            "grid_layout": {
                "grid_mode": "fixed-32",
                "rows": 4,
                "columns": 4,
                "cell_size": {"width": 32, "height": 32},
                "logical_cell_size": {"width": 32, "height": 32},
                "scale_factor": 1,
            },
            "total_frames": 16,
        },
        "palette": {"name": "snapper-16", "color_count": 16},
    }

    metric = QualityAuditor.audit_single(
        src_img=raw_img,
        sheet_img=sheet_img,
        metadata=meta,
        name="test_char",
    )

    assert metric.name == "test_char"
    assert "PASS" in metric.verdict

    report_path = QualityAuditor.generate_markdown_report(
        metrics=[metric],
        output_dir=tmp_path,
        report_name="result.md",
    )

    assert report_path.is_file()
    content = report_path.read_text(encoding="utf-8")
    assert "PixelArtSmith: Sprite Sheet Quality & Verification Audit Report" in content
    assert "test_char" in content
    assert "Pass Rate" in content


def test_generate_folder_readmes(tmp_path):
    # Setup dummy directory structure
    dir_1x = tmp_path / "1x"
    dir_4x = tmp_path / "4x"
    dir_native = tmp_path / "native"
    dir_1x.mkdir()
    dir_4x.mkdir()
    dir_native.mkdir()

    raw_img = Image.new("RGB", (64, 64), (255, 255, 255))
    sheet_1x = Image.new("RGBA", (64, 64), (100, 100, 100, 255))

    dummy_src = dir_native / "fantasy_sword.png"
    raw_img.save(dummy_src)

    meta = {
        "schema_version": "2.0",
        "sprite_sheet": {
            "format": "RGBA8888",
            "width": 64,
            "height": 64,
            "grid_layout": {
                "grid_mode": "fixed-64",
                "rows": 1,
                "columns": 1,
                "cell_size": {"width": 64, "height": 64},
                "logical_cell_size": {"width": 64, "height": 64},
                "scale_factor": 1,
            },
            "total_frames": 1,
        },
        "palette": {"name": "snapper-16", "color_count": 16},
    }

    metric = QualityAuditor.audit_single(
        src_img=raw_img,
        sheet_img=sheet_1x,
        metadata=meta,
        name="fantasy_sword",
    )

    # Test 1x readme
    p_1x = QualityAuditor.generate_folder_readme_1x(tmp_path, [metric])
    assert p_1x.is_file()
    assert (dir_1x / "readme.md").is_file()
    content_1x = p_1x.read_text(encoding="utf-8")
    assert "1x Native Pixel Art Assets" in content_1x
    assert "fantasy_sword.png" in content_1x

    # Test 4x readme
    p_4x = QualityAuditor.generate_folder_readme_4x(tmp_path, [metric])
    assert p_4x.is_file()
    assert (dir_4x / "readme.md").is_file()
    content_4x = p_4x.read_text(encoding="utf-8")
    assert "4x Scaled Pixel Art Assets" in content_4x
    assert "fantasy_sword.png" in content_4x
    assert "Nearest-Neighbor" in content_4x

    # Test native readme
    p_nat = QualityAuditor.generate_folder_readme_native(dir_native, [dummy_src], [metric])
    assert p_nat.is_file()
    assert (dir_native / "readme.md").is_file()
    content_nat = p_nat.read_text(encoding="utf-8")
    assert "Original Native Source Assets" in content_nat
    assert "fantasy_sword.png" in content_nat
    assert "FANTASY Sword" in content_nat
