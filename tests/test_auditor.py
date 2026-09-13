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
