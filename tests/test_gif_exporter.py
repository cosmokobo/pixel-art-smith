"""Tests for GifExporter: transparent motion GIFs and composite multi-motion previews."""

from PIL import Image

from pixel_art_smith.core.gif_exporter import GifExporter


def test_export_motion_gif(tmp_path):
    frames = [
        Image.new("RGBA", (32, 32), (255, 0, 0, 255)),
        Image.new("RGBA", (32, 32), (0, 255, 0, 255)),
    ]
    out_gif = tmp_path / "test_motion.gif"
    result = GifExporter.export_motion_gif(frames, out_gif, duration=100)

    assert result.is_file()
    assert result.stat().st_size > 0

    with Image.open(result) as im:
        assert im.format == "GIF"
        assert getattr(im, "is_animated", False)
        assert getattr(im, "n_frames", 1) == 2


def test_export_all_gifs(tmp_path):
    std_grid = [
        [Image.new("RGBA", (16, 16), (255, 0, 0, 255)) for _ in range(4)]
        for _ in range(4)
    ]
    out_dir = tmp_path / "gifs"
    exported = GifExporter.export_all_gifs(
        std_grid=std_grid,
        output_dir=out_dir,
        stem="hero",
        duration=120,
    )

    # 4 directional motion GIFs + 1 composite preview GIF = 5 GIFs total
    assert len(exported["individual_gifs"]) == 4
    assert exported["composite_gif"] is not None
    assert (out_dir / "hero_motion_00_down.gif").is_file()
    assert (out_dir / "hero_motion_01_left.gif").is_file()
    assert (out_dir / "hero_motion_02_right.gif").is_file()
    assert (out_dir / "hero_motion_03_up.gif").is_file()
    assert (out_dir / "hero_all_motions.gif").is_file()
