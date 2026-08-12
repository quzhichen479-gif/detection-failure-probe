"""Regenerate the tiny deterministic demo images."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1] / "demo" / "dataset" / "images"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    scenes = [
        (
            "scene_01.png",
            (25, 39, 67),
            [
                (56, 72, 136, 168, (65, 201, 180)),
                (253, 46, 259, 51, (255, 190, 80)),
            ],
        ),
        ("scene_02.png", (48, 31, 54), [(80, 116, 240, 124, (220, 100, 170))]),
        ("scene_03.png", (26, 55, 44), []),
        ("scene_04.png", (58, 48, 24), [(125, 70, 200, 180, (120, 180, 255))]),
    ]
    for name, background, boxes in scenes:
        image = Image.new("RGB", (320, 240), background)
        draw = ImageDraw.Draw(image)
        for step in range(0, 320, 32):
            draw.line((step, 0, step, 240), fill=tuple(min(255, value + 7) for value in background))
        for box in boxes:
            draw.rectangle(box[:4], outline=box[4], width=3)
        draw.text((12, 12), name.removesuffix(".png"), fill=(235, 240, 250))
        image.save(ROOT / name, optimize=True)


if __name__ == "__main__":
    main()
