"""
Top-down renderer for eyeball eval.

Takes a floor plan JSON (our canonical schema) and produces a PNG showing
walls, doors, windows, and labeled rooms. Optionally overlays the result
on top of the source image at 50% opacity so you can eyeball whether the
predicted geometry lines up with the real plan.

  # pred on white background
  python3 render_plan.py pred.json -o pred.png

  # pred overlaid on the original input image
  python3 render_plan.py pred.json -o compare.png --overlay plan.jpg

  # side-by-side: original | render
  python3 render_plan.py pred.json -o compare.png --side-by-side plan.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WALL_COLOR = (30, 30, 30, 255)
WALL_WIDTH_PX = 3
DOOR_COLOR = (60, 180, 60, 255)
WINDOW_COLOR = (60, 120, 220, 255)
FIXTURE_RADIUS_PX = 6
ROOM_OUTLINE = (200, 80, 80, 255)
ROOM_OUTLINE_WIDTH_PX = 2
ROOM_FILL = (255, 220, 220, 90)      # translucent
LABEL_FILL = (30, 30, 30, 255)
LABEL_BG = (255, 255, 255, 200)

FONT_CANDIDATES = (
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSans.ttf",
)


def _load_font(size: int):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _bounds(plan: dict) -> tuple[float, float, float, float]:
    """Return (min_x, min_y, max_x, max_y) in meters across all geometry."""
    xs: list[float] = []
    ys: list[float] = []
    for w in plan.get("walls", []):
        for pt in (w["start"], w["end"]):
            xs.append(pt[0])
            ys.append(pt[1])
    for r in plan.get("rooms", []):
        for pt in r.get("polygon", []):
            xs.append(pt[0])
            ys.append(pt[1])
    if not xs:
        return 0.0, 0.0, 10.0, 10.0
    return min(xs), min(ys), max(xs), max(ys)


def render(plan: dict, canvas_size: tuple[int, int] | None = None,
           padding_px: int = 40) -> Image.Image:
    """Render a floor plan dict to a new RGBA PIL image."""
    ppm = int(plan.get("scale", {}).get("pixels_per_meter", 50))
    min_x, min_y, max_x, max_y = _bounds(plan)
    width_m = max(max_x - min_x, 0.1)
    height_m = max(max_y - min_y, 0.1)

    if canvas_size is None:
        w_px = int(width_m * ppm) + 2 * padding_px
        h_px = int(height_m * ppm) + 2 * padding_px
    else:
        w_px, h_px = canvas_size

    def to_px(pt):
        return (
            int((pt[0] - min_x) * ppm) + padding_px,
            int((pt[1] - min_y) * ppm) + padding_px,
        )

    img = Image.new("RGBA", (w_px, h_px), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    for room in plan.get("rooms", []):
        poly = [to_px(p) for p in room.get("polygon", [])]
        if len(poly) >= 3:
            draw.polygon(poly, fill=ROOM_FILL, outline=ROOM_OUTLINE,
                         width=ROOM_OUTLINE_WIDTH_PX)

    for wall in plan.get("walls", []):
        draw.line([to_px(wall["start"]), to_px(wall["end"])],
                  fill=WALL_COLOR, width=WALL_WIDTH_PX)

    for door in plan.get("doors", []):
        cx, cy = to_px(door["position"])
        r = FIXTURE_RADIUS_PX
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     fill=DOOR_COLOR, outline=(0, 80, 0, 255), width=1)

    for win in plan.get("windows", []):
        cx, cy = to_px(win["position"])
        r = FIXTURE_RADIUS_PX
        draw.rectangle([cx - r, cy - r, cx + r, cy + r],
                       fill=WINDOW_COLOR, outline=(0, 40, 120, 255), width=1)

    font = _load_font(14)
    for room in plan.get("rooms", []):
        poly = room.get("polygon", [])
        if not poly:
            continue
        cx_m = sum(p[0] for p in poly) / len(poly)
        cy_m = sum(p[1] for p in poly) / len(poly)
        cx, cy = to_px((cx_m, cy_m))
        label = str(room.get("label", ""))
        tw = draw.textlength(label, font=font) if label else 0
        th = 14
        bg = (cx - tw / 2 - 4, cy - th / 2 - 2,
              cx + tw / 2 + 4, cy + th / 2 + 2)
        draw.rectangle(bg, fill=LABEL_BG)
        draw.text((cx - tw / 2, cy - th / 2), label, fill=LABEL_FILL, font=font)

    return img


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _compose_overlay(render_img: Image.Image, source_img_path: str,
                     alpha: float = 0.55) -> Image.Image:
    src = Image.open(source_img_path).convert("RGBA")
    # Scale source to match render height, preserving aspect.
    scale = render_img.height / src.height
    new_w = int(src.width * scale)
    src = src.resize((new_w, render_img.height), Image.BILINEAR)
    # Center-pad or crop source to match render width.
    if src.width < render_img.width:
        bg = Image.new("RGBA", render_img.size, (255, 255, 255, 255))
        bg.paste(src, ((render_img.width - src.width) // 2, 0))
        src = bg
    elif src.width > render_img.width:
        x0 = (src.width - render_img.width) // 2
        src = src.crop((x0, 0, x0 + render_img.width, render_img.height))
    # Make the render partially transparent over the source.
    r = render_img.copy()
    alpha_mask = r.split()[3].point(lambda v: int(v * alpha))
    r.putalpha(alpha_mask)
    return Image.alpha_composite(src, r)


def _compose_side_by_side(render_img: Image.Image,
                          source_img_path: str) -> Image.Image:
    src = Image.open(source_img_path).convert("RGBA")
    scale = render_img.height / src.height
    src = src.resize((int(src.width * scale), render_img.height), Image.BILINEAR)
    out = Image.new("RGBA", (src.width + render_img.width, render_img.height),
                    (255, 255, 255, 255))
    out.paste(src, (0, 0))
    out.paste(render_img, (src.width, 0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("plan_json", help="Path to floor plan JSON")
    ap.add_argument("-o", "--out", required=True, help="Output PNG path")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--overlay", metavar="IMAGE",
                       help="Overlay the render on this source image")
    group.add_argument("--side-by-side", metavar="IMAGE", dest="side_by_side",
                       help="Place source image next to the render")
    args = ap.parse_args()

    plan = _load(args.plan_json)
    img = render(plan)
    if args.overlay:
        img = _compose_overlay(img, args.overlay)
    elif args.side_by_side:
        img = _compose_side_by_side(img, args.side_by_side)
    img.convert("RGB").save(args.out)
    print(f"wrote {args.out}  ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
