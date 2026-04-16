# Real-MLS eval set

A small hand-labelled set of genuine listing / brochure floor plans that the
model has never seen during training. This is the last-mile eval signal:
CubiCasa accuracy is a necessary-but-not-sufficient proxy, because CubiCasa
is European apartments, internally consistent SVG labels, and typically
cleaner line-work than an MLS upload. A plan fresh off a listing site has
JPEG artifacts, paper tint, watermarks, uneven scans, vendor glyphs, and
US-layout conventions the synthetic corpus is trying to approximate.

**This set is eval-only.** Do not train on it. Adding samples here is
additive — a larger real set makes the metric more trustworthy, never less.

## Directory layout

```
real_mls/
├── README.md
└── samples/
    ├── {slug}.png     # the input image (png / jpg, any size)
    └── {slug}.json    # canonical schema annotation (see model/schema.py)
```

One `{slug}.png` + one `{slug}.json` per sample, in the same directory,
sharing the same stem. The loader (`dataset.RealMLSLoader`) picks them up
automatically.

## Labeling conventions

Each JSON must validate against `schema.validate` from `model/schema.py`
and is the same shape the training targets use. Two things to get right:

1. **Scale.** `pixels_per_meter` is per-image. Measure against a known
   room dimension (US convention: king bed 2.0 m wide, 1-car garage door
   2.4 m). Off-by-20% scale is the single biggest error source — worth
   spending two minutes on.

2. **Room vocabulary.** Use the labels in `synthesize.US_ROOM_LABELS`
   (`living_room`, `kitchen`, `master_bedroom`, `en_suite`, `powder_room`,
   `walk_in_closet`, `garage`, etc). If the plan shows a room concept
   outside that list, pick the closest match and note it in a comment;
   we'd rather grow the vocabulary than have label drift.

Geometry conventions:

- **Walls**: `start` / `end` in meters, axis-aligned preferred; the model
  handles angled walls too but keep corners at integer-mm precision.
- **Openings**: `doors` and `windows` use the midpoint on the wall as
  `position`, `width` is the opening width in meters, `wall_index` points
  into the `walls` array.
- **Rooms**: closed polygon in meters, counter-clockwise, matching the
  visible room boundary (not the inner-wall surface).

## Producing a label

For a quick first pass:

1. Open the image in any pixel editor to read off corners.
2. Use `model/schema.py::FloorPlan` + `serialize` to emit the JSON — it
   applies the same rounding the training targets use.
3. Run the smoke-test CLI to confirm the file parses:
   ```
   python model/dataset.py --real-mls model/data/real_mls --limit 1
   ```

## Why not ZiND

ZiND (Zillow Indoor Dataset) is a related public corpus with real
listings, but its annotations are room-graph metadata and 360 photos, not
the 2D floor plan geometry this model learns from. If we do use ZiND later
it's eval-only: compare predicted room *topology* to ZiND's room graph.
It can't substitute for labelled 2D plans.
