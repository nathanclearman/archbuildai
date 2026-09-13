"""
GPU draw handler for the 2D correction overlay.

Draws confidence-colored wall lines, door arcs, window marks, and room
labels on top of the 3D viewport during correction mode.
"""

import math

import bpy
import blf
import gpu
from gpu_extras.batch import batch_for_shader
from bpy_extras.view3d_utils import location_3d_to_region_2d


# ── Colors ────────────────────────────────────────────────────────────

COLOR_WALL = (0.2, 0.9, 0.3, 0.9)       # Green
COLOR_DOOR = (1.0, 0.6, 0.1, 0.9)       # Orange
COLOR_WINDOW = (0.2, 0.7, 1.0, 0.9)     # Cyan
COLOR_ROOM_TEXT = (1.0, 1.0, 1.0, 0.7)  # White semi-transparent


# ── Main draw callback ────────────────────────────────────────────────

def draw_correction_overlay(state):
    """Called by SpaceView3D draw_handler_add on every viewport redraw."""
    if not state or not state.active:
        return

    region, rv3d = _get_active_view3d()
    if not region or not rv3d:
        return

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')

    # Draw wall lines
    for wall in state.walls:
        start_3d = _empty_loc(wall.start_empty, wall.start)
        end_3d = _empty_loc(wall.end_empty, wall.end)

        s2d = location_3d_to_region_2d(region, rv3d, start_3d)
        e2d = location_3d_to_region_2d(region, rv3d, end_3d)
        if not s2d or not e2d:
            continue

        gpu.state.line_width_set(3.0)
        batch = batch_for_shader(shader, 'LINES', {"pos": [s2d, e2d]})
        shader.bind()
        shader.uniform_float("color", COLOR_WALL)
        batch.draw(shader)

    # Draw door markers (small cross)
    for door in state.doors:
        pos_3d = _empty_loc(door.empty, door.position)
        p2d = location_3d_to_region_2d(region, rv3d, pos_3d)
        if not p2d:
            continue
        _draw_cross(shader, p2d, 8, COLOR_DOOR)

    # Draw window markers (small diamond)
    for window in state.windows:
        pos_3d = _empty_loc(window.empty, window.position)
        p2d = location_3d_to_region_2d(region, rv3d, pos_3d)
        if not p2d:
            continue
        _draw_diamond(shader, p2d, 8, COLOR_WINDOW)

    gpu.state.line_width_set(1.0)

    # Draw room labels
    rooms = state.original_data.get("rooms", [])
    font_id = 0
    for room in rooms:
        polygon = room.get("polygon", [])
        if not polygon:
            continue
        cx = sum(p[0] for p in polygon) / len(polygon)
        cy = sum(p[1] for p in polygon) / len(polygon)
        p2d = location_3d_to_region_2d(region, rv3d, (cx, cy, 0))
        if not p2d:
            continue

        label = room.get("label", "")
        area = room.get("area")
        text = label.replace("_", " ").title()
        if area:
            text += f" ({area:.1f}m\u00b2)"

        blf.position(font_id, p2d[0], p2d[1], 0)
        blf.size(font_id, 14)
        blf.color(font_id, *COLOR_ROOM_TEXT)
        blf.draw(font_id, text)


# ── Drawing primitives ────────────────────────────────────────────────

def _draw_cross(shader, center, size, color):
    """Draw a + marker at center."""
    x, y = center
    gpu.state.line_width_set(2.0)
    verts = [
        (x - size, y), (x + size, y),
        (x, y - size), (x, y + size),
    ]
    batch = batch_for_shader(shader, 'LINES', {"pos": verts})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def _draw_diamond(shader, center, size, color):
    """Draw a diamond marker at center."""
    x, y = center
    gpu.state.line_width_set(2.0)
    verts = [
        (x, y + size), (x + size, y),
        (x + size, y), (x, y - size),
        (x, y - size), (x - size, y),
        (x - size, y), (x, y + size),
    ]
    batch = batch_for_shader(shader, 'LINES', {"pos": verts})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


# ── Helpers ───────────────────────────────────────────────────────────

def _empty_loc(empty, fallback):
    """Return a 3-tuple from an Empty's location or a fallback [x, y]."""
    if empty:
        return (empty.location.x, empty.location.y, 0.0)
    return (fallback[0], fallback[1], 0.0)


def _get_active_view3d():
    """Return (region, region_3d) for the first 3D viewport found."""
    for area in bpy.context.screen.areas:
        if area.type != 'VIEW_3D':
            continue
        for region in area.regions:
            if region.type == 'WINDOW':
                rv3d = area.spaces[0].region_3d
                return region, rv3d
    return None, None
