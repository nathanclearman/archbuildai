"""
Geometry generation from parsed floor plan JSON data.

Creates Blender meshes for walls, floors, ceilings, door/window openings,
room labels, and furniture from the structured JSON output of the floor plan parser.
"""

import copy
import os

import bpy
import bmesh
import math
from mathutils import Vector, Matrix


# ── Collection helpers ─────────────────────────────────────────────────

def _recursive_remove_collection(collection):
    """Remove all objects and nested sub-collections from a collection."""
    # Recurse into children first
    for child in list(collection.children):
        _recursive_remove_collection(child)
        collection.children.unlink(child)
        bpy.data.collections.remove(child)
    # Remove objects (and their mesh data to prevent orphans)
    for obj in list(collection.objects):
        mesh = obj.data if obj.type == 'MESH' else None
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def create_floorplan_collection(context):
    """Create or clear the FloorPlan3D collection (including story sub-collections).

    Performs a thorough cleanup: recursively removes all child collections,
    objects, and orphaned mesh data blocks to prevent stale `.001` suffixes
    from breaking name-based lookups (e.g. boolean modifiers).
    """
    collection = bpy.data.collections.get("FloorPlan3D")
    if collection:
        _recursive_remove_collection(collection)
    else:
        collection = bpy.data.collections.new("FloorPlan3D")
        context.scene.collection.children.link(collection)

    # Also clean up any orphaned Story_ collections not under FloorPlan3D
    # (can happen if a previous run crashed or was interrupted).
    for col in list(bpy.data.collections):
        if col.name.startswith("Story_") and col != collection:
            _recursive_remove_collection(col)
            # Only remove if not linked anywhere
            if col.users == 0:
                bpy.data.collections.remove(col)

    return collection


def get_or_create_story_collection(parent, story_index):
    """Get or create a Story_N sub-collection under parent.

    Looks only in parent's children, not globally, to avoid picking up
    stale collections from previous runs.
    """
    name = f"Story_{story_index}"
    # Search only among parent's children
    for child in parent.children:
        if child.name == name:
            return child
    col = bpy.data.collections.new(name)
    parent.children.link(col)
    return col


def _link_to_collection(obj, collection):
    """Link an object to the given collection only."""
    collection.objects.link(obj)


# ── Polygon simplification ────────────────────────────────────────────

def _remove_collinear(polygon, tol=0.2):
    """Remove vertices that lie approximately on the line between their neighbours."""
    if len(polygon) < 4:
        return polygon
    result = []
    n = len(polygon)
    for i in range(n):
        p_prev = polygon[(i - 1) % n]
        p_curr = polygon[i]
        p_next = polygon[(i + 1) % n]
        # Cross product magnitude / edge length
        ex, ey = p_next[0] - p_prev[0], p_next[1] - p_prev[1]
        edge_len = math.sqrt(ex * ex + ey * ey)
        if edge_len < 1e-6:
            continue
        cross = abs((p_curr[0] - p_prev[0]) * ey - (p_curr[1] - p_prev[1]) * ex)
        dist_from_line = cross / edge_len
        if dist_from_line > tol:
            result.append(p_curr)
    return result if len(result) >= 3 else polygon


def _remove_close_duplicates(polygon, tol=0.1):
    """Remove consecutive duplicate vertices within tolerance."""
    if len(polygon) < 3:
        return polygon
    result = [polygon[0]]
    for i in range(1, len(polygon)):
        dx = polygon[i][0] - result[-1][0]
        dy = polygon[i][1] - result[-1][1]
        if math.sqrt(dx * dx + dy * dy) > tol:
            result.append(polygon[i])
    # Check wrap-around
    if len(result) > 1:
        dx = result[0][0] - result[-1][0]
        dy = result[0][1] - result[-1][1]
        if math.sqrt(dx * dx + dy * dy) <= tol:
            result.pop()
    return result if len(result) >= 3 else polygon


def _shoelace_area(polygon):
    """Compute polygon area using the shoelace formula."""
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


def _signed_polygon_area(polygon):
    """Signed shoelace area of a 2D polygon. Positive means CCW winding in
    a standard math frame (y-up). Blender's world XY is the same frame
    so a CCW polygon gets a +Z face normal when built with `bm.faces.new`.
    Degenerate / <3-vertex inputs return 0."""
    n = len(polygon)
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += polygon[i][0] * polygon[j][1]
        a -= polygon[j][0] * polygon[i][1]
    return a / 2.0


def _ensure_ccw(polygon):
    """Return `polygon` in CCW winding, reversing if it came in CW.

    `bm.faces.new(verts)` derives the face normal from vertex order:
    CCW -> +Z (faces up), CW -> -Z (faces down). Floors must face up and
    ceilings are flipped from a CCW base, so both need a known winding.
    Upstream JSON gives no guarantee (the synthetic generator's flip_x
    augmentation reverses winding on ~50% of samples, and Shapely's
    buffer() in _expand_polygon emits CW exterior rings), so normalize
    here instead of trusting the input.
    """
    if _signed_polygon_area(polygon) < 0:
        return list(reversed(polygon))
    return polygon


def _convex_hull(points):
    """Compute the convex hull of a set of 2D points (Andrew's monotone chain)."""
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _min_bounding_rect(hull):
    """Find the minimum area bounding rectangle for a convex hull.
    Returns (corners, area, angle) where corners is a list of 4 [x,y] points."""
    if len(hull) < 3:
        return hull, 0, 0

    min_area = float('inf')
    best_rect = None
    best_angle = 0

    n = len(hull)
    for i in range(n):
        # Edge from hull[i] to hull[(i+1)%n]
        ex = hull[(i + 1) % n][0] - hull[i][0]
        ey = hull[(i + 1) % n][1] - hull[i][1]
        edge_len = math.sqrt(ex * ex + ey * ey)
        if edge_len < 1e-9:
            continue
        # Unit vector along edge
        ux, uy = ex / edge_len, ey / edge_len
        angle = math.atan2(uy, ux)

        # Project all hull points onto this edge's coordinate system
        min_proj = float('inf')
        max_proj = float('-inf')
        min_perp = float('inf')
        max_perp = float('-inf')

        for p in hull:
            proj = (p[0] - hull[i][0]) * ux + (p[1] - hull[i][1]) * uy
            perp = (p[0] - hull[i][0]) * (-uy) + (p[1] - hull[i][1]) * ux
            min_proj = min(min_proj, proj)
            max_proj = max(max_proj, proj)
            min_perp = min(min_perp, perp)
            max_perp = max(max_perp, perp)

        area = (max_proj - min_proj) * (max_perp - min_perp)
        if area < min_area:
            min_area = area
            best_angle = angle
            # Reconstruct corners
            ox = hull[i][0]
            oy = hull[i][1]
            corners = []
            for pp, pr in [(min_proj, min_perp), (max_proj, min_perp),
                           (max_proj, max_perp), (min_proj, max_perp)]:
                cx = ox + pp * ux + pr * (-uy)
                cy = oy + pp * uy + pr * ux
                corners.append([round(cx, 3), round(cy, 3)])
            best_rect = corners

    return best_rect, min_area, best_angle


def _compute_exposed_ground_outline(ground_plan, upper_plan,
                                     upper_hull=None):
    """Compute the ground floor area NOT covered by the upper floor.

    Uses grid rasterisation: fills ground floor rooms, subtracts the upper
    floor hull, then traces the remaining contour.  Returns a list of [x, y]
    world-coordinate points, or None if no significant exposed area exists.

    Args:
        ground_plan: ground floor plan dict with "rooms".
        upper_plan: upper floor plan dict (used as fallback if no hull).
        upper_hull: pre-computed upper floor hull polygon (preferred).
                    Using the hull instead of individual rooms avoids gaps
                    between rooms creating false exposed areas.
    """
    ground_rooms = [r["polygon"] for r in ground_plan.get("rooms", [])
                    if "polygon" in r and len(r["polygon"]) >= 3]

    # Prefer the hull (single polygon covering entire upper floor) over
    # individual rooms, which can have gaps between them.
    if upper_hull and len(upper_hull) >= 3:
        upper_polys = [upper_hull]
    else:
        upper_polys = [r["polygon"] for r in upper_plan.get("rooms", [])
                       if "polygon" in r and len(r["polygon"]) >= 3]
    if not ground_rooms:
        return None

    # Bounding box across both floors
    all_x, all_y = [], []
    for poly in ground_rooms + upper_polys:
        for p in poly:
            all_x.append(p[0])
            all_y.append(p[1])

    pad = 1.0
    min_x = min(all_x) - pad
    min_y = min(all_y) - pad
    max_x = max(all_x) + pad
    max_y = max(all_y) + pad

    res = 5
    gw = max(int((max_x - min_x) * res) + 1, 3)
    gh = max(int((max_y - min_y) * res) + 1, 3)

    if gw > 800 or gh > 800:
        res = 3
        gw = max(int((max_x - min_x) * res) + 1, 3)
        gh = max(int((max_y - min_y) * res) + 1, 3)

    grid = [False] * (gw * gh)

    def _set(x, y, val=True):
        if 0 <= x < gw and 0 <= y < gh:
            grid[y * gw + x] = val

    def _fill_poly(polygon, val=True):
        pts = [((p[0] - min_x) * res, (p[1] - min_y) * res)
               for p in polygon]
        if len(pts) < 3:
            return
        ys = [p[1] for p in pts]
        y_min_i = max(0, int(min(ys)))
        y_max_i = min(gh - 1, int(max(ys)))
        n = len(pts)
        for y in range(y_min_i, y_max_i + 1):
            intersections = []
            for i in range(n):
                j = (i + 1) % n
                y0, y1 = pts[i][1], pts[j][1]
                if y0 == y1:
                    continue
                if y0 > y1:
                    y0, y1 = y1, y0
                    x0, x1 = pts[j][0], pts[i][0]
                else:
                    x0, x1 = pts[i][0], pts[j][0]
                if y0 <= y < y1:
                    t = (y - y0) / (y1 - y0)
                    ix = x0 + t * (x1 - x0)
                    intersections.append(ix)
            intersections.sort()
            for k in range(0, len(intersections) - 1, 2):
                x_start = max(0, int(intersections[k]))
                x_end = min(gw - 1, int(intersections[k + 1]))
                for x in range(x_start, x_end + 1):
                    _set(x, y, val)

    # Rasterize ground floor rooms (fill)
    for poly in ground_rooms:
        _fill_poly(poly, True)

    # Dilate slightly to close gaps
    dilate_r = max(int(0.3 * res), 1)
    old_grid = list(grid)
    for y in range(gh):
        for x in range(gw):
            if old_grid[y * gw + x]:
                for dy in range(-dilate_r, dilate_r + 1):
                    for dx in range(-dilate_r, dilate_r + 1):
                        nx_, ny_ = x + dx, y + dy
                        if 0 <= nx_ < gw and 0 <= ny_ < gh:
                            grid[ny_ * gw + nx_] = True

    # Subtract upper floor footprint (un-fill) — with extra margin to ensure
    # the upper floor fully covers its area and doesn't leave a thin strip
    # at the boundary between floors.
    upper_grid = [False] * (gw * gh)
    for poly in upper_polys:
        pts = [((p[0] - min_x) * res, (p[1] - min_y) * res)
               for p in poly]
        if len(pts) < 3:
            continue
        ys = [p[1] for p in pts]
        y_min_i = max(0, int(min(ys)))
        y_max_i = min(gh - 1, int(max(ys)))
        n = len(pts)
        for y in range(y_min_i, y_max_i + 1):
            intersections = []
            for i in range(n):
                j = (i + 1) % n
                y0, y1 = pts[i][1], pts[j][1]
                if y0 == y1:
                    continue
                if y0 > y1:
                    y0, y1 = y1, y0
                    x0, x1 = pts[j][0], pts[i][0]
                else:
                    x0, x1 = pts[i][0], pts[j][0]
                if y0 <= y < y1:
                    t = (y - y0) / (y1 - y0)
                    ix = x0 + t * (x1 - x0)
                    intersections.append(ix)
            intersections.sort()
            for k in range(0, len(intersections) - 1, 2):
                x_start = max(0, int(intersections[k]))
                x_end = min(gw - 1, int(intersections[k + 1]))
                for x in range(x_start, x_end + 1):
                    if 0 <= x < gw and 0 <= y < gh:
                        upper_grid[y * gw + x] = True

    # Dilate upper floor to add margin
    old_upper = list(upper_grid)
    margin = max(int(0.5 * res), 1)
    for y in range(gh):
        for x in range(gw):
            if old_upper[y * gw + x]:
                for dy in range(-margin, margin + 1):
                    for dx in range(-margin, margin + 1):
                        nx_, ny_ = x + dx, y + dy
                        if 0 <= nx_ < gw and 0 <= ny_ < gh:
                            upper_grid[ny_ * gw + nx_] = True

    # Subtract: exposed = ground AND NOT upper
    for i in range(gw * gh):
        if upper_grid[i]:
            grid[i] = False

    # Check if any cells remain
    if not any(grid):
        return None

    # Flood-fill from border to mark exterior of exposed area
    exterior = [False] * (gw * gh)
    stack = []
    for x in range(gw):
        if not grid[x]:
            stack.append((x, 0))
        if not grid[(gh - 1) * gw + x]:
            stack.append((x, gh - 1))
    for y in range(gh):
        if not grid[y * gw]:
            stack.append((0, y))
        if not grid[y * gw + gw - 1]:
            stack.append((gw - 1, y))

    while stack:
        sx, sy = stack.pop()
        if not (0 <= sx < gw and 0 <= sy < gh):
            continue
        idx = sy * gw + sx
        if exterior[idx] or grid[idx]:
            continue
        exterior[idx] = True
        stack.append((sx - 1, sy))
        stack.append((sx + 1, sy))
        stack.append((sx, sy - 1))
        stack.append((sx, sy + 1))

    # Trace contour
    outline = _trace_grid_contour(grid, exterior, gw, gh, min_x, min_y, res)
    return outline


def _compute_building_outline(floor_plan_data):
    """Compute the actual building outline from room polygons.

    Pure Python implementation — no numpy/cv2 required (works inside
    Blender's bundled Python).

    Strategy:
    1. Rasterize all room polygons + wall thickness onto a boolean grid
    2. Dilate the grid to close small gaps between rooms
    3. Flood-fill from border to mark exterior
    4. Trace the boundary between interior and exterior cells
    5. Simplify the resulting outline polygon

    Returns a list of [x, y] points tracing the building perimeter, or
    None if there aren't enough points.
    """
    rooms = floor_plan_data.get("rooms", [])
    room_polys = [r["polygon"] for r in rooms
                  if "polygon" in r and len(r["polygon"]) >= 3]

    if not room_polys:
        return None

    # Collect bounding box
    all_x, all_y = [], []
    for poly in room_polys:
        for p in poly:
            all_x.append(p[0])
            all_y.append(p[1])

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    pad = 1.0  # padding so flood-fill can reach all borders
    min_x -= pad
    min_y -= pad
    max_x += pad
    max_y += pad

    res = 5  # pixels per metre (lower than cv2 version for speed)
    gw = max(int((max_x - min_x) * res) + 1, 3)
    gh = max(int((max_y - min_y) * res) + 1, 3)

    if gw > 800 or gh > 800:
        res = 3
        gw = max(int((max_x - min_x) * res) + 1, 3)
        gh = max(int((max_y - min_y) * res) + 1, 3)

    # Flat boolean grid (row-major)
    grid = [False] * (gw * gh)

    def _set(x, y):
        if 0 <= x < gw and 0 <= y < gh:
            grid[y * gw + x] = True

    def _get(x, y):
        if 0 <= x < gw and 0 <= y < gh:
            return grid[y * gw + x]
        return False

    # --- Rasterize room polygons (scanline fill) ---
    def _fill_poly(polygon):
        pts = [((p[0] - min_x) * res, (p[1] - min_y) * res)
               for p in polygon]
        if len(pts) < 3:
            return
        # Find y range
        ys = [p[1] for p in pts]
        y_min_i = max(0, int(min(ys)))
        y_max_i = min(gh - 1, int(max(ys)))
        n = len(pts)
        for y in range(y_min_i, y_max_i + 1):
            intersections = []
            for i in range(n):
                j = (i + 1) % n
                y0, y1 = pts[i][1], pts[j][1]
                if y0 == y1:
                    continue
                if y0 > y1:
                    y0, y1 = y1, y0
                    x0, x1 = pts[j][0], pts[i][0]
                else:
                    x0, x1 = pts[i][0], pts[j][0]
                if y0 <= y < y1:
                    t = (y - y0) / (y1 - y0)
                    ix = x0 + t * (x1 - x0)
                    intersections.append(ix)
            intersections.sort()
            for k in range(0, len(intersections) - 1, 2):
                x_start = max(0, int(intersections[k]))
                x_end = min(gw - 1, int(intersections[k + 1]))
                for x in range(x_start, x_end + 1):
                    _set(x, y)

    for poly in room_polys:
        _fill_poly(poly)

    # --- Rasterize walls (thick lines) to connect rooms ---
    def _draw_thick_line(x0, y0, x1, y1, thickness):
        dx = x1 - x0
        dy = y1 - y0
        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.5:
            return
        ux, uy = dx / length, dy / length
        nx, ny = -uy, ux  # perpendicular
        ht = thickness / 2
        # Rectangle corners
        corners = [
            (x0 + nx * ht, y0 + ny * ht),
            (x1 + nx * ht, y1 + ny * ht),
            (x1 - nx * ht, y1 - ny * ht),
            (x0 - nx * ht, y0 - ny * ht),
        ]
        # Convert to grid coords and fill
        grid_corners = [((c[0] - min_x) * res, (c[1] - min_y) * res)
                        for c in corners]
        _fill_poly_raw(grid_corners)

    def _fill_poly_raw(pts):
        """Fill polygon given in grid coordinates."""
        if len(pts) < 3:
            return
        ys = [p[1] for p in pts]
        y_min_i = max(0, int(min(ys)))
        y_max_i = min(gh - 1, int(max(ys)))
        n = len(pts)
        for y in range(y_min_i, y_max_i + 1):
            intersections = []
            for i in range(n):
                j = (i + 1) % n
                y0, y1 = pts[i][1], pts[j][1]
                if y0 == y1:
                    continue
                if y0 > y1:
                    y0, y1 = y1, y0
                    x0, x1 = pts[j][0], pts[i][0]
                else:
                    x0, x1 = pts[i][0], pts[j][0]
                if y0 <= y < y1:
                    t = (y - y0) / (y1 - y0)
                    ix = x0 + t * (x1 - x0)
                    intersections.append(ix)
            intersections.sort()
            for k in range(0, len(intersections) - 1, 2):
                x_start = max(0, int(intersections[k]))
                x_end = min(gw - 1, int(intersections[k + 1]))
                for x in range(x_start, x_end + 1):
                    _set(x, y)

    walls = floor_plan_data.get("walls", [])
    for w in walls:
        _draw_thick_line(w["start"][0], w["start"][1],
                         w["end"][0], w["end"][1], 0.3)

    # --- Dilate grid to close small gaps ---
    dilate_r = max(int(0.3 * res), 1)
    old_grid = list(grid)
    for y in range(gh):
        for x in range(gw):
            if old_grid[y * gw + x]:
                for dy in range(-dilate_r, dilate_r + 1):
                    for dx in range(-dilate_r, dilate_r + 1):
                        nx_, ny_ = x + dx, y + dy
                        if 0 <= nx_ < gw and 0 <= ny_ < gh:
                            grid[ny_ * gw + nx_] = True

    # --- Flood-fill from border to mark exterior ---
    exterior = [False] * (gw * gh)
    stack = []
    # Seed from all border cells that are not filled
    for x in range(gw):
        if not grid[x]:
            stack.append((x, 0))
        if not grid[(gh - 1) * gw + x]:
            stack.append((x, gh - 1))
    for y in range(gh):
        if not grid[y * gw]:
            stack.append((0, y))
        if not grid[y * gw + gw - 1]:
            stack.append((gw - 1, y))

    while stack:
        sx, sy = stack.pop()
        idx = sy * gw + sx
        if exterior[idx] or grid[idx]:
            continue
        if not (0 <= sx < gw and 0 <= sy < gh):
            continue
        exterior[idx] = True
        if sx > 0:
            stack.append((sx - 1, sy))
        if sx < gw - 1:
            stack.append((sx + 1, sy))
        if sy > 0:
            stack.append((sx, sy - 1))
        if sy < gh - 1:
            stack.append((sx, sy + 1))

    # Interior = filled OR not exterior
    # Building cells = not exterior
    # Find boundary cells (building cells adjacent to exterior)
    boundary_pts = []
    for y in range(gh):
        for x in range(gw):
            if exterior[y * gw + x]:
                continue
            # Check if adjacent to exterior
            is_boundary = False
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ax, ay = x + dx, y + dy
                if ax < 0 or ax >= gw or ay < 0 or ay >= gh:
                    is_boundary = True
                    break
                if exterior[ay * gw + ax]:
                    is_boundary = True
                    break
            if is_boundary:
                # Convert back to world coordinates
                wx = x / res + min_x
                wy = y / res + min_y
                boundary_pts.append((wx, wy))

    if len(boundary_pts) < 3:
        return None

    # Trace the actual grid boundary using contour following.
    # This preserves concave shapes (L-shaped, U-shaped, etc.)
    outline = _trace_grid_contour(grid, exterior, gw, gh,
                                  min_x, min_y, res)
    if outline and len(outline) >= 3:
        return outline

    # Fallback: convex hull of boundary points
    hull = _convex_hull(boundary_pts)
    return [[round(p[0], 3), round(p[1], 3)] for p in hull]


def _trace_grid_contour(grid, exterior, gw, gh, min_x, min_y, res):
    """Trace the building outline on the boolean grid using contour following.

    Uses a clockwise boundary-tracing algorithm (Moore neighbourhood):
    1. Find the topmost-leftmost building cell adjacent to exterior
    2. Walk clockwise around the boundary
    3. Convert grid coordinates to world coordinates
    4. Simplify with Douglas-Peucker

    Returns a list of [x, y] world-coordinate points, or None.
    """
    def is_building(x, y):
        """Cell is part of the building (filled and not exterior)."""
        if x < 0 or x >= gw or y < 0 or y >= gh:
            return False
        idx = y * gw + x
        return not exterior[idx] and grid[idx]

    def is_outside(x, y):
        """Cell is outside the building."""
        if x < 0 or x >= gw or y < 0 or y >= gh:
            return True
        return exterior[y * gw + x]

    # Find starting cell: topmost row, leftmost building cell on boundary
    start = None
    for y in range(gh):
        for x in range(gw):
            if not is_building(x, y):
                continue
            # Check if it's a boundary cell (adjacent to outside)
            if (is_outside(x - 1, y) or is_outside(x + 1, y) or
                    is_outside(x, y - 1) or is_outside(x, y + 1)):
                start = (x, y)
                break
        if start:
            break

    if not start:
        return None

    # Moore neighbourhood tracing (clockwise)
    # Direction indices: 0=right, 1=down-right, 2=down, 3=down-left,
    #                    4=left, 5=up-left, 6=up, 7=up-right
    dx8 = [1, 1, 0, -1, -1, -1, 0, 1]
    dy8 = [0, 1, 1, 1, 0, -1, -1, -1]

    contour = [start]
    cx, cy = start
    # Start looking from the left (direction 4) since we found from top-left
    direction = 6  # start looking upward (we came from above)

    max_steps = gw * gh * 2  # safety limit
    for _ in range(max_steps):
        # Start searching clockwise from (direction + 5) % 8
        # (i.e., turn back-left relative to the direction we came from)
        start_dir = (direction + 5) % 8
        found = False
        for k in range(8):
            d = (start_dir + k) % 8
            nx_, ny_ = cx + dx8[d], cy + dy8[d]
            if is_building(nx_, ny_):
                cx, cy = nx_, ny_
                direction = d
                contour.append((cx, cy))
                found = True
                break

        if not found:
            break

        # Stop when we return to start
        if (cx, cy) == start and len(contour) > 3:
            break

    # Strip duplicate closing point (contour tracer appends start again)
    while len(contour) > 3 and contour[-1] == contour[0]:
        contour.pop()

    if len(contour) < 3:
        return None

    # Convert grid coords to world coords
    world_pts = []
    for gx, gy in contour:
        wx = gx / res + min_x
        wy = gy / res + min_y
        world_pts.append((wx, wy))

    # Douglas-Peucker simplification
    simplified = _douglas_peucker(world_pts, epsilon=0.3)

    if len(simplified) < 3:
        return None

    # Ensure no duplicate closing point (bmesh faces need unique vertices)
    result = [[round(p[0], 3), round(p[1], 3)] for p in simplified]
    while len(result) > 3 and result[-1] == result[0]:
        result.pop()

    return result


def _douglas_peucker(points, epsilon):
    """Simplify a polyline using the Douglas-Peucker algorithm.

    Reduces the number of points while preserving the shape within
    the given epsilon tolerance (in world units / metres).
    """
    if len(points) <= 2:
        return list(points)

    # Find the point with maximum distance from the line start->end
    start = points[0]
    end = points[-1]
    max_dist = 0
    max_idx = 0

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    line_len_sq = dx * dx + dy * dy

    for i in range(1, len(points) - 1):
        if line_len_sq < 1e-10:
            dist = math.sqrt((points[i][0] - start[0]) ** 2 +
                             (points[i][1] - start[1]) ** 2)
        else:
            t = ((points[i][0] - start[0]) * dx +
                 (points[i][1] - start[1]) * dy) / line_len_sq
            t = max(0, min(1, t))
            proj_x = start[0] + t * dx
            proj_y = start[1] + t * dy
            dist = math.sqrt((points[i][0] - proj_x) ** 2 +
                             (points[i][1] - proj_y) ** 2)
        if dist > max_dist:
            max_dist = dist
            max_idx = i

    if max_dist > epsilon:
        left = _douglas_peucker(points[:max_idx + 1], epsilon)
        right = _douglas_peucker(points[max_idx:], epsilon)
        return left[:-1] + right
    else:
        return [start, end]


def _simplify_roof_outline(hull_pts, epsilon=1.5):
    """Simplify a building outline for use as a roof footprint.

    Roofs look terrible with many small jogs from room boundaries.
    This function simplifies the outline (use smaller epsilon for
    multi-story roofs where the outline represents the actual building
    shape with wings/setbacks that must be preserved):
    1. Douglas-Peucker with 1.5m epsilon to remove small indentations
    2. Remove near-collinear points (< 10° deviation)
    3. Snap near-axis-aligned edges to be perfectly axis-aligned

    Returns a simplified list of [x, y] points (minimum 3).
    """
    if len(hull_pts) <= 4:
        return hull_pts  # already simple enough

    # Convert to tuples for processing
    pts = [(p[0], p[1]) for p in hull_pts]

    # Step 1: Douglas-Peucker simplification
    simplified = _douglas_peucker(pts, epsilon=epsilon)

    if len(simplified) < 3:
        simplified = _douglas_peucker(pts, epsilon=epsilon * 0.5)

    if len(simplified) < 3:
        return hull_pts  # give up, use original

    # Step 2: Remove near-collinear points (angle threshold: ~10°)
    cleaned = []
    n = len(simplified)
    for i in range(n):
        p_prev = simplified[(i - 1) % n]
        p_curr = simplified[i]
        p_next = simplified[(i + 1) % n]

        # Vectors
        v1x = p_curr[0] - p_prev[0]
        v1y = p_curr[1] - p_prev[1]
        v2x = p_next[0] - p_curr[0]
        v2y = p_next[1] - p_curr[1]

        len1 = math.sqrt(v1x * v1x + v1y * v1y)
        len2 = math.sqrt(v2x * v2x + v2y * v2y)

        if len1 < 0.01 or len2 < 0.01:
            continue  # skip degenerate points

        # Cross product magnitude = sin(angle) * len1 * len2
        cross = abs(v1x * v2y - v1y * v2x)
        sin_angle = cross / (len1 * len2)

        # Keep point only if angle deviation > ~10° (sin(10°) ≈ 0.17)
        if sin_angle > 0.17:
            cleaned.append(p_curr)

    if len(cleaned) < 3:
        cleaned = list(simplified)

    # Step 3: Snap near-axis-aligned edges.
    # If two consecutive points differ by < 0.5m in X or Y, align them.
    for i in range(len(cleaned)):
        j = (i + 1) % len(cleaned)
        dx = abs(cleaned[j][0] - cleaned[i][0])
        dy = abs(cleaned[j][1] - cleaned[i][1])
        if dx < 0.5 and dy > 1.0:
            # Nearly vertical edge — average X
            avg_x = (cleaned[i][0] + cleaned[j][0]) / 2
            cleaned[i] = (avg_x, cleaned[i][1])
            cleaned[j] = (avg_x, cleaned[j][1])
        elif dy < 0.5 and dx > 1.0:
            # Nearly horizontal edge — average Y
            avg_y = (cleaned[i][1] + cleaned[j][1]) / 2
            cleaned[i] = (cleaned[i][0], avg_y)
            cleaned[j] = (cleaned[j][0], avg_y)

    return [[round(p[0], 3), round(p[1], 3)] for p in cleaned]


def compute_building_footprint(floor_plan_data):
    """Compute the building's outer footprint from room polygons and walls.

    Prefers the actual building outline computed from room polygons
    (which accurately represents L-shapes and complex footprints).
    Falls back to convex hull of wall endpoints.

    Returns:
        dict with keys:
            'hull': list of [x,y] points (actual building outline or convex hull)
            'rect_corners': list of 4 [x,y] points (min-area bounding rect)
            'rect_angle': float (angle of the bounding rect's long axis, radians)
            'rect_long': float (length of long side in metres)
            'rect_short': float (length of short side in metres)
            'center': [x, y] centre of bounding rect
    """
    # Try to get the actual building outline from room polygons
    outline = _compute_building_outline(floor_plan_data)

    if outline and len(outline) >= 3:
        # Ensure no duplicate closing point (causes degenerate bmesh faces)
        while len(outline) > 3 and outline[-1] == outline[0]:
            outline.pop()
        hull = outline
    else:
        # Fallback: convex hull of wall endpoints
        walls = floor_plan_data.get("walls", [])
        pts = set()
        for w in walls:
            s = w.get("start", [0, 0])
            e = w.get("end", [0, 0])
            pts.add((round(s[0], 3), round(s[1], 3)))
            pts.add((round(e[0], 3), round(e[1], 3)))
        if len(pts) < 3:
            return None
        hull = [[p[0], p[1]] for p in _convex_hull(list(pts))]

    # Compute bounding rectangle from the hull/outline
    hull_tuples = [(p[0], p[1]) for p in hull]
    convex = _convex_hull(hull_tuples)
    rect_corners, _area, rect_angle = _min_bounding_rect(convex)
    if not rect_corners or len(rect_corners) < 4:
        return None

    def _dist(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    side_a = _dist(rect_corners[0], rect_corners[1])
    side_b = _dist(rect_corners[1], rect_corners[2])
    rect_long = max(side_a, side_b)
    rect_short = min(side_a, side_b)

    cx = sum(c[0] for c in rect_corners) / 4.0
    cy = sum(c[1] for c in rect_corners) / 4.0

    return {
        "hull": hull,
        "rect_corners": rect_corners,
        "rect_angle": rect_angle,
        "rect_long": rect_long,
        "rect_short": rect_short,
        "center": [cx, cy],
    }


def _compute_room_rects(floor_plan_data):
    """Compute a bounding rectangle for each room's polygon.

    Returns a list of (label, rect_corners) tuples — one per room that has
    a valid polygon with at least 3 points.  Each rect_corners is a 4-point
    list suitable for passing to the roof builder functions.

    If no rooms have valid polygons, returns an empty list (the caller should
    fall back to the whole-building bounding rect).
    """
    rooms = floor_plan_data.get("rooms", [])
    result = []
    for room in rooms:
        poly = room.get("polygon", [])
        label = room.get("label", "room")
        if len(poly) < 3:
            continue
        pts = [(p[0], p[1]) for p in poly]
        hull = _convex_hull(pts)
        if len(hull) < 3:
            continue
        rect_corners, _area, _angle = _min_bounding_rect(hull)
        if rect_corners and len(rect_corners) >= 4:
            result.append((label, rect_corners))
    return result


def _simplify_to_rect(polygon, threshold=0.70):
    """Try to simplify a polygon to a rectangle if it's close enough.

    Pipeline:
    1. Compute convex hull; if polygon is >85% convex, use the hull
    2. Compute minimum bounding rectangle; if fill ratio > threshold, snap to rectangle
    3. Prefer axis-aligned rectangles when edge angle is close to 0/90 degrees
    """
    if len(polygon) <= 4:
        return polygon

    poly_area = _shoelace_area(polygon)
    if poly_area < 0.1:
        return polygon

    # Step 1: convex hull
    hull = _convex_hull(polygon)
    hull_area = _shoelace_area(hull)
    if hull_area < 0.1:
        return polygon

    convex_ratio = poly_area / hull_area
    working = hull if convex_ratio > 0.85 else polygon

    # Step 2: minimum bounding rectangle
    working_hull = _convex_hull(working)
    rect, rect_area, angle = _min_bounding_rect(working_hull)
    if rect is None or rect_area < 0.1:
        return working if len(working) < len(polygon) else polygon

    working_area = _shoelace_area(working)
    fill_ratio = working_area / rect_area

    if fill_ratio >= threshold:
        # Step 3: prefer axis-aligned if angle is close to 0 or 90 degrees
        norm_angle = angle % (math.pi / 2)
        if norm_angle < math.radians(5) or norm_angle > math.radians(85):
            # Snap to axis-aligned bounding box
            xs = [p[0] for p in working_hull]
            ys = [p[1] for p in working_hull]
            return [
                [round(min(xs), 3), round(min(ys), 3)],
                [round(max(xs), 3), round(min(ys), 3)],
                [round(max(xs), 3), round(max(ys), 3)],
                [round(min(xs), 3), round(max(ys), 3)],
            ]
        return rect

    # Return hull if it has fewer vertices
    if len(working) < len(polygon):
        return working
    return polygon


def _validated_polygon(polygon):
    """Clean and simplify a room polygon for floor/ceiling generation."""
    if len(polygon) < 3:
        return polygon

    # Try shapely first for robust validation
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
        from shapely.validation import make_valid
        sp = ShapelyPolygon(polygon)
        if not sp.is_valid:
            sp = make_valid(sp)
        if sp.geom_type == 'MultiPolygon':
            sp = max(sp.geoms, key=lambda g: g.area)
        if sp.geom_type == 'Polygon' and sp.area > 0:
            coords = list(sp.exterior.coords)[:-1]
            polygon = [[round(c[0], 3), round(c[1], 3)] for c in coords]
        else:
            # Validation produced degenerate geometry — fall back to convex hull
            sp = ShapelyPolygon(polygon).convex_hull
            if sp.geom_type == 'Polygon' and sp.area > 0:
                coords = list(sp.exterior.coords)[:-1]
                polygon = [[round(c[0], 3), round(c[1], 3)] for c in coords]
    except ImportError:
        pass

    polygon = _remove_close_duplicates(polygon, tol=0.1)
    polygon = _remove_collinear(polygon, tol=0.2)
    polygon = _simplify_to_rect(polygon, threshold=0.70)
    return polygon


def _expand_polygon(polygon, offset):
    """Expand polygon outward by *offset* metres to fill wall-thickness gaps."""
    if offset <= 0 or len(polygon) < 3:
        return polygon
    # Try Shapely for robust buffering on arbitrary polygons
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
        sp = ShapelyPolygon(polygon).buffer(offset, join_style=2)  # mitre join
        if sp.geom_type == 'Polygon' and sp.area > 0:
            coords = list(sp.exterior.coords)[:-1]
            return [[round(c[0], 3), round(c[1], 3)] for c in coords]
    except ImportError:
        pass
    # Fallback: expand corners from centroid (only reliable for simple shapes)
    if len(polygon) in (3, 4):
        return _expand_corners(polygon, offset)
    return polygon


def _get_wall_half_thickness(floor_plan_data):
    """Return half of the typical wall thickness from floor plan data."""
    walls = floor_plan_data.get("walls", [])
    if not walls:
        return 0.075  # default half of 0.15m
    thicknesses = [w.get("thickness", 0.15) for w in walls]
    median_t = sorted(thicknesses)[len(thicknesses) // 2]
    return median_t / 2.0


# ── Wall helpers ───────────────────────────────────────────────────────

def _wall_direction(start, end):
    """Return normalized direction and length between two 2D points."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return Vector((1, 0)), 0.0
    return Vector((dx / length, dy / length)), length


def _perpendicular_2d(direction):
    """Return the 2D perpendicular (rotated 90 degrees CCW)."""
    return Vector((-direction.y, direction.x))


# ── Boolean helper ────────────────────────────────────────────────────

def _apply_boolean_difference(target_obj, cutter_obj, mod_name):
    """Apply a boolean DIFFERENCE modifier from cutter_obj on target_obj.

    Handles the depsgraph update required in Blender 4.x/5.x for the
    boolean solver to see the cutter's world-space transform, tries
    EXACT solver first (most reliable), falls back to FAST, and cleans
    up the cutter object afterwards.

    Returns True on success, False on failure.
    """
    # Blender needs an up-to-date depsgraph before the boolean can see
    # the cutter's location/rotation.  Without this the cutter is
    # evaluated at the origin and the boolean silently does nothing.
    bpy.context.view_layer.update()

    mod = target_obj.modifiers.new(name=mod_name, type='BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.object = cutter_obj

    # Prefer EXACT (most reliable in Blender 4+/5+), fall back to FAST
    for solver in ('EXACT', 'FAST'):
        try:
            mod.solver = solver
            break
        except TypeError:
            continue

    bpy.context.view_layer.objects.active = target_obj
    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
    except RuntimeError:
        # Boolean failed — remove modifier and cutter, continue gracefully
        if mod.name in target_obj.modifiers:
            target_obj.modifiers.remove(mod)
        bpy.data.objects.remove(cutter_obj, do_unlink=True)
        return False

    bpy.data.objects.remove(cutter_obj, do_unlink=True)
    return True


# ── Multi-story alignment ─────────────────────────────────────────────

def _compute_wall_bbox_center(floor_plan_data):
    """Compute the bounding-box center of all walls."""
    xs, ys = [], []
    for w in floor_plan_data.get("walls", []):
        xs.extend([w["start"][0], w["end"][0]])
        ys.extend([w["start"][1], w["end"][1]])
    if not xs:
        return 0.0, 0.0
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    return cx, cy


def align_story_to_reference(story_data, ref_data):
    """Translate story_data so its bounding-box center matches ref_data's.

    When two floor plan images are parsed independently, their coordinate
    origins differ.  This shifts all coordinates in story_data (walls,
    doors, windows, rooms) so the building footprints are centred on top
    of each other.

    Works on a deep copy — does NOT mutate the input.
    Returns the shifted copy.
    """
    ref_cx, ref_cy = _compute_wall_bbox_center(ref_data)
    src_cx, src_cy = _compute_wall_bbox_center(story_data)

    dx = ref_cx - src_cx
    dy = ref_cy - src_cy

    # Skip if already aligned (within 5 cm)
    if abs(dx) < 0.05 and abs(dy) < 0.05:
        return story_data

    data = copy.deepcopy(story_data)

    # Shift walls
    for w in data.get("walls", []):
        w["start"] = [w["start"][0] + dx, w["start"][1] + dy]
        w["end"] = [w["end"][0] + dx, w["end"][1] + dy]

    # Shift doors
    for d in data.get("doors", []):
        if "position" in d:
            d["position"] = [d["position"][0] + dx, d["position"][1] + dy]

    # Shift windows
    for w in data.get("windows", []):
        if "position" in w:
            w["position"] = [w["position"][0] + dx, w["position"][1] + dy]

    # Shift room polygons
    for r in data.get("rooms", []):
        if "polygon" in r:
            r["polygon"] = [[p[0] + dx, p[1] + dy] for p in r["polygon"]]

    return data


# ── Output sanity filtering ───────────────────────────────────────────

def sanitize_floor_plan_data(floor_plan_data):
    """Filter out unreasonable detections from model output.

    When the CV model struggles with an image (e.g. a cropped upper story),
    it produces many low-confidence walls, phantom rooms (dozens of tiny
    "bathrooms"), and spurious door/window detections. This function applies
    several heuristics to clean up the output:

    1. Adaptive confidence filtering — if the wall count is unreasonably
       high relative to the building footprint area, progressively raise
       the confidence threshold until the density is reasonable.
    2. Room count cap — limit total rooms based on footprint area (roughly
       1 room per 15 sq m, minimum 3, maximum ~20 for very large plans).
    3. Small room filtering — limit the number of very small rooms (<4 sq m)
       to at most 3 (typically bathrooms/WC).
    4. Door/window sanity — remove doors and windows whose wall_index points
       to a wall that was removed.

    Works on a deep copy — does NOT mutate the input.
    """
    import math as _math

    data = copy.deepcopy(floor_plan_data)
    walls = data.get("walls", [])
    doors = data.get("doors", [])
    windows = data.get("windows", [])
    rooms = data.get("rooms", [])

    if not walls:
        return data

    # ── 1. Adaptive confidence filtering ─────────────────────────────
    # Compute building footprint area from wall bounding box
    xs = [w["start"][0] for w in walls] + [w["end"][0] for w in walls]
    ys = [w["start"][1] for w in walls] + [w["end"][1] for w in walls]
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)
    footprint_area = max(bbox_w * bbox_h, 1.0)  # sq meters

    # Reasonable wall count: residential plans rarely exceed 80 walls
    # even for large houses. If we have significantly more, the model
    # is detecting noise. Use both absolute and density-based caps.
    MAX_WALLS_ABSOLUTE = 100  # hard cap
    MAX_DENSITY = 0.5  # walls per sq m (lowered from 0.8)
    target_by_density = int(footprint_area * MAX_DENSITY)
    target_count = min(MAX_WALLS_ABSOLUTE, max(target_by_density, 30))

    if len(walls) > target_count:
            # Sort by confidence descending, keep top target_count
            indexed = sorted(enumerate(walls),
                             key=lambda x: x[1].get("confidence", 0.5),
                             reverse=True)
            keep_indices = set(idx for idx, _ in indexed[:target_count])

            # Build old→new index map for door/window remapping
            old_to_new = {}
            new_walls = []
            for i, w in enumerate(walls):
                if i in keep_indices:
                    old_to_new[i] = len(new_walls)
                    new_walls.append(w)

            # Remap door/window wall_index, drop those pointing to removed walls
            new_doors = []
            for d in doors:
                old_idx = d.get("wall_index", 0)
                if old_idx in old_to_new:
                    d["wall_index"] = old_to_new[old_idx]
                    new_doors.append(d)
                else:
                    # Find nearest surviving wall
                    nearest = _find_nearest_wall_idx(
                        d.get("position", [0, 0]), new_walls)
                    if nearest is not None:
                        d["wall_index"] = nearest
                        new_doors.append(d)

            new_windows = []
            for w in windows:
                old_idx = w.get("wall_index", 0)
                if old_idx in old_to_new:
                    w["wall_index"] = old_to_new[old_idx]
                    new_windows.append(w)
                else:
                    nearest = _find_nearest_wall_idx(
                        w.get("position", [0, 0]), new_walls)
                    if nearest is not None:
                        w["wall_index"] = nearest
                        new_windows.append(w)

            walls = new_walls
            doors = new_doors
            windows = new_windows

    # ── 2. Filter low-confidence doors and windows ───────────────────
    # Remove doors/windows with very low confidence (likely false positives)
    MIN_DOOR_CONF = 0.35
    MIN_WIN_CONF = 0.35
    doors = [d for d in doors if d.get("confidence", 0.5) >= MIN_DOOR_CONF]
    windows = [w for w in windows if w.get("confidence", 0.5) >= MIN_WIN_CONF]

    # Cap door/window count — residential plans rarely have >15 of either
    MAX_DOORS = 15
    MAX_WINDOWS = 15
    if len(doors) > MAX_DOORS:
        doors.sort(key=lambda d: d.get("confidence", 0.5), reverse=True)
        doors = doors[:MAX_DOORS]
    if len(windows) > MAX_WINDOWS:
        windows.sort(key=lambda w: w.get("confidence", 0.5), reverse=True)
        windows = windows[:MAX_WINDOWS]

    # ── 2b. Geometric validation: reassign & validate doors/windows ──
    # The model often assigns doors/windows to the wrong wall. Re-assign
    # each to its actual nearest wall, then drop any that are too far
    # from any wall (likely false positives).
    MAX_DIST_TO_WALL = 1.5  # meters — if further than this, it's bogus

    validated_doors = []
    for d in doors:
        pos = d.get("position", [0, 0])
        nearest_idx, dist = _find_nearest_wall_with_dist(pos, walls)
        if nearest_idx is not None and dist <= MAX_DIST_TO_WALL:
            d["wall_index"] = nearest_idx
            validated_doors.append(d)
    doors = validated_doors

    validated_windows = []
    for w in windows:
        pos = w.get("position", [0, 0])
        nearest_idx, dist = _find_nearest_wall_with_dist(pos, walls)
        if nearest_idx is not None and dist <= MAX_DIST_TO_WALL:
            w["wall_index"] = nearest_idx
            validated_windows.append(w)
    windows = validated_windows

    # ── 3. Room count sanity check ───────────────────────────────────
    # Split oversized rooms that are clearly multiple rooms merged
    MAX_ROOM_RATIO = 0.35
    if rooms:
        split_rooms = []
        for r in rooms:
            area = r.get("area", 0)
            if area > footprint_area * MAX_ROOM_RATIO and area > 40.0:
                split_rooms.extend(_split_oversized_room(r))
            else:
                split_rooms.append(r)
        rooms = split_rooms

    # Cap rooms based on footprint area (~1 room per 20 sq m, max 12)
    if rooms:
        max_rooms = max(3, min(12, int(footprint_area / 20.0) + 2))

        # Also limit very small rooms (< 4 sq m) to at most 3
        small_rooms = [r for r in rooms if r.get("area", 0) < 4.0]
        normal_rooms = [r for r in rooms if r.get("area", 0) >= 4.0]

        if len(small_rooms) > 4:
            # Keep only the 4 largest small rooms
            small_rooms.sort(key=lambda r: r.get("area", 0), reverse=True)
            small_rooms = small_rooms[:4]

        rooms = normal_rooms + small_rooms
        rooms.sort(key=lambda r: r.get("area", 0), reverse=True)

        if len(rooms) > max_rooms:
            rooms = rooms[:max_rooms]

        # Re-label rooms after filtering (the labels may have gaps now)
        rooms = _relabel_rooms(rooms)

    data["walls"] = walls
    data["doors"] = doors
    data["windows"] = windows
    data["rooms"] = rooms

    return data


def _find_nearest_wall_idx(position, walls):
    """Find index of the nearest wall to a position [x, y]."""
    if not walls or not position:
        return None
    idx, _ = _find_nearest_wall_with_dist(position, walls)
    return idx


def _find_nearest_wall_with_dist(position, walls):
    """Find index of the nearest wall and distance to it.

    Returns (index, distance).  Returns (None, inf) if walls is empty.
    """
    if not walls or not position:
        return None, float("inf")
    px, py = position[0], position[1]
    best_idx = 0
    best_dist = float("inf")
    for i, w in enumerate(walls):
        sx, sy = w["start"]
        ex, ey = w["end"]
        # Point-to-segment distance
        dx, dy = ex - sx, ey - sy
        lsq = dx * dx + dy * dy
        if lsq < 1e-10:
            dist = ((px - sx) ** 2 + (py - sy) ** 2) ** 0.5
        else:
            t = max(0, min(1, ((px - sx) * dx + (py - sy) * dy) / lsq))
            cx, cy = sx + t * dx, sy + t * dy
            dist = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx, best_dist


def _split_oversized_room(room):
    """Split an oversized room into two halves along its longer bounding-box axis."""
    polygon = room.get("polygon", [])
    if len(polygon) < 3:
        return [room]

    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w, h = max_x - min_x, max_y - min_y

    if w >= h:
        mid = (min_x + max_x) / 2
        left = [[min_x, min_y], [mid, min_y], [mid, max_y], [min_x, max_y]]
        right = [[mid, min_y], [max_x, min_y], [max_x, max_y], [mid, max_y]]
        area_l, area_r = (mid - min_x) * h, (max_x - mid) * h
    else:
        mid = (min_y + max_y) / 2
        left = [[min_x, min_y], [max_x, min_y], [max_x, mid], [min_x, mid]]
        right = [[min_x, mid], [max_x, mid], [max_x, max_y], [min_x, max_y]]
        area_l, area_r = w * (mid - min_y), w * (max_y - mid)

    return [
        {"label": "room", "polygon": left, "area": round(area_l, 2)},
        {"label": "room", "polygon": right, "area": round(area_r, 2)},
    ]


def _relabel_rooms(rooms):
    """Re-assign room labels using area-based heuristics.

    Similar to the inference model's _label_rooms, but simplified for
    post-filtering when rooms have already been reduced.
    """
    import math as _math

    if not rooms:
        return rooms

    n = len(rooms)
    labels = [None] * n

    # Pre-compute geometry
    aspects = []
    narrow_dims = []
    for room in rooms:
        polygon = room.get("polygon", [])
        if polygon:
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            bb_w = max(xs) - min(xs) if xs else 1
            bb_h = max(ys) - min(ys) if ys else 1
        else:
            bb_w = bb_h = 1
        aspect = max(bb_w, bb_h) / max(min(bb_w, bb_h), 0.1)
        narrow = min(bb_w, bb_h)
        aspects.append(aspect)
        narrow_dims.append(narrow)

    # Pass 1: hallways (high aspect ratio, narrow)
    hall_count = 0
    for i in range(n):
        if aspects[i] > 3.0 and narrow_dims[i] < 4.0:
            hall_count += 1
            labels[i] = "hallway" if hall_count == 1 else f"hallway_{hall_count}"

    # Pass 2: living room (largest non-hallway)
    remaining = [(i, rooms[i].get("area", 0)) for i in range(n)
                 if labels[i] is None]
    remaining.sort(key=lambda x: x[1], reverse=True)
    if remaining:
        labels[remaining[0][0]] = "living_room"
        remaining = remaining[1:]

    # Pass 3: kitchen (one medium room, 10-35 sq m)
    kitchen_candidates = [(i, a) for i, a in remaining if 10 <= a <= 35]
    if kitchen_candidates:
        best = max(kitchen_candidates, key=lambda x: x[1])
        labels[best[0]] = "kitchen"

    # Pass 4: bathrooms / WC (small rooms < 8 sq m)
    bath_count = 0
    bath_labels = ["bathroom", "wc", "utility"]
    remaining = [(i, rooms[i].get("area", 0)) for i in range(n)
                 if labels[i] is None]
    remaining.sort(key=lambda x: x[1])  # smallest first
    for i, area in remaining:
        if area < 8:
            if bath_count < len(bath_labels):
                labels[i] = bath_labels[bath_count]
            else:
                labels[i] = f"bathroom_{bath_count}"
            bath_count += 1

    # Pass 5: bedrooms
    bed_count = 0
    for i in range(n):
        if labels[i] is None:
            bed_count += 1
            labels[i] = "bedroom" if bed_count == 1 else f"bedroom_{bed_count}"

    for i, room in enumerate(rooms):
        room["label"] = labels[i]

    return rooms


# ── Wall deduplication ────────────────────────────────────────────────

def deduplicate_walls(floor_plan_data):
    """Remove overlapping and near-duplicate walls from floor plan data.

    IMPORTANT: This function works on a deep copy of the input and returns
    the copy.  The original dict is never mutated, which is critical for
    multi-story builds where multiple stories may share the same source dict.

    The inference pipeline often detects both inner and outer edges of the
    same physical wall, and also detects a long wall *plus* its sub-segments.
    This produces double geometry and hides boolean cutouts behind the twin.

    Strategy:
    1. For each pair of collinear walls within one wall-thickness distance,
       merge them into a single wall at the midpoint.
    2. When a short wall is fully contained within a longer collinear wall,
       keep only the longer one and remap door/window indices.

    Returns a (possibly modified) copy — the original is never mutated.
    """
    floor_plan_data = copy.deepcopy(floor_plan_data)

    walls = floor_plan_data.get("walls", [])
    if len(walls) < 2:
        return floor_plan_data

    PARALLEL_TOL = 0.4   # max lateral distance to consider walls "same"
    OVERLAP_MIN_FAR = 0.3   # min overlap for walls 0.2-0.4m apart
    OVERLAP_MIN_CLOSE = 0.15  # min overlap for walls <0.2m apart (inner/outer face)

    # Classify walls as vertical or horizontal (or diagonal)
    keep = [True] * len(walls)
    merged_into = list(range(len(walls)))  # index remapping for doors/windows

    for i in range(len(walls)):
        if not keep[i]:
            continue
        si, ei = walls[i]["start"], walls[i]["end"]
        di, li = _wall_direction(Vector(si), Vector(ei))

        for j in range(i + 1, len(walls)):
            if not keep[j]:
                continue
            sj, ej = walls[j]["start"], walls[j]["end"]
            dj, lj = _wall_direction(Vector(sj), Vector(ej))

            # Check if walls are roughly parallel (dot product near +/-1)
            dot = abs(di.x * dj.x + di.y * dj.y)
            if dot < 0.95:
                continue

            # Project both walls onto the longer wall's axis
            # Check lateral distance (perpendicular offset)
            perp_i = _perpendicular_2d(di)
            lateral = abs(perp_i.x * (sj[0] - si[0]) + perp_i.y * (sj[1] - si[1]))
            if lateral > PARALLEL_TOL:
                continue

            # Use tighter overlap threshold for very close walls (inner/outer face)
            overlap_min = OVERLAP_MIN_CLOSE if lateral < 0.2 else OVERLAP_MIN_FAR

            # Check overlap along the wall direction
            def _project(pt, origin, direction):
                return direction.x * (pt[0] - origin[0]) + direction.y * (pt[1] - origin[1])

            pi_s = 0.0
            pi_e = li
            pj_s = _project(sj, si, di)
            pj_e = _project(ej, si, di)

            min_j, max_j = min(pj_s, pj_e), max(pj_s, pj_e)
            overlap = min(pi_e, max_j) - max(pi_s, min_j)

            if overlap < overlap_min:
                continue

            # These walls overlap — keep the longer one
            if li >= lj:
                keep[j] = False
                merged_into[j] = i
            else:
                keep[i] = False
                merged_into[i] = j
                break  # wall i is gone, stop comparing it

    # Build new wall list and index remap
    old_to_new = {}
    new_walls = []
    for i, wall in enumerate(walls):
        if keep[i]:
            old_to_new[i] = len(new_walls)
            new_walls.append(wall)

    # Map merged walls to the surviving wall's new index
    for i in range(len(walls)):
        if not keep[i]:
            target = merged_into[i]
            # Follow chain
            while not keep[target]:
                target = merged_into[target]
            old_to_new[i] = old_to_new[target]

    # Remap door/window wall_index references
    for door in floor_plan_data.get("doors", []):
        old_idx = door.get("wall_index", 0)
        if old_idx < len(walls):
            door["wall_index"] = old_to_new.get(old_idx, 0)

    for win in floor_plan_data.get("windows", []):
        old_idx = win.get("wall_index", 0)
        if old_idx < len(walls):
            win["wall_index"] = old_to_new.get(old_idx, 0)

    removed = len(walls) - len(new_walls)
    if removed > 0:
        floor_plan_data["walls"] = new_walls

    return floor_plan_data


# ── Geometry generators ───────────────────────────────────────────────

def generate_walls(floor_plan_data, collection, wall_height,
                    z_offset=0.0, name_prefix=""):
    """Generate wall meshes from floor plan data.

    Each wall is a rectangular box defined by start/end points, thickness, and height.
    Returns the number of walls generated.
    """
    from . import materials

    count = 0
    walls = floor_plan_data.get("walls", [])
    for i, wall_data in enumerate(walls):
        start = Vector(wall_data["start"])
        end = Vector(wall_data["end"])
        thickness = wall_data.get("thickness", 0.15)

        direction, length = _wall_direction(start, end)
        if length < 1e-6:
            continue

        perp = _perpendicular_2d(direction)
        half_t = thickness / 2.0

        # Four corners of the wall base
        corners = [
            Vector((start.x - perp.x * half_t, start.y - perp.y * half_t, z_offset)),
            Vector((start.x + perp.x * half_t, start.y + perp.y * half_t, z_offset)),
            Vector((end.x + perp.x * half_t, end.y + perp.y * half_t, z_offset)),
            Vector((end.x - perp.x * half_t, end.y - perp.y * half_t, z_offset)),
        ]

        mesh = bpy.data.meshes.new(f"{name_prefix}Wall_{i}")
        bm = bmesh.new()

        # Create base face
        verts = [bm.verts.new(c) for c in corners]
        bm.faces.new(verts)

        # Extrude upward
        result = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
        extruded_verts = [v for v in result["geom"] if isinstance(v, bmesh.types.BMVert)]
        bmesh.ops.translate(bm, vec=Vector((0, 0, wall_height)), verts=extruded_verts)

        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        obj = bpy.data.objects.new(f"{name_prefix}Wall_{i}", mesh)
        obj["fp3d_type"] = "wall"
        obj["fp3d_wall_index"] = i
        obj["fp3d_original_height"] = wall_height
        _link_to_collection(obj, collection)

        materials.assign_wall_material(obj)
        count += 1

    return count


def generate_door_openings(floor_plan_data, collection, wall_height,
                           z_offset=0.0, name_prefix=""):
    """Cut door openings into walls using boolean modifiers.

    Doors are full-height openings (default 2.1m) cut from the bottom of the wall.
    Returns the number of door openings created.
    """
    count = 0
    doors = floor_plan_data.get("doors", [])
    walls = floor_plan_data.get("walls", [])

    for i, door_data in enumerate(doors):
        wall_idx = door_data.get("wall_index", 0)
        # wall_index=-1 is the schema sentinel for "not attached to a wall";
        # a one-sided `>= len` check let it fall through to walls[-1].
        if wall_idx < 0 or wall_idx >= len(walls):
            continue

        wall_data = walls[wall_idx]
        start = Vector(wall_data["start"])
        end = Vector(wall_data["end"])
        thickness = wall_data.get("thickness", 0.15)

        direction, wall_length = _wall_direction(start, end)
        if wall_length < 1e-6:
            continue


        # Door position along wall (distance from start)
        door_pos = door_data.get("position", [0, 0])
        if isinstance(door_pos, (list, tuple)) and len(door_pos) == 2:
            dp = Vector(door_pos) - start
            dist_along = dp.dot(direction)
        else:
            dist_along = float(door_pos)

        door_width = door_data.get("width", 0.9)
        door_height = door_data.get("height", 2.1)

        # Clamp door position to within wall bounds (with half-width margin)
        half_w_door = door_width / 2.0
        dist_along = max(half_w_door, min(wall_length - half_w_door, dist_along))

        # Skip if door is wider than wall
        if door_width >= wall_length:
            continue

        center_along = dist_along
        half_w = door_width / 2.0

        cutter_center = (
            start.x + direction.x * center_along,
            start.y + direction.y * center_along,
            z_offset + door_height / 2.0,
        )

        mesh = bpy.data.meshes.new(f"{name_prefix}DoorCutter_{i}")
        bm = bmesh.new()

        margin = thickness * 3.0
        corners = [
            Vector((-half_w, -margin, -door_height / 2.0)),
            Vector((half_w, -margin, -door_height / 2.0)),
            Vector((half_w, margin, -door_height / 2.0)),
            Vector((-half_w, margin, -door_height / 2.0)),
        ]
        verts = [bm.verts.new(c) for c in corners]
        bm.faces.new(verts)
        result = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
        extruded_verts = [v for v in result["geom"] if isinstance(v, bmesh.types.BMVert)]
        bmesh.ops.translate(bm, vec=Vector((0, 0, door_height)), verts=extruded_verts)

        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        cutter_obj = bpy.data.objects.new(f"{name_prefix}DoorCutter_{i}", mesh)
        cutter_obj.location = Vector(cutter_center)

        angle = math.atan2(direction.y, direction.x)
        cutter_obj.rotation_euler.z = angle

        _link_to_collection(cutter_obj, collection)

        wall_obj_name = f"{name_prefix}Wall_{wall_idx}"
        wall_obj = bpy.data.objects.get(wall_obj_name)
        if wall_obj:
            if _apply_boolean_difference(wall_obj, cutter_obj, f"Door_{i}"):
                count += 1

    return count


def generate_door_panels(floor_plan_data, collection, wall_height,
                         z_offset=0.0, name_prefix=""):
    """Place visible door panel meshes inside each door opening.

    After the boolean cutter has removed wall material, this creates a thin
    coloured panel (like a real door leaf) sitting in the opening so users
    can clearly see where doors are.  Returns the number of panels created.
    """
    from . import materials

    count = 0
    doors = floor_plan_data.get("doors", [])
    walls = floor_plan_data.get("walls", [])

    for i, door_data in enumerate(doors):
        wall_idx = door_data.get("wall_index", 0)
        # wall_index=-1 is the schema sentinel for "not attached to a wall";
        # a one-sided `>= len` check let it fall through to walls[-1].
        if wall_idx < 0 or wall_idx >= len(walls):
            continue

        wall_data = walls[wall_idx]
        start = Vector(wall_data["start"])
        end = Vector(wall_data["end"])

        direction, wall_length = _wall_direction(start, end)
        if wall_length < 1e-6:
            continue


        # Door position along wall
        door_pos = door_data.get("position", [0, 0])
        if isinstance(door_pos, (list, tuple)) and len(door_pos) == 2:
            dp = Vector(door_pos) - start
            dist_along = dp.dot(direction)
        else:
            dist_along = float(door_pos)

        door_width = door_data.get("width", 0.9)
        door_height = door_data.get("height", 2.1)
        panel_thickness = 0.04  # ~4 cm thick door leaf

        # Centre of the door panel
        cx = start.x + direction.x * dist_along
        cy = start.y + direction.y * dist_along
        cz = z_offset + door_height / 2.0

        # Build a thin box for the door leaf via bmesh
        mesh = bpy.data.meshes.new(f"{name_prefix}DoorPanel_{i}")
        bm = bmesh.new()

        half_w = door_width / 2.0
        half_t = panel_thickness / 2.0
        half_h = door_height / 2.0

        # Local-space box (will be rotated to align with wall)
        corners_bot = [
            Vector((-half_w, -half_t, -half_h)),
            Vector(( half_w, -half_t, -half_h)),
            Vector(( half_w,  half_t, -half_h)),
            Vector((-half_w,  half_t, -half_h)),
        ]
        verts_bot = [bm.verts.new(c) for c in corners_bot]
        bm.faces.new(verts_bot)
        result = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
        extruded_verts = [v for v in result["geom"]
                          if isinstance(v, bmesh.types.BMVert)]
        bmesh.ops.translate(bm, vec=Vector((0, 0, door_height)),
                            verts=extruded_verts)

        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        obj = bpy.data.objects.new(f"{name_prefix}DoorPanel_{i}", mesh)
        obj.location = Vector((cx, cy, cz))
        angle = math.atan2(direction.y, direction.x)
        obj.rotation_euler.z = angle

        _link_to_collection(obj, collection)

        obj["fp3d_type"] = "door_panel"

        materials.assign_door_material(obj, door_data.get("type"))
        count += 1

    return count


def generate_window_openings(floor_plan_data, collection, wall_height,
                             z_offset=0.0, name_prefix=""):
    """Cut window openings into walls.

    Windows are openings at a given sill height (default 0.9m) with a
    default height of 1.2m.
    Returns the number of window openings created.
    """
    count = 0
    windows = floor_plan_data.get("windows", [])
    walls = floor_plan_data.get("walls", [])

    for i, win_data in enumerate(windows):
        wall_idx = win_data.get("wall_index", 0)
        # wall_index=-1 is the schema sentinel for "not attached to a wall";
        # a one-sided `>= len` check let it fall through to walls[-1].
        if wall_idx < 0 or wall_idx >= len(walls):
            continue

        wall_data = walls[wall_idx]
        start = Vector(wall_data["start"])
        end = Vector(wall_data["end"])
        thickness = wall_data.get("thickness", 0.15)

        direction, wall_length = _wall_direction(start, end)
        if wall_length < 1e-6:
            continue


        win_pos = win_data.get("position", [0, 0])
        if isinstance(win_pos, (list, tuple)) and len(win_pos) == 2:
            dp = Vector(win_pos) - start
            dist_along = dp.dot(direction)
        else:
            dist_along = float(win_pos)

        win_width = win_data.get("width", 1.2)
        win_height = win_data.get("height", 1.2)
        sill_height = win_data.get("sill_height", 0.9)
        full_wall = win_data.get("full_wall", False)

        # Full-wall window: span the entire wall minus a small structural margin
        if full_wall or win_width >= wall_length * 0.95:
            edge_margin = min(thickness, 0.1)  # keep a sliver at each end
            win_width = wall_length - edge_margin * 2
            center_along = wall_length / 2.0
            dist_along = center_along
            # Floor-to-ceiling style: lower sill, taller window
            if full_wall:
                sill_height = win_data.get("sill_height", 0.1)
                win_height = win_data.get("height", wall_height - sill_height - 0.1)

        # Clamp window position within wall bounds
        half_w_win = win_width / 2.0
        dist_along = max(half_w_win, min(wall_length - half_w_win, dist_along))

        # Skip if window is wider than wall
        if win_width >= wall_length:
            continue

        center_along = dist_along
        half_w = win_width / 2.0

        cutter_center = (
            start.x + direction.x * center_along,
            start.y + direction.y * center_along,
            z_offset + sill_height + win_height / 2.0,
        )

        mesh = bpy.data.meshes.new(f"{name_prefix}WindowCutter_{i}")
        bm = bmesh.new()

        margin = thickness * 3.0
        corners = [
            Vector((-half_w, -margin, -win_height / 2.0)),
            Vector((half_w, -margin, -win_height / 2.0)),
            Vector((half_w, margin, -win_height / 2.0)),
            Vector((-half_w, margin, -win_height / 2.0)),
        ]
        verts = [bm.verts.new(c) for c in corners]
        bm.faces.new(verts)
        result = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
        extruded_verts = [v for v in result["geom"] if isinstance(v, bmesh.types.BMVert)]
        bmesh.ops.translate(bm, vec=Vector((0, 0, win_height)), verts=extruded_verts)

        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        cutter_obj = bpy.data.objects.new(f"{name_prefix}WindowCutter_{i}", mesh)
        cutter_obj.location = Vector(cutter_center)

        angle = math.atan2(direction.y, direction.x)
        cutter_obj.rotation_euler.z = angle

        _link_to_collection(cutter_obj, collection)

        wall_obj_name = f"{name_prefix}Wall_{wall_idx}"
        wall_obj = bpy.data.objects.get(wall_obj_name)
        if wall_obj:
            if _apply_boolean_difference(wall_obj, cutter_obj, f"Window_{i}"):
                count += 1

    return count


def generate_window_panes(floor_plan_data, collection, wall_height,
                          z_offset=0.0, name_prefix=""):
    """Place thin glass pane meshes inside each window opening.

    Similar to generate_door_panels() but for windows.  Creates a thin
    translucent panel sitting in the window opening so users can see glass.
    Handles both regular windows and full-wall windows.
    Returns the number of panes created.
    """
    from . import materials

    count = 0
    windows = floor_plan_data.get("windows", [])
    walls = floor_plan_data.get("walls", [])

    for i, win_data in enumerate(windows):
        wall_idx = win_data.get("wall_index", 0)
        # wall_index=-1 is the schema sentinel for "not attached to a wall";
        # a one-sided `>= len` check let it fall through to walls[-1].
        if wall_idx < 0 or wall_idx >= len(walls):
            continue

        wall_data = walls[wall_idx]
        start = Vector(wall_data["start"])
        end = Vector(wall_data["end"])
        thickness = wall_data.get("thickness", 0.15)

        direction, wall_length = _wall_direction(start, end)
        if wall_length < 1e-6:
            continue

        win_pos = win_data.get("position", [0, 0])
        if isinstance(win_pos, (list, tuple)) and len(win_pos) == 2:
            dp = Vector(win_pos) - start
            dist_along = dp.dot(direction)
        else:
            dist_along = float(win_pos)

        win_width = win_data.get("width", 1.2)
        win_height = win_data.get("height", 1.2)
        sill_height = win_data.get("sill_height", 0.9)
        full_wall = win_data.get("full_wall", False)

        # Match the same sizing logic as generate_window_openings
        if full_wall or win_width >= wall_length * 0.95:
            edge_margin = min(thickness, 0.1)
            win_width = wall_length - edge_margin * 2
            dist_along = wall_length / 2.0
            if full_wall:
                sill_height = win_data.get("sill_height", 0.1)
                win_height = win_data.get("height", wall_height - sill_height - 0.1)

        # Centre of the glass pane
        cx = start.x + direction.x * dist_along
        cy = start.y + direction.y * dist_along
        cz = z_offset + sill_height + win_height / 2.0

        # Build a flat plane (single face, no depth) to avoid z-fighting
        mesh = bpy.data.meshes.new(f"{name_prefix}WindowPane_{i}")
        bm = bmesh.new()

        half_w = win_width / 2.0
        half_h = win_height / 2.0

        # Flat quad in local XZ plane (Y=0), rotated to align with wall later
        corners = [
            Vector((-half_w, 0, -half_h)),
            Vector(( half_w, 0, -half_h)),
            Vector(( half_w, 0,  half_h)),
            Vector((-half_w, 0,  half_h)),
        ]
        verts = [bm.verts.new(c) for c in corners]
        bm.faces.new(verts)

        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        obj = bpy.data.objects.new(f"{name_prefix}WindowPane_{i}", mesh)
        obj.location = Vector((cx, cy, cz))
        angle = math.atan2(direction.y, direction.x)
        obj.rotation_euler.z = angle

        _link_to_collection(obj, collection)

        obj["fp3d_type"] = "window_pane"

        materials.assign_window_glass_material(obj)
        count += 1

    return count


def generate_floors(floor_plan_data, collection, z_offset=0.0, name_prefix=""):
    """Generate floor planes from room polygons. Returns the number of floors generated."""
    from . import materials

    half_t = _get_wall_half_thickness(floor_plan_data)
    count = 0
    rooms = floor_plan_data.get("rooms", [])
    for i, room_data in enumerate(rooms):
        polygon = room_data.get("polygon", [])
        if len(polygon) < 3:
            continue

        label = room_data.get("label", f"Room_{i}")

        # Simplify polygon then expand to fill wall-thickness gaps
        polygon = _validated_polygon(polygon)
        if len(polygon) < 3:
            continue
        polygon = _expand_polygon(polygon, half_t)
        polygon = _ensure_ccw(polygon)

        mesh = bpy.data.meshes.new(f"{name_prefix}Floor_{label}_{i}")
        bm = bmesh.new()

        verts = [bm.verts.new(Vector((p[0], p[1], z_offset))) for p in polygon]
        try:
            bm.faces.new(verts)
        except ValueError:
            bm.free()
            continue

        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        obj = bpy.data.objects.new(f"{name_prefix}Floor_{label}_{i}", mesh)
        obj["fp3d_type"] = "floor"
        obj["fp3d_room_label"] = label
        _link_to_collection(obj, collection)

        materials.assign_floor_material(obj, room_label=label)
        count += 1

    return count


def generate_ceilings(floor_plan_data, collection, wall_height,
                      z_offset=0.0, name_prefix=""):
    """Generate ceiling planes from room polygons. Returns the number of ceilings generated."""
    from . import materials

    half_t = _get_wall_half_thickness(floor_plan_data)
    count = 0
    rooms = floor_plan_data.get("rooms", [])
    for i, room_data in enumerate(rooms):
        polygon = room_data.get("polygon", [])
        if len(polygon) < 3:
            continue

        label = room_data.get("label", f"Room_{i}")

        polygon = _validated_polygon(polygon)
        if len(polygon) < 3:
            continue
        polygon = _expand_polygon(polygon, half_t)
        polygon = _ensure_ccw(polygon)

        mesh = bpy.data.meshes.new(f"{name_prefix}Ceiling_{label}_{i}")
        bm = bmesh.new()

        verts = [bm.verts.new(Vector((p[0], p[1], z_offset + wall_height))) for p in polygon]
        try:
            face = bm.faces.new(verts)
            face.normal_flip()
        except ValueError:
            bm.free()
            continue

        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        obj = bpy.data.objects.new(f"{name_prefix}Ceiling_{label}_{i}", mesh)
        obj["fp3d_type"] = "ceiling"
        obj["fp3d_room_label"] = label
        _link_to_collection(obj, collection)

        materials.assign_ceiling_material(obj)
        count += 1

    return count


def generate_room_labels(floor_plan_data, collection, z_offset=0.0, name_prefix=""):
    """Create text objects for room labels positioned at room centroids.
    Returns the number of labels generated.
    """
    count = 0
    rooms = floor_plan_data.get("rooms", [])
    for i, room_data in enumerate(rooms):
        label = room_data.get("label", f"Room_{i}")
        polygon = room_data.get("polygon", [])
        if len(polygon) < 3:
            continue

        # Calculate centroid
        cx = sum(p[0] for p in polygon) / len(polygon)
        cy = sum(p[1] for p in polygon) / len(polygon)

        # Create text object
        font_curve = bpy.data.curves.new(name=f"{name_prefix}Label_{label}_{i}", type='FONT')
        font_curve.body = label.replace("_", " ").title()
        font_curve.size = 0.3
        font_curve.align_x = 'CENTER'
        font_curve.align_y = 'CENTER'

        obj = bpy.data.objects.new(f"{name_prefix}Label_{label}_{i}", font_curve)
        obj.location = Vector((cx, cy, z_offset + 0.01))  # Slightly above floor
        obj.rotation_euler.x = 0  # Flat on the floor
        obj["fp3d_type"] = "label"
        obj["fp3d_room_label"] = label
        _link_to_collection(obj, collection)

        area = room_data.get("area")
        if area is not None:
            obj["fp3d_room_area"] = area
        count += 1

    return count


# ── Staircase generation ─────────────────────────────────────────────


def _find_stairwell_rooms(floor_plan_data):
    """Find rooms labeled as stairwells. Returns list of room dicts."""
    stair_keywords = {"stair", "stairs", "stairwell", "staircase", "stairway"}
    result = []
    for room in floor_plan_data.get("rooms", []):
        label = room.get("label", "").lower().replace("_", " ")
        if any(kw in label for kw in stair_keywords):
            result.append(room)
    return result


def _find_best_stair_candidate(floor_plan_data):
    """When no explicit stairwell room exists, find the best room to place
    stairs (typically the largest hallway).  Returns (cx, cy, run_dx, run_dy,
    run_length, width) or None.
    """
    hallway_keywords = {"hallway", "hall", "corridor", "foyer", "entry",
                        "landing", "passage", "vestibule"}
    candidates = []
    for room in floor_plan_data.get("rooms", []):
        label = room.get("label", "").lower().replace("_", " ")
        if any(kw in label for kw in hallway_keywords):
            poly = room.get("polygon", [])
            if len(poly) >= 3:
                area = room.get("area", 0)
                if area == 0:
                    # Rough shoelace
                    n = len(poly)
                    area = abs(sum(poly[i][0]*poly[(i+1)%n][1] -
                                   poly[(i+1)%n][0]*poly[i][1]
                                   for i in range(n))) / 2
                candidates.append((area, room))

    if not candidates:
        return None

    # Pick the largest hallway
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_room = candidates[0][1]
    center, run_dir, run_len, sw_width = _stairwell_center_and_run(best_room)
    return center[0], center[1], run_dir[0], run_dir[1], run_len, sw_width


def _compute_building_centroid(floor_plan_data):
    """Compute the centroid of all room polygons as a fallback position."""
    all_x, all_y = [], []
    for room in floor_plan_data.get("rooms", []):
        for pt in room.get("polygon", []):
            all_x.append(pt[0])
            all_y.append(pt[1])
    if not all_x:
        # Try walls as a last resort
        for wall in floor_plan_data.get("walls", []):
            s = wall.get("start", [0, 0])
            e = wall.get("end", [0, 0])
            all_x.extend([s[0], e[0]])
            all_y.extend([s[1], e[1]])
    if all_x:
        return sum(all_x) / len(all_x), sum(all_y) / len(all_y)
    return 0.0, 0.0


def _stairwell_center_and_run(room):
    """Compute centroid and run direction from a stairwell room polygon.

    Returns:
        (cx, cy): centroid
        run_dx, run_dy: unit vector along the long axis
        run_length: length of the bounding box along the long axis
        width: length of the bounding box along the short axis
    """
    polygon = room.get("polygon", [])
    if len(polygon) < 3:
        return (0, 0), (1, 0), 3.0, 1.0

    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    bb_w = max(xs) - min(xs)
    bb_h = max(ys) - min(ys)

    if bb_w >= bb_h:
        return (cx, cy), (1, 0), bb_w, bb_h
    else:
        return (cx, cy), (0, 1), bb_h, bb_w


def _generate_stair_mesh(bm, cx, cy, z_bottom, z_top, run_dx, run_dy,
                         run_length, stair_width):
    """Generate a straight-run staircase into the given bmesh.

    Args:
        bm: bmesh to populate.
        cx, cy: center of the stairwell (bottom of run).
        z_bottom: elevation of the lower floor.
        z_top: elevation of the upper floor.
        run_dx, run_dy: unit direction vector for the stair run.
        run_length: total horizontal distance available.
        stair_width: width of the stair perpendicular to run direction.
    """
    riser_h = 0.18  # standard riser height
    tread_d = 0.28  # standard tread depth

    height = z_top - z_bottom
    num_steps = max(2, round(height / riser_h))
    actual_riser = height / num_steps

    # Limit tread depth so stairs fit within run_length
    max_tread = run_length / num_steps
    tread = min(tread_d, max_tread)

    total_run = tread * num_steps
    # Start position: half the total run back from center
    start_x = cx - run_dx * (total_run / 2)
    start_y = cy - run_dy * (total_run / 2)

    half_w = stair_width / 2

    # Compute half-extents aligned to the run direction.
    # _add_box is axis-aligned, so we map run/perp to X/Y.
    # When running in X: sx = tread along run, sy = width perpendicular.
    # When running in Y: sx = width perpendicular, sy = tread along run.
    run_is_x = abs(run_dx) >= abs(run_dy)

    for i in range(num_steps):
        step_z = z_bottom + i * actual_riser
        step_cx = start_x + run_dx * (i * tread + tread / 2)
        step_cy = start_y + run_dy * (i * tread + tread / 2)

        if run_is_x:
            sx, sy = tread / 2, half_w
        else:
            sx, sy = half_w, tread / 2

        _add_box(bm,
                 step_cx, step_cy, step_z + actual_riser / 2,
                 sx, sy, actual_riser / 2)

    # Landing slab at the top (connects to upper floor)
    landing_cx = start_x + run_dx * (num_steps * tread + tread / 2)
    landing_cy = start_y + run_dy * (num_steps * tread + tread / 2)
    if run_is_x:
        lsx, lsy = tread / 2, half_w
    else:
        lsx, lsy = half_w, tread / 2
    _add_box(bm,
             landing_cx, landing_cy, z_top - 0.05,
             lsx, lsy, 0.05)


def generate_staircases(context, story_data_list, collection):
    """Generate staircases connecting each pair of adjacent stories.

    Auto-detects stairwell rooms when fp3d_stair_auto is enabled,
    otherwise uses the manual position from scene properties.

    Returns:
        int: Number of staircase connections generated.
    """
    from . import materials as mat

    scene = context.scene
    wall_height = scene.fp3d_wall_height
    num_stories = getattr(scene, "fp3d_num_stories", 1)
    auto = getattr(scene, "fp3d_stair_auto", True)
    stair_width = getattr(scene, "fp3d_stair_width", 1.0)

    total = 0
    for story in range(num_stories - 1):
        z_bottom = story * wall_height
        z_top = (story + 1) * wall_height

        data = (story_data_list[story]
                if story < len(story_data_list)
                else story_data_list[0])

        placements = []  # list of (cx, cy, run_dx, run_dy, run_length, width)

        if auto:
            # 1) Look for explicit stairwell rooms
            stairwells = _find_stairwell_rooms(data)
            for room in stairwells:
                center, run_dir, run_len, sw_width = _stairwell_center_and_run(room)
                placements.append((
                    center[0], center[1],
                    run_dir[0], run_dir[1],
                    run_len,
                    min(sw_width, stair_width),
                ))

            # 2) No stairwell room? Try the largest hallway/corridor
            if not placements:
                candidate = _find_best_stair_candidate(data)
                if candidate:
                    cx, cy, dx, dy, run_len, sw_w = candidate
                    placements.append((
                        cx, cy, dx, dy, run_len,
                        min(sw_w, stair_width),
                    ))

        # 3) Still nothing? Use manual position (default to building center)
        if not placements:
            pos_x = getattr(scene, "fp3d_stair_position_x", 0.0)
            pos_y = getattr(scene, "fp3d_stair_position_y", 0.0)

            # If manual position is still at default (0,0), use building
            # centroid so stairs appear inside the building, not at the origin.
            if abs(pos_x) < 0.01 and abs(pos_y) < 0.01:
                pos_x, pos_y = _compute_building_centroid(data)

            dir_map = {
                'X_POS': (1, 0), 'X_NEG': (-1, 0),
                'Y_POS': (0, 1), 'Y_NEG': (0, -1),
            }
            direction = getattr(scene, "fp3d_stair_direction", "X_POS")
            dx, dy = dir_map.get(direction, (1, 0))
            # Default run length based on wall height (comfortable slope)
            run_length = wall_height / 0.18 * 0.28  # steps * tread
            placements.append((pos_x, pos_y, dx, dy, run_length, stair_width))

        for cx, cy, dx, dy, run_len, sw in placements:
            bm = bmesh.new()
            _generate_stair_mesh(bm, cx, cy, z_bottom, z_top,
                                 dx, dy, run_len, sw)

            mesh = bpy.data.meshes.new(f"Staircase_{story}_{story + 1}")
            bm.to_mesh(mesh)
            bm.free()

            obj = bpy.data.objects.new(f"Staircase_{story}_{story + 1}", mesh)
            obj["fp3d_type"] = "staircase"
            obj["fp3d_story_bottom"] = story
            obj["fp3d_story_top"] = story + 1
            mat.assign_stair_material(obj)
            _link_to_collection(obj, collection)
            total += 1

    return total


# ── Furniture generation ──────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Furniture shape builders
# ---------------------------------------------------------------------------
# Each builder populates a bmesh with recognisable geometry for a furniture
# type.  All coordinates are in local space centered at the origin; the
# caller positions and rotates the final Blender object.
#
# Convention:  X = width, Y = depth, Z = up.  (0,0,0) is the bottom-centre.
# ---------------------------------------------------------------------------


def _add_box(bm, cx, cy, cz, sx, sy, sz):
    """Add an axis-aligned box to *bm*.

    Args:
        bm: bmesh to add geometry to.
        cx, cy, cz: centre of the box.
        sx, sy, sz: **half**-extents (half width/depth/height).
    """
    x0, x1 = cx - sx, cx + sx
    y0, y1 = cy - sy, cy + sy
    z0, z1 = cz - sz, cz + sz

    v = [
        bm.verts.new((x0, y0, z0)),
        bm.verts.new((x1, y0, z0)),
        bm.verts.new((x1, y1, z0)),
        bm.verts.new((x0, y1, z0)),
        bm.verts.new((x0, y0, z1)),
        bm.verts.new((x1, y0, z1)),
        bm.verts.new((x1, y1, z1)),
        bm.verts.new((x0, y1, z1)),
    ]
    # 6 faces (CCW winding when viewed from outside)
    bm.faces.new((v[0], v[3], v[2], v[1]))  # bottom
    bm.faces.new((v[4], v[5], v[6], v[7]))  # top
    bm.faces.new((v[0], v[1], v[5], v[4]))  # front  (-Y)
    bm.faces.new((v[2], v[3], v[7], v[6]))  # back   (+Y)
    bm.faces.new((v[0], v[4], v[7], v[3]))  # left   (-X)
    bm.faces.new((v[1], v[2], v[6], v[5]))  # right  (+X)




def _add_oriented_box(bm, cx, cy, cz, u_dir, v_dir, half_u, half_v, half_h):
    """Add a box oriented along arbitrary 2D axes to *bm*.

    Unlike ``_add_box`` (axis-aligned), this creates a box aligned to the
    given *u* and *v* direction vectors in the XY plane.

    Args:
        bm: bmesh to add geometry to.
        cx, cy, cz: centre of the box.
        u_dir: (ux, uy) unit vector for the 'width' direction.
        v_dir: (vx, vy) unit vector for the 'depth' direction.
        half_u: half-extent along u_dir.
        half_v: half-extent along v_dir.
        half_h: half-extent in Z.
    """
    ux, uy = u_dir[0], u_dir[1]
    vx, vy = v_dir[0], v_dir[1]

    corners_2d = [
        (cx - ux * half_u - vx * half_v, cy - uy * half_u - vy * half_v),
        (cx + ux * half_u - vx * half_v, cy + uy * half_u - vy * half_v),
        (cx + ux * half_u + vx * half_v, cy + uy * half_u + vy * half_v),
        (cx - ux * half_u + vx * half_v, cy - uy * half_u + vy * half_v),
    ]
    z_bot = cz - half_h
    z_top = cz + half_h

    v = []
    for x2, y2 in corners_2d:
        v.append(bm.verts.new((x2, y2, z_bot)))
    for x2, y2 in corners_2d:
        v.append(bm.verts.new((x2, y2, z_top)))

    # 6 faces (CCW from outside)
    bm.faces.new((v[0], v[3], v[2], v[1]))  # bottom
    bm.faces.new((v[4], v[5], v[6], v[7]))  # top
    bm.faces.new((v[0], v[1], v[5], v[4]))  # front
    bm.faces.new((v[2], v[3], v[7], v[6]))  # back
    bm.faces.new((v[0], v[4], v[7], v[3]))  # left
    bm.faces.new((v[1], v[2], v[6], v[5]))  # right


def _add_cylinder(bm, cx, cy, cz, radius, height, segments=12):
    """Add a vertical cylinder to *bm* at the given centre.

    Args:
        bm: bmesh to add geometry to.
        cx, cy, cz: centre of the cylinder.
        radius: cylinder radius.
        height: total height (extends ± height/2 from cz).
        segments: number of sides (default 12 for smooth-ish look).
    """
    z_bot = cz - height / 2
    z_top = cz + height / 2
    bot_verts = []
    top_verts = []
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        bot_verts.append(bm.verts.new((x, y, z_bot)))
        top_verts.append(bm.verts.new((x, y, z_top)))

    # Bottom and top caps
    bm.faces.new(list(reversed(bot_verts)))
    bm.faces.new(top_verts)
    # Side faces
    for i in range(segments):
        j = (i + 1) % segments
        bm.faces.new((bot_verts[i], bot_verts[j], top_verts[j], top_verts[i]))


def _add_wall_quad(bm, p0, p1, z_bot, z_top):
    """Add a single vertical wall quad between two 2D points.

    Args:
        bm: bmesh to add geometry to.
        p0, p1: (x, y) tuples for the wall's start and end.
        z_bot, z_top: bottom and top Z coordinates.
    """
    v0 = bm.verts.new((p0[0], p0[1], z_bot))
    v1 = bm.verts.new((p1[0], p1[1], z_bot))
    v2 = bm.verts.new((p1[0], p1[1], z_top))
    v3 = bm.verts.new((p0[0], p0[1], z_top))
    bm.faces.new((v0, v1, v2, v3))


# -- Beds --------------------------------------------------------------------

def _shape_bed(bm, w, d):
    """Mattress + headboard + low footboard."""
    mat_h = 0.30   # mattress height
    frame_h = 0.15  # frame under mattress
    head_h = 0.70   # headboard total height
    head_t = 0.06   # headboard thickness
    foot_h = 0.38
    foot_t = 0.05

    # Bed frame (full width/depth, low)
    _add_box(bm, 0, 0, frame_h / 2, w / 2, d / 2, frame_h / 2)
    # Mattress (slightly inset)
    inset = 0.03
    _add_box(bm, 0, 0, frame_h + mat_h / 2,
             w / 2 - inset, d / 2 - inset, mat_h / 2)
    # Headboard at +Y end
    _add_box(bm, 0, d / 2 - head_t / 2, head_h / 2,
             w / 2, head_t / 2, head_h / 2)
    # Footboard at -Y end
    _add_box(bm, 0, -d / 2 + foot_t / 2, foot_h / 2,
             w / 2, foot_t / 2, foot_h / 2)


# -- Seating -----------------------------------------------------------------

def _shape_sofa(bm, w, d):
    """Seat base + backrest + two armrests."""
    seat_h = 0.42
    back_h = 0.75
    back_t = 0.15
    arm_w = 0.12
    arm_h = 0.58

    # Seat cushion
    _add_box(bm, 0, -back_t / 2, seat_h / 2,
             w / 2, (d - back_t) / 2, seat_h / 2)
    # Backrest
    _add_box(bm, 0, d / 2 - back_t / 2, back_h / 2,
             w / 2, back_t / 2, back_h / 2)
    # Left armrest
    _add_box(bm, -w / 2 + arm_w / 2, 0, arm_h / 2,
             arm_w / 2, d / 2, arm_h / 2)
    # Right armrest
    _add_box(bm, w / 2 - arm_w / 2, 0, arm_h / 2,
             arm_w / 2, d / 2, arm_h / 2)


def _shape_armchair(bm, w, d):
    """Like a sofa but with thicker armrests."""
    seat_h = 0.42
    back_h = 0.80
    back_t = 0.14
    arm_w = 0.15
    arm_h = 0.60

    _add_box(bm, 0, -back_t / 2, seat_h / 2,
             w / 2, (d - back_t) / 2, seat_h / 2)
    _add_box(bm, 0, d / 2 - back_t / 2, back_h / 2,
             w / 2, back_t / 2, back_h / 2)
    _add_box(bm, -w / 2 + arm_w / 2, 0, arm_h / 2,
             arm_w / 2, d / 2, arm_h / 2)
    _add_box(bm, w / 2 - arm_w / 2, 0, arm_h / 2,
             arm_w / 2, d / 2, arm_h / 2)


def _shape_chair(bm, w, d):
    """Seat + 4 legs + backrest panel."""
    seat_h = 0.45
    seat_t = 0.04
    leg = 0.03  # leg half-width
    back_h = 0.85
    back_t = 0.03

    # 4 legs
    inset = 0.04
    for sx in (-1, 1):
        for sy in (-1, 1):
            lx = sx * (w / 2 - inset - leg)
            ly = sy * (d / 2 - inset - leg)
            _add_box(bm, lx, ly, (seat_h - seat_t) / 2,
                     leg, leg, (seat_h - seat_t) / 2)
    # Seat slab
    _add_box(bm, 0, 0, seat_h - seat_t / 2,
             w / 2, d / 2, seat_t / 2)
    # Backrest
    _add_box(bm, 0, d / 2 - back_t / 2, (seat_h + back_h) / 2,
             w / 2 - 0.02, back_t / 2, (back_h - seat_h) / 2)


# -- Tables ------------------------------------------------------------------

def _shape_table(bm, w, d):
    """Tabletop + 4 legs."""
    top_h = 0.75
    top_t = 0.04
    leg = 0.03
    inset = 0.06

    # 4 legs
    for sx in (-1, 1):
        for sy in (-1, 1):
            lx = sx * (w / 2 - inset - leg)
            ly = sy * (d / 2 - inset - leg)
            _add_box(bm, lx, ly, (top_h - top_t) / 2,
                     leg, leg, (top_h - top_t) / 2)
    # Tabletop
    _add_box(bm, 0, 0, top_h - top_t / 2,
             w / 2, d / 2, top_t / 2)


def _shape_coffee_table(bm, w, d):
    """Lower table with thicker top."""
    top_h = 0.42
    top_t = 0.05
    leg = 0.03
    inset = 0.05

    for sx in (-1, 1):
        for sy in (-1, 1):
            lx = sx * (w / 2 - inset - leg)
            ly = sy * (d / 2 - inset - leg)
            _add_box(bm, lx, ly, (top_h - top_t) / 2,
                     leg, leg, (top_h - top_t) / 2)
    _add_box(bm, 0, 0, top_h - top_t / 2,
             w / 2, d / 2, top_t / 2)


def _shape_desk(bm, w, d):
    """Tabletop + 2 solid side panels."""
    top_h = 0.75
    top_t = 0.04
    panel_t = 0.04  # thickness of side panels
    inset = 0.02

    # Left side panel
    _add_box(bm, -w / 2 + panel_t / 2 + inset, 0, (top_h - top_t) / 2,
             panel_t / 2, d / 2 - inset, (top_h - top_t) / 2)
    # Right side panel
    _add_box(bm, w / 2 - panel_t / 2 - inset, 0, (top_h - top_t) / 2,
             panel_t / 2, d / 2 - inset, (top_h - top_t) / 2)
    # Tabletop
    _add_box(bm, 0, 0, top_h - top_t / 2,
             w / 2, d / 2, top_t / 2)
    # Back panel (thin modesty panel)
    _add_box(bm, 0, d / 2 - 0.01, (top_h - top_t) * 0.35,
             w / 2 - panel_t - inset, 0.01, (top_h - top_t) * 0.35)


# -- Storage -----------------------------------------------------------------

def _shape_nightstand(bm, w, d):
    """Box body + slightly overhanging top."""
    body_h = 0.50
    top_t = 0.03
    overhang = 0.02

    # Body
    _add_box(bm, 0, 0, body_h / 2, w / 2, d / 2, body_h / 2)
    # Top surface (overhangs slightly)
    _add_box(bm, 0, 0, body_h + top_t / 2,
             w / 2 + overhang, d / 2 + overhang, top_t / 2)


def _shape_wardrobe(bm, w, d):
    """Tall box + crown moulding overhang."""
    body_h = 2.00
    top_t = 0.04
    overhang = 0.02

    _add_box(bm, 0, 0, body_h / 2, w / 2, d / 2, body_h / 2)
    _add_box(bm, 0, 0, body_h + top_t / 2,
             w / 2 + overhang, d / 2 + overhang, top_t / 2)


def _shape_bookshelf(bm, w, d):
    """Two side panels + top + base + 3 shelves."""
    h = 1.80
    panel_t = 0.03
    shelf_t = 0.025

    # Left side panel
    _add_box(bm, -w / 2 + panel_t / 2, 0, h / 2,
             panel_t / 2, d / 2, h / 2)
    # Right side panel
    _add_box(bm, w / 2 - panel_t / 2, 0, h / 2,
             panel_t / 2, d / 2, h / 2)
    # Top
    _add_box(bm, 0, 0, h - shelf_t / 2,
             w / 2, d / 2, shelf_t / 2)
    # Base
    _add_box(bm, 0, 0, shelf_t / 2,
             w / 2, d / 2, shelf_t / 2)
    # Back panel (thin)
    _add_box(bm, 0, d / 2 - 0.005, h / 2,
             w / 2 - panel_t, 0.005, h / 2)
    # 3 interior shelves
    inner_w = w / 2 - panel_t
    for i in range(1, 4):
        sz = h * i / 4.0
        _add_box(bm, 0, 0, sz, inner_w, d / 2 - 0.005, shelf_t / 2)


# -- Bathroom ----------------------------------------------------------------

def _shape_toilet(bm, w, d):
    """Bowl base + tank at rear."""
    bowl_h = 0.40
    bowl_d = d * 0.65  # bowl takes front 65%
    tank_h = 0.65
    tank_d = d * 0.35
    tank_w = w * 0.75

    # Bowl
    _add_box(bm, 0, -d / 2 + bowl_d / 2, bowl_h / 2,
             w / 2, bowl_d / 2, bowl_h / 2)
    # Tank
    _add_box(bm, 0, d / 2 - tank_d / 2, tank_h / 2,
             tank_w / 2, tank_d / 2, tank_h / 2)


def _shape_bathtub(bm, w, d):
    """Outer shell with thick walls (solid block approach — outer + rim)."""
    outer_h = 0.55
    wall_t = 0.06
    rim_h = 0.04

    # Outer block
    _add_box(bm, 0, 0, outer_h / 2, w / 2, d / 2, outer_h / 2)
    # Inner cavity (lower, to create rim) — slightly shorter
    # We approximate the hollow by just adding a visible rim ledge on top
    inner_d = d / 2 - wall_t
    # Rim: 4 border strips on top
    _add_box(bm, 0, d / 2 - wall_t / 2, outer_h + rim_h / 2,
             w / 2, wall_t / 2, rim_h / 2)       # back
    _add_box(bm, 0, -d / 2 + wall_t / 2, outer_h + rim_h / 2,
             w / 2, wall_t / 2, rim_h / 2)       # front
    _add_box(bm, -w / 2 + wall_t / 2, 0, outer_h + rim_h / 2,
             wall_t / 2, inner_d, rim_h / 2)     # left
    _add_box(bm, w / 2 - wall_t / 2, 0, outer_h + rim_h / 2,
             wall_t / 2, inner_d, rim_h / 2)     # right


def _shape_sink(bm, w, d):
    """Pedestal + basin on top."""
    pedestal_h = 0.65
    pedestal_w = w * 0.35
    pedestal_d = d * 0.35
    basin_h = 0.20
    basin_w = w * 0.85
    basin_d = d * 0.75

    # Pedestal
    _add_box(bm, 0, 0, pedestal_h / 2,
             pedestal_w / 2, pedestal_d / 2, pedestal_h / 2)
    # Basin
    _add_box(bm, 0, 0, pedestal_h + basin_h / 2,
             basin_w / 2, basin_d / 2, basin_h / 2)


# -- Kitchen / Appliances ---------------------------------------------------

def _shape_fridge(bm, w, d):
    """Tall box + top cap."""
    body_h = 1.75
    cap_t = 0.05

    _add_box(bm, 0, 0, body_h / 2, w / 2, d / 2, body_h / 2)
    _add_box(bm, 0, 0, body_h + cap_t / 2,
             w / 2, d / 2, cap_t / 2)


def _shape_oven(bm, w, d):
    """Box body + handle bar on front."""
    body_h = 0.85
    handle_h = 0.06
    handle_t = 0.025

    _add_box(bm, 0, 0, body_h / 2, w / 2, d / 2, body_h / 2)
    # Handle bar centered on front face
    _add_box(bm, 0, -d / 2 - handle_t / 2, body_h * 0.72,
             w / 3, handle_t / 2, handle_h / 2)


def _shape_washing_machine(bm, w, d):
    """Box + slight top lip."""
    body_h = 0.85
    lip_t = 0.03

    _add_box(bm, 0, 0, body_h / 2, w / 2, d / 2, body_h / 2)
    _add_box(bm, 0, 0, body_h + lip_t / 2,
             w / 2 + 0.01, d / 2 + 0.01, lip_t / 2)


# -- Electronics -------------------------------------------------------------

def _shape_tv(bm, w, d):
    """Thin wide screen panel + small base stand."""
    screen_h = w * 0.56  # ~16:9
    screen_t = 0.035
    stand_w = w * 0.30
    stand_d = d * 0.80
    stand_h = 0.04
    neck_w = 0.06
    neck_d = 0.06
    neck_h = 0.08

    # Stand base
    _add_box(bm, 0, 0, stand_h / 2,
             stand_w / 2, stand_d / 2, stand_h / 2)
    # Neck
    _add_box(bm, 0, 0, stand_h + neck_h / 2,
             neck_w / 2, neck_d / 2, neck_h / 2)
    # Screen panel
    _add_box(bm, 0, 0, stand_h + neck_h + screen_h / 2,
             w / 2, screen_t / 2, screen_h / 2)


# -- Fallback ----------------------------------------------------------------

def _shape_default(bm, w, d):
    """Plain box — fallback for unrecognised furniture."""
    h = 0.75
    _add_box(bm, 0, 0, h / 2, w / 2, d / 2, h / 2)


# Dispatch map from furniture key → builder
_FURNITURE_SHAPE_BUILDERS = {
    "bed": _shape_bed, "double_bed": _shape_bed, "single_bed": _shape_bed,
    "sofa": _shape_sofa, "couch": _shape_sofa,
    "armchair": _shape_armchair,
    "table": _shape_table, "dining_table": _shape_table,
    "coffee_table": _shape_coffee_table,
    "desk": _shape_desk,
    "chair": _shape_chair, "dining_chair": _shape_chair, "stool": _shape_chair,
    "nightstand": _shape_nightstand,
    "wardrobe": _shape_wardrobe, "dresser": _shape_nightstand,
    "bookshelf": _shape_bookshelf, "cabinet": _shape_bookshelf,
    "toilet": _shape_toilet,
    "bathtub": _shape_bathtub, "shower": _shape_bathtub,
    "sink": _shape_sink,
    "refrigerator": _shape_fridge, "fridge": _shape_fridge,
    "oven": _shape_oven, "stove": _shape_oven,
    "tv": _shape_tv, "television": _shape_tv,
    "washing_machine": _shape_washing_machine,
}


def _shape_from_geometry_list(bm, geometry_list, w, d):
    """Build furniture from a declarative geometry list returned by the AI.

    Each entry in *geometry_list* is a dict describing a primitive:

    Box:
        {"type": "box", "cx": 0, "cy": 0, "cz": 0.4,
         "sx": 0.3, "sy": 0.2, "sz": 0.4}
        cx/cy/cz = centre, sx/sy/sz = half-extents

    Cylinder:
        {"type": "cylinder", "cx": 0, "cy": 0, "cz": 0.3,
         "r": 0.15, "h": 0.6, "segments": 12}

    All coordinates are in *local* furniture space (origin at item centre,
    X-right, Y-back, Z-up) with the overall item fitting inside the nominal
    width ``w`` and depth ``d`` envelope.
    """
    if not geometry_list:
        # Fallback: plain box
        _add_box(bm, 0, 0, 0.375, w / 2, d / 2, 0.375)
        return

    for g in geometry_list:
        ptype = g.get("type", "box").lower()
        if ptype == "cylinder":
            _add_cylinder(
                bm,
                float(g.get("cx", 0)),
                float(g.get("cy", 0)),
                float(g.get("cz", 0)),
                float(g.get("r", 0.1)),
                float(g.get("h", 0.5)),
                int(g.get("segments", 12)),
            )
        else:  # box (default)
            _add_box(
                bm,
                float(g.get("cx", 0)),
                float(g.get("cy", 0)),
                float(g.get("cz", 0)),
                float(g.get("sx", w / 2)),
                float(g.get("sy", d / 2)),
                float(g.get("sz", 0.375)),
            )


def generate_furniture(furniture_data, collection, z_offset=0.0, name_prefix=""):
    """Create shaped meshes representing furniture items.

    Each furniture type is built from multiple box primitives to produce
    a recognisable silhouette (beds with headboards, tables with legs, etc.).

    If an item contains a ``"geometry"`` key with a list of primitive dicts,
    that declarative geometry is used instead of the template builder (hybrid
    mode — lets the AI define custom shapes for style-specific pieces).

    Args:
        furniture_data: dict keyed by room label, each value is a list of items:
            [{"name": "sofa", "position": [x,y], "dimensions": [w,d],
              "rotation": deg, "geometry": [...]}, ...]
        collection: Blender collection to add objects to.
        z_offset: vertical offset for multi-story placement.
        name_prefix: prefix for object names (e.g. "S1_" for story 1).

    Returns:
        int: Number of furniture items generated.
    """
    from . import materials

    count = 0
    for room_label, items in furniture_data.items():
        # Handle both flat format (items is a list) and nested format
        # (items is {"furniture": [...]}) as returned by some AI backends.
        if isinstance(items, dict):
            items = items.get("furniture", items.get("items", []))
        if not isinstance(items, list):
            continue
        for item in items:
            name = item.get("name", "furniture")
            pos = item.get("position", [0, 0])
            dims = item.get("dimensions", [0.5, 0.5])
            rotation_deg = item.get("rotation", 0)

            w = dims[0] if len(dims) > 0 else 0.5
            d = dims[1] if len(dims) > 1 else 0.5

            # Build the mesh — prefer inline geometry, fall back to template
            mesh = bpy.data.meshes.new(f"{name_prefix}Furn_{name}_{count}")
            bm = bmesh.new()

            geom_list = item.get("geometry")
            if geom_list and isinstance(geom_list, list):
                # AI-defined custom geometry
                _shape_from_geometry_list(bm, geom_list, w, d)
            else:
                # Template shape builder
                key = name.lower().replace(" ", "_")
                builder = _FURNITURE_SHAPE_BUILDERS.get(key, _shape_default)
                builder(bm, w, d)

            bm.to_mesh(mesh)
            bm.free()
            mesh.update()

            obj = bpy.data.objects.new(f"{name_prefix}Furn_{name}_{count}", mesh)
            obj.location = Vector((pos[0], pos[1], z_offset))
            obj.rotation_euler.z = math.radians(rotation_deg)
            obj["fp3d_type"] = "furniture"
            obj["fp3d_furniture_name"] = name
            obj["fp3d_room_label"] = room_label
            _link_to_collection(obj, collection)

            materials.assign_furniture_material(obj, furniture_name=name)
            count += 1

    return count


def remove_furniture(collection):
    """Remove all furniture objects from the collection (including story sub-collections)."""
    if not collection:
        return 0
    removed = 0
    for obj in list(collection.objects):
        if obj.get("fp3d_type") == "furniture":
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    # Also remove from story sub-collections
    for child in list(collection.children):
        if child.name.startswith("Story_"):
            for obj in list(child.objects):
                if obj.get("fp3d_type") == "furniture":
                    bpy.data.objects.remove(obj, do_unlink=True)
                    removed += 1
    return removed


# ═══════════════════════════════════════════════════════════════════════════════
# EXTERIOR GEOMETRY — Roof, foundation, window frames, door surrounds
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULT_EXTERIOR_CONFIG = {
    "roof": {
        "type": "gable",
        "pitch_degrees": 30,
        "overhang": 0.5,
        "ridge_direction": "auto",
        "material": "clay_tile",
    },
    "facade": {
        "foundation": {"height": 0.3, "material": "concrete"},
        "cladding": {"material": "white_stucco"},
        "trim": {"material": "dark_wood", "width": 0.08},
        "window_frames": {"material": "dark_wood", "depth": 0.06},
        "door_surround": {"material": "stone", "width": 0.12},
    },
    "details": [],
}


def _deep_merge(base, override):
    """Recursively merge *override* into a copy of *base*."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _validate_exterior_config(config):
    """Merge AI response with defaults so every key exists."""
    return _deep_merge(_DEFAULT_EXTERIOR_CONFIG, config or {})


# -- Roof generators --------------------------------------------------------

def _roof_rect_helpers(rect_corners):
    """Compute oriented bounding rect vectors and dimensions from 4 corners.

    Returns (long_len, short_len, lu, pu, eave_a, eave_b, centre)
    where lu = unit along long axis, pu = unit along short axis.
    """
    def _ev(a, b):
        return [b[0] - a[0], b[1] - a[1]]

    def _ln(v):
        return math.sqrt(v[0] ** 2 + v[1] ** 2)

    e01 = _ev(rect_corners[0], rect_corners[1])
    e12 = _ev(rect_corners[1], rect_corners[2])
    l01 = _ln(e01)
    l12 = _ln(e12)

    if l01 >= l12:
        long_len, short_len = l01, l12
        eave_a = [rect_corners[0], rect_corners[1]]
        eave_b = [rect_corners[3], rect_corners[2]]
    else:
        long_len, short_len = l12, l01
        eave_a = [rect_corners[1], rect_corners[2]]
        eave_b = [rect_corners[0], rect_corners[3]]

    lu_raw = _ev(eave_a[0], eave_a[1])
    lu = [lu_raw[0] / long_len, lu_raw[1] / long_len]

    mid_a = [(eave_a[0][0] + eave_a[1][0]) / 2, (eave_a[0][1] + eave_a[1][1]) / 2]
    mid_b = [(eave_b[0][0] + eave_b[1][0]) / 2, (eave_b[0][1] + eave_b[1][1]) / 2]
    perp_raw = [mid_b[0] - mid_a[0], mid_b[1] - mid_a[1]]
    plen = _ln(perp_raw)
    pu = [perp_raw[0] / plen, perp_raw[1] / plen] if plen > 0 else [0, 1]

    centre = [(mid_a[0] + mid_b[0]) / 2, (mid_a[1] + mid_b[1]) / 2]

    return long_len, short_len, lu, pu, eave_a, eave_b, centre


def _expand_corners(rect_corners, oh):
    """Expand polygon corners outward from centroid by *oh* metres."""
    n = len(rect_corners)
    cx = sum(c[0] for c in rect_corners) / n
    cy = sum(c[1] for c in rect_corners) / n
    exp = []
    for c in rect_corners:
        dx = c[0] - cx
        dy = c[1] - cy
        d = math.sqrt(dx * dx + dy * dy) or 1.0
        exp.append([c[0] + dx / d * oh, c[1] + dy / d * oh])
    return exp


def _offset_polygon_edges(corners, offset):
    """Offset a polygon (convex or concave) by moving each edge outward.

    Unlike ``_expand_corners`` (which pushes each vertex radially from the
    centroid — fine for convex shapes but wrong for concave ones), this
    computes per-edge outward normals and intersects adjacent offset edges
    to find new corner positions.  Works correctly for any simple polygon.
    """
    n = len(corners)
    if n < 3 or abs(offset) < 1e-6:
        return list(corners)

    # Determine polygon winding (CW vs CCW) via signed area
    area2 = 0.0
    for i in range(n):
        j = (i + 1) % n
        area2 += corners[i][0] * corners[j][1]
        area2 -= corners[j][0] * corners[i][1]
    sign = 1.0 if area2 >= 0 else -1.0  # CCW → positive

    # Build offset edge lines (moved outward by *offset*)
    lines = []  # each: (point_on_line, direction_unit)
    for i in range(n):
        j = (i + 1) % n
        dx = corners[j][0] - corners[i][0]
        dy = corners[j][1] - corners[i][1]
        elen = math.sqrt(dx * dx + dy * dy)
        if elen < 1e-9:
            lines.append(None)
            continue
        # Outward normal (perpendicular, direction depends on winding)
        nx = -dy / elen * sign
        ny = dx / elen * sign
        # Offset point
        px = corners[i][0] + nx * offset
        py = corners[i][1] + ny * offset
        lines.append(((px, py), (dx / elen, dy / elen)))

    result = []
    for i in range(n):
        prev = (i - 1) % n
        l0 = lines[prev]
        l1 = lines[i]
        if l0 is None or l1 is None:
            result.append(list(corners[i]))
            continue
        # Intersect two lines:  l0.p + t*l0.d  ==  l1.p + s*l1.d
        p0, d0 = l0
        p1, d1 = l1
        denom = d0[0] * d1[1] - d0[1] * d1[0]
        if abs(denom) < 1e-9:
            # Parallel edges — just use midpoint of the two offset positions
            result.append([(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2])
        else:
            dpx = p1[0] - p0[0]
            dpy = p1[1] - p0[1]
            t = (dpx * d1[1] - dpy * d1[0]) / denom
            result.append([p0[0] + t * d0[0], p0[1] + t * d0[1]])
    return result


def _generate_lower_roof_slab(bm, outline, config, wall_height):
    """Simple flat slab for the lower roof of a multi-story setback building.

    Unlike ``_generate_modern_flat_roof``, this:
    - Uses edge-normal offset instead of radial expansion (handles concave)
    - Creates NO parapet walls (the junction with the upper floor shouldn't
      have parapets, and the outer edges are short single-story walls)
    - Creates a solid slab by fan-triangulating top and bottom faces (avoids
      BMesh ngon tessellation issues with concave polygons)
    """
    oh = min(config.get("overhang", 0.15), 0.3)  # smaller overhang for lower
    thickness = 0.25
    parapet_h = 0.4  # short parapet for clean edge

    expanded = _offset_polygon_edges(outline, oh)
    z_bot = wall_height
    z_top = wall_height + thickness

    n = len(expanded)
    if n < 3:
        return

    bot_v = [bm.verts.new((p[0], p[1], z_bot)) for p in expanded]
    top_v = [bm.verts.new((p[0], p[1], z_top)) for p in expanded]

    new_faces = []

    # Fan-triangulate top and bottom faces from vertex 0 to avoid
    # BMesh ngon issues with concave polygons.
    for i in range(1, n - 1):
        new_faces.append(bm.faces.new((top_v[0], top_v[i], top_v[i + 1])))
        new_faces.append(
            bm.faces.new((bot_v[0], bot_v[i + 1], bot_v[i])))

    # Side quads
    for i in range(n):
        j = (i + 1) % n
        new_faces.append(
            bm.faces.new((bot_v[i], bot_v[j], top_v[j], top_v[i])))

    # Short parapet walls on outer edges for a clean roofline
    cx = sum(c[0] for c in expanded) / n
    cy = sum(c[1] for c in expanded) / n
    for i in range(n):
        j = (i + 1) % n
        p0, p1 = expanded[i], expanded[j]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        elen = math.sqrt(dx * dx + dy * dy)
        if elen < 0.05:
            continue
        eu = (dx / elen, dy / elen)
        nx, ny = -dy / elen, dx / elen
        mid = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2]
        if (mid[0] + nx - cx) ** 2 + (mid[1] + ny - cy) ** 2 < \
           (mid[0] - nx - cx) ** 2 + (mid[1] - ny - cy) ** 2:
            nx, ny = -nx, -ny
        pt = 0.10  # parapet thickness
        pcx = (p0[0] + p1[0]) / 2 + nx * pt / 2
        pcy = (p0[1] + p1[1]) / 2 + ny * pt / 2
        _add_oriented_box(bm, pcx, pcy,
                          wall_height + parapet_h / 2,
                          eu, (nx, ny),
                          elen / 2, pt / 2, parapet_h / 2)

    bmesh.ops.recalc_face_normals(bm, faces=new_faces)


def _generate_flat_roof(bm, rect_corners, config, wall_height):
    """Flat roof — simple slab with overhang."""
    oh = min(config.get("overhang", 0.5), 0.5)
    thickness = 0.20
    expanded = _expand_corners(rect_corners, oh)
    z_bot = wall_height
    z_top = wall_height + thickness
    bot_v = [bm.verts.new((p[0], p[1], z_bot)) for p in expanded]
    top_v = [bm.verts.new((p[0], p[1], z_top)) for p in expanded]
    bm.faces.new(top_v)
    bm.faces.new(list(reversed(bot_v)))
    n = len(expanded)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((bot_v[i], bot_v[j], top_v[j], top_v[i]))


def _generate_modern_flat_roof(bm, rect_corners, config, wall_height):
    """Modern flat roof with parapet wall and a slightly recessed top slab.

    Looks like contemporary/modern architecture — clean lines, hidden drainage.
    Uses edge-normal polygon offset (not radial expansion) so concave
    footprints (L-shapes, setbacks) are handled correctly.
    """
    oh = min(config.get("overhang", 0.3), 0.5)  # clamp overhang to 0.5m max
    parapet_h = config.get("parapet_height", 0.6)
    slab_thickness = 0.25
    parapet_thickness = 0.15

    # Use proper edge-normal offset for concave polygon support
    expanded = _offset_polygon_edges(rect_corners, oh)
    n = len(expanded)
    if n < 3:
        return

    # Parapet walls — vertical strips around perimeter
    cx = sum(c[0] for c in expanded) / n
    cy = sum(c[1] for c in expanded) / n

    for i in range(n):
        j = (i + 1) % n
        p0, p1 = expanded[i], expanded[j]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        elen = math.sqrt(dx * dx + dy * dy)
        if elen < 0.01:
            continue
        eu = (dx / elen, dy / elen)
        # Outward normal
        nx, ny = -dy / elen, dx / elen
        mid = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2]
        if (mid[0] + nx - cx) ** 2 + (mid[1] + ny - cy) ** 2 < \
           (mid[0] - nx - cx) ** 2 + (mid[1] - ny - cy) ** 2:
            nx, ny = -nx, -ny
        nv = (nx, ny)

        pcx = (p0[0] + p1[0]) / 2 + nx * parapet_thickness / 2
        pcy = (p0[1] + p1[1]) / 2 + ny * parapet_thickness / 2
        _add_oriented_box(bm, pcx, pcy,
                          wall_height + parapet_h / 2,
                          eu, nv,
                          elen / 2, parapet_thickness / 2, parapet_h / 2)

    # Recessed flat slab (slightly below parapet top)
    # Use edge-normal offset and fan-triangulate for concave polygon support
    inset = _offset_polygon_edges(rect_corners, oh - parapet_thickness)
    bot_v = [bm.verts.new((p[0], p[1], wall_height)) for p in inset]
    top_v = [bm.verts.new((p[0], p[1], wall_height + slab_thickness)) for p in inset]
    ni = len(inset)
    for i in range(1, ni - 1):
        bm.faces.new((top_v[0], top_v[i], top_v[i + 1]))
        bm.faces.new((bot_v[0], bot_v[i + 1], bot_v[i]))


def _make_roof_slab(bm, top_profile, thickness=0.15):
    """Build a solid roof slab from a top-surface profile.

    Creates a closed prismatic solid (top face, bottom face, side quads).
    Because the mesh is a closed volume, recalc_face_normals reliably
    points all normals outward — no manual winding order needed.

    Works for triangles (3 points), quads (4 points), or any polygon.
    """
    n = len(top_profile)
    top_verts = [bm.verts.new(p) for p in top_profile]
    bot_verts = [bm.verts.new((p[0], p[1], p[2] - thickness))
                 for p in top_profile]

    new_faces = []
    new_faces.append(bm.faces.new(top_verts))
    new_faces.append(bm.faces.new(list(reversed(bot_verts))))
    for i in range(n):
        j = (i + 1) % n
        new_faces.append(
            bm.faces.new((top_verts[i], bot_verts[i], bot_verts[j], top_verts[j]))
        )
    # Closed solid → recalc is reliable
    bmesh.ops.recalc_face_normals(bm, faces=new_faces)


def _roof_edge_lengths(exp):
    """Return lengths of the 4 edges of an expanded footprint quad."""
    lengths = []
    for i in range(4):
        j = (i + 1) % 4
        dx = exp[j][0] - exp[i][0]
        dy = exp[j][1] - exp[i][1]
        lengths.append(math.sqrt(dx * dx + dy * dy))
    return lengths


def _roof_sides(exp):
    """Split expanded quad into long-eave sides vs short-gable sides.

    Returns (eave_a, eave_b, mid0, mid1, half_span) where:
      eave_a = (corner, corner) — one long side
      eave_b = (corner, corner) — other long side
      mid0, mid1 = midpoints of short sides (ridge / valley line)
      half_span = half the short-side distance (perpendicular to eaves)
    """
    el = _roof_edge_lengths(exp)
    if el[0] + el[2] >= el[1] + el[3]:
        # Edges 0→1 and 3→2 are long (eaves)
        eave_a = (exp[0], exp[1])
        eave_b = (exp[3], exp[2])
        mid0 = ((exp[0][0] + exp[3][0]) / 2, (exp[0][1] + exp[3][1]) / 2)
        mid1 = ((exp[1][0] + exp[2][0]) / 2, (exp[1][1] + exp[2][1]) / 2)
        half_span = (el[1] + el[3]) / 4.0
    else:
        # Edges 1→2 and 0→3 are long (eaves)
        eave_a = (exp[1], exp[2])
        eave_b = (exp[0], exp[3])
        mid0 = ((exp[0][0] + exp[1][0]) / 2, (exp[0][1] + exp[1][1]) / 2)
        mid1 = ((exp[2][0] + exp[3][0]) / 2, (exp[2][1] + exp[3][1]) / 2)
        half_span = (el[0] + el[2]) / 4.0
    return eave_a, eave_b, mid0, mid1, half_span


def _generate_gable_roof(bm, rect_corners, config, wall_height):
    """Gable roof — two solid slabs meeting at a central ridge."""
    pitch = math.radians(config.get("pitch_degrees", 30))
    oh = min(config.get("overhang", 0.5), 0.5)
    exp = _expand_corners(rect_corners, oh)
    eave_a, eave_b, r0, r1, half_span = _roof_sides(exp)

    ridge_z = wall_height + math.tan(pitch) * half_span

    _make_roof_slab(bm, [
        (*eave_a[0], wall_height), (*eave_a[1], wall_height),
        (*r1, ridge_z), (*r0, ridge_z),
    ])
    _make_roof_slab(bm, [
        (*eave_b[0], wall_height), (*eave_b[1], wall_height),
        (*r1, ridge_z), (*r0, ridge_z),
    ])


def _generate_hip_roof(bm, rect_corners, config, wall_height):
    """Hip roof — four solid slabs (two quads + two triangles)."""
    pitch = math.radians(config.get("pitch_degrees", 30))
    oh = min(config.get("overhang", 0.5), 0.5)
    exp = _expand_corners(rect_corners, oh)
    eave_a, eave_b, end0, end1, half_span = _roof_sides(exp)

    rise = math.tan(pitch) * half_span
    ridge_z = wall_height + rise

    # Ridge is inset from each end by half_span to keep hip pitch == main pitch
    # Direction from end0 to end1
    dx = end1[0] - end0[0]
    dy = end1[1] - end0[1]
    length = math.sqrt(dx * dx + dy * dy) or 1.0
    lu = (dx / length, dy / length)
    inset = min(half_span, length / 2.0 - 0.05)
    cx = (end0[0] + end1[0]) / 2
    cy = (end0[1] + end1[1]) / 2
    ridge_half = max(length / 2.0 - inset, 0.1)
    r0 = (cx - lu[0] * ridge_half, cy - lu[1] * ridge_half)
    r1 = (cx + lu[0] * ridge_half, cy + lu[1] * ridge_half)

    # Two long quad slopes
    _make_roof_slab(bm, [
        (*eave_a[0], wall_height), (*eave_a[1], wall_height),
        (*r1, ridge_z), (*r0, ridge_z),
    ])
    _make_roof_slab(bm, [
        (*eave_b[0], wall_height), (*eave_b[1], wall_height),
        (*r1, ridge_z), (*r0, ridge_z),
    ])
    # Two triangular hip ends
    _make_roof_slab(bm, [
        (*eave_a[0], wall_height), (*eave_b[0], wall_height),
        (*r0, ridge_z),
    ])
    _make_roof_slab(bm, [
        (*eave_a[1], wall_height), (*eave_b[1], wall_height),
        (*r1, ridge_z),
    ])


def _generate_hip_roof_poly(bm, polygon, config, wall_height):
    """Hip roof over any convex polygon with a proper ridge line.

    Uses the minimum bounding rectangle of the polygon to determine the
    long-axis direction.  A ridge line runs along that axis (inset from
    each end by half_span so the hip-end pitch matches the main pitch).
    Each hull edge then gets a roof face that slopes up to the nearest
    segment of the ridge — trapezoidal for long-side edges, triangular
    for hip-end edges.

    Works with 4-point rects AND arbitrary convex hulls, preventing
    the roof from overshooting the building footprint while still
    producing a realistic ridged hip roof (not a flat pyramid).
    """
    pitch = math.radians(config.get("pitch_degrees", 25))
    oh = min(config.get("overhang", 0.5), 0.5)
    exp = _expand_corners(polygon, oh)
    n = len(exp)

    # --- Find the long axis via minimum bounding rect of the *original*
    #     polygon (before overhang expansion) so orientation is stable. ---
    rect, _area, _angle = _min_bounding_rect(polygon)
    long_len, short_len, lu, pu, _ea, _eb, centre = _roof_rect_helpers(rect)

    half_span = short_len / 2.0
    rise = math.tan(pitch) * half_span
    ridge_z = wall_height + rise

    # Ridge runs along the long axis, inset from each end by half_span
    # (so the hip-end pitch equals the main-slope pitch).
    inset = min(half_span, long_len / 2.0 - 0.05)
    ridge_half = max(long_len / 2.0 - inset, 0.1)
    r0 = (centre[0] - lu[0] * ridge_half, centre[1] - lu[1] * ridge_half)
    r1 = (centre[0] + lu[0] * ridge_half, centre[1] + lu[1] * ridge_half)

    # --- Helper: closest point on ridge segment r0→r1 for a given (x,y) ---
    def _closest_on_ridge(px, py):
        rdx = r1[0] - r0[0]
        rdy = r1[1] - r0[1]
        rlen2 = rdx * rdx + rdy * rdy
        if rlen2 < 1e-9:
            return r0  # degenerate → single point
        t = ((px - r0[0]) * rdx + (py - r0[1]) * rdy) / rlen2
        t = max(0.0, min(1.0, t))
        return (r0[0] + t * rdx, r0[1] + t * rdy)

    # --- Build one roof face per hull edge ---
    for i in range(n):
        j = (i + 1) % n
        pi = exp[i]
        pj = exp[j]

        ci = _closest_on_ridge(pi[0], pi[1])
        cj = _closest_on_ridge(pj[0], pj[1])

        # If both vertices project to (nearly) the same ridge point → triangle
        dist2 = (ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2
        if dist2 < 0.01:
            mid = ((ci[0] + cj[0]) / 2, (ci[1] + cj[1]) / 2)
            _make_roof_slab(bm, [
                (*pi, wall_height),
                (*pj, wall_height),
                (*mid, ridge_z),
            ])
        else:
            # Quad / trapezoid — order matters for correct normals
            _make_roof_slab(bm, [
                (*pi, wall_height),
                (*pj, wall_height),
                (*cj, ridge_z),
                (*ci, ridge_z),
            ])


def _generate_shed_roof(bm, rect_corners, config, wall_height):
    """Shed / mono-pitch roof — one solid tilted slab."""
    pitch = math.radians(config.get("pitch_degrees", 12))
    oh = min(config.get("overhang", 0.4), 0.5)
    exp = _expand_corners(rect_corners, oh)
    eave_a, eave_b, _, _, half_span = _roof_sides(exp)

    rise = math.tan(pitch) * half_span * 2
    low_z = wall_height
    high_z = wall_height + rise

    _make_roof_slab(bm, [
        (*eave_a[0], low_z), (*eave_a[1], low_z),
        (*eave_b[1], high_z), (*eave_b[0], high_z),
    ], thickness=0.18)


def _generate_butterfly_roof(bm, rect_corners, config, wall_height):
    """Butterfly roof — eaves rise above wall_height, valley sits at wall_height.

    Built as one closed solid: top is two tilted planes meeting at a
    valley, bottom is flat, sides are quads. Closed volume ⇒ reliable
    normals via recalc_face_normals.

    The valley stays flush with the wall tops; only the eaves rise,
    giving the characteristic butterfly silhouette without an ugly V dip.
    """
    pitch = math.radians(max(config.get("pitch_degrees", 18), 12))
    oh = min(config.get("overhang", 0.6), 0.5)
    slab_t = 0.14
    exp = _expand_corners(rect_corners, oh)
    eave_a, eave_b, v0, v1, half_span = _roof_sides(exp)

    rise = math.tan(pitch) * half_span
    # Eaves rise above wall_height; valley stays flush with wall tops
    eave_z = wall_height + rise
    valley_z = wall_height
    bot_z = valley_z - slab_t  # flat bottom below the valley

    # 6 top vertices: 4 eave corners (high) + 2 valley midpoints (low)
    ta0 = bm.verts.new((*eave_a[0], eave_z))
    ta1 = bm.verts.new((*eave_a[1], eave_z))
    tb0 = bm.verts.new((*eave_b[0], eave_z))
    tb1 = bm.verts.new((*eave_b[1], eave_z))
    tv0 = bm.verts.new((*v0, valley_z))
    tv1 = bm.verts.new((*v1, valley_z))

    # 4 bottom vertices: flat rect at bot_z
    ba0 = bm.verts.new((*eave_a[0], bot_z))
    ba1 = bm.verts.new((*eave_a[1], bot_z))
    bb0 = bm.verts.new((*eave_b[0], bot_z))
    bb1 = bm.verts.new((*eave_b[1], bot_z))

    new_faces = []
    # Top faces — two tilted quads forming the V
    new_faces.append(bm.faces.new((ta0, ta1, tv1, tv0)))   # slope A
    new_faces.append(bm.faces.new((tb1, tb0, tv0, tv1)))   # slope B

    # Bottom face — single flat quad
    new_faces.append(bm.faces.new((ba0, bb0, bb1, ba1)))

    # Side faces — connect top edges to bottom edges
    new_faces.append(bm.faces.new((ta0, ba0, ba1, ta1)))   # eave A side
    new_faces.append(bm.faces.new((tb0, tb1, bb1, bb0)))   # eave B side

    # End caps — pentagons connecting top V-profile to bottom rect
    new_faces.append(bm.faces.new((ta0, tv0, tb0, bb0, ba0)))  # end 0
    new_faces.append(bm.faces.new((ta1, ba1, bb1, tb1, tv1)))  # end 1

    bmesh.ops.recalc_face_normals(bm, faces=new_faces)


_ROOF_BUILDERS = {
    "flat": _generate_flat_roof,
    "modern_flat": _generate_modern_flat_roof,
    "gable": _generate_gable_roof,
    "hip": _generate_hip_roof_poly,
    "shed": _generate_shed_roof,
    # butterfly removed — maps to modern_flat as fallback (see dispatch below)
}

# Roof types that work with arbitrary polygon footprints (convex hull).
# Hip uses _generate_hip_roof_poly which handles any convex polygon.
# Gable and shed require a 4-point bounding rectangle for ridge geometry.
_HULL_COMPATIBLE_ROOFS = {"flat", "modern_flat", "hip"}


# -- Foundation --------------------------------------------------------------

def _generate_foundation_mesh(bm, footprint_hull, config):
    """Add a thickened foundation strip around the building perimeter."""
    fh = config.get("height", 0.3)
    thickness = 0.15  # how far foundation extends beyond walls
    z_bot = -0.05
    z_top = fh

    n = len(footprint_hull)
    if n < 3:
        return

    # Centre for outward expansion direction
    cx = sum(p[0] for p in footprint_hull) / n
    cy = sum(p[1] for p in footprint_hull) / n

    for i in range(n):
        j = (i + 1) % n
        p0 = footprint_hull[i]
        p1 = footprint_hull[j]

        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        elen = math.sqrt(dx * dx + dy * dy)
        if elen < 0.01:
            continue

        # Edge-aligned unit vectors
        eu = (dx / elen, dy / elen)        # along edge
        # Outward perpendicular
        nx, ny = -dy / elen, dx / elen
        mid = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2]
        if (mid[0] + nx - cx) ** 2 + (mid[1] + ny - cy) ** 2 < \
           (mid[0] - nx - cx) ** 2 + (mid[1] - ny - cy) ** 2:
            nx, ny = -nx, -ny
        ev = (nx, ny)  # outward normal

        # Centre of the strip
        strip_cx = mid[0] + nx * thickness / 2
        strip_cy = mid[1] + ny * thickness / 2

        _add_oriented_box(bm,
                          strip_cx, strip_cy, (z_bot + z_top) / 2.0,
                          eu, ev,
                          elen / 2.0 + 0.02,
                          thickness / 2.0 + 0.02,
                          (z_top - z_bot) / 2.0)


# -- Window frames -----------------------------------------------------------

def _generate_window_frame_meshes(bm, floor_plan_data, frame_config, wall_height):
    """Add decorative frames around window openings."""
    frame_depth = frame_config.get("depth", 0.08)
    frame_width = 0.08  # frame strip width
    windows = floor_plan_data.get("windows", [])
    walls = floor_plan_data.get("walls", [])

    for win in windows:
        wi = win.get("wall_index")
        if wi is None or wi >= len(walls):
            continue
        wall = walls[wi]
        s = wall["start"]
        e = wall["end"]
        dx = e[0] - s[0]
        dy = e[1] - s[1]
        wlen = math.sqrt(dx * dx + dy * dy)
        if wlen < 0.1:
            continue

        wu = (dx / wlen, dy / wlen)          # along wall
        wn = (-dy / wlen, dx / wlen)         # perpendicular (outward)

        pos = win.get("position", [0, 0])
        if isinstance(pos, list) and len(pos) >= 2:
            wcx, wcy = pos[0], pos[1]
        else:
            dist = win.get("distance_along", wlen / 2.0)
            wcx = s[0] + wu[0] * dist
            wcy = s[1] + wu[1] * dist

        win_w = win.get("width", 1.2)
        win_h = win.get("height", 1.2)
        sill_h = win.get("sill_height", 0.9)
        fcz = sill_h + win_h / 2.0

        # Offset position along wall normal so frame sits on exterior face
        ox = wcx + wn[0] * frame_depth / 2
        oy = wcy + wn[1] * frame_depth / 2

        # Top lintel
        _add_oriented_box(bm, ox, oy, sill_h + win_h + frame_width / 2,
                          wu, wn,
                          win_w / 2 + frame_width, frame_depth / 2, frame_width / 2)
        # Bottom sill
        _add_oriented_box(bm, ox, oy, sill_h - frame_width / 2,
                          wu, wn,
                          win_w / 2 + frame_width, frame_depth / 2, frame_width / 2)
        # Left jamb
        jlx = wcx - wu[0] * win_w / 2 + wn[0] * frame_depth / 2
        jly = wcy - wu[1] * win_w / 2 + wn[1] * frame_depth / 2
        _add_oriented_box(bm, jlx, jly, fcz,
                          wu, wn,
                          frame_width / 2, frame_depth / 2, win_h / 2 + frame_width)
        # Right jamb
        jrx = wcx + wu[0] * win_w / 2 + wn[0] * frame_depth / 2
        jry = wcy + wu[1] * win_w / 2 + wn[1] * frame_depth / 2
        _add_oriented_box(bm, jrx, jry, fcz,
                          wu, wn,
                          frame_width / 2, frame_depth / 2, win_h / 2 + frame_width)


# -- Exterior door detection --------------------------------------------------

def _point_in_polygon(px, py, polygon):
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if ((yi > py) != (yj > py)) and \
                (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _door_leads_outside(door, walls, rooms):
    """Return True if one side of the door opens to outside (no room).

    IMPORTANT: The door "position" field is the raw AI-detected coordinate
    which may be far from the actual wall.  We must PROJECT it onto the
    wall first to get the true door centre, then probe each side.

    Tests a point 0.6 m out from the projected door centre on each side
    of the wall.  If one side is NOT inside any room polygon, the door
    leads outside and is a genuine exterior / entry door.
    """
    di = door.get("wall_index")
    if di is None or di >= len(walls):
        return False

    wall = walls[di]
    s = wall["start"]
    e = wall["end"]
    dx = e[0] - s[0]
    dy = e[1] - s[1]
    wlen = math.sqrt(dx * dx + dy * dy)
    if wlen < 0.1:
        return False

    wu = (dx / wlen, dy / wlen)
    wn = (-dy / wlen, dx / wlen)   # perpendicular

    # Project the door position onto the wall to get the true centre
    pos = door.get("position", [0, 0])
    if isinstance(pos, list) and len(pos) >= 2:
        # Project [x,y] onto the wall segment
        dpx = pos[0] - s[0]
        dpy = pos[1] - s[1]
        dist_along = dpx * wu[0] + dpy * wu[1]
    else:
        dist_along = float(pos)

    # Clamp to wall length
    dist_along = max(0.0, min(wlen, dist_along))

    # True door centre ON the wall
    dcx = s[0] + wu[0] * dist_along
    dcy = s[1] + wu[1] * dist_along

    # Test points on each side of the wall, 0.6 m out from door centre
    probe = 0.6
    side_a = (dcx + wn[0] * probe, dcy + wn[1] * probe)
    side_b = (dcx - wn[0] * probe, dcy - wn[1] * probe)

    room_polys = [r.get("polygon", []) for r in rooms if r.get("polygon")]

    a_in = any(_point_in_polygon(side_a[0], side_a[1], p) for p in room_polys)
    b_in = any(_point_in_polygon(side_b[0], side_b[1], p) for p in room_polys)

    # Exterior door: one side in a room, the other side outside all rooms
    return (a_in != b_in)


# -- Door surrounds ----------------------------------------------------------

def _generate_door_surround_meshes(bm, floor_plan_data, surround_config,
                                   wall_height, footprint_hull=None):
    """Add decorative surrounds around exterior door openings."""
    sur_w = surround_config.get("width", 0.12)
    sur_depth = 0.06
    doors = floor_plan_data.get("doors", [])
    walls = floor_plan_data.get("walls", [])
    rooms = floor_plan_data.get("rooms", [])

    for door in doors:
        di = door.get("wall_index")
        if di is None or di >= len(walls):
            continue

        # Skip interior doors — surrounds only on doors that lead outside
        if rooms and not _door_leads_outside(door, walls, rooms):
            continue

        wall = walls[di]
        s = wall["start"]
        e = wall["end"]
        dx = e[0] - s[0]
        dy = e[1] - s[1]
        wlen = math.sqrt(dx * dx + dy * dy)
        if wlen < 0.1:
            continue

        wu = (dx / wlen, dy / wlen)
        wn = (-dy / wlen, dx / wlen)

        pos = door.get("position", [0, 0])
        if isinstance(pos, list) and len(pos) >= 2:
            dcx, dcy = pos[0], pos[1]
        else:
            dist = door.get("distance_along", wlen / 2.0)
            dcx = s[0] + wu[0] * dist
            dcy = s[1] + wu[1] * dist

        door_w = door.get("width", 0.9)
        door_h = door.get("height", 2.1)

        ox = dcx + wn[0] * sur_depth / 2
        oy = dcy + wn[1] * sur_depth / 2

        # Lintel
        _add_oriented_box(bm, ox, oy, door_h + sur_w / 2,
                          wu, wn,
                          door_w / 2 + sur_w, sur_depth / 2, sur_w / 2)
        # Left jamb
        jlx = dcx - wu[0] * door_w / 2 + wn[0] * sur_depth / 2
        jly = dcy - wu[1] * door_w / 2 + wn[1] * sur_depth / 2
        _add_oriented_box(bm, jlx, jly, door_h / 2,
                          wu, wn,
                          sur_w / 2, sur_depth / 2, door_h / 2)
        # Right jamb
        jrx = dcx + wu[0] * door_w / 2 + wn[0] * sur_depth / 2
        jry = dcy + wu[1] * door_w / 2 + wn[1] * sur_depth / 2
        _add_oriented_box(bm, jrx, jry, door_h / 2,
                          wu, wn,
                          sur_w / 2, sur_depth / 2, door_h / 2)


# -- Exterior door panels ----------------------------------------------------

def _generate_door_panel_meshes(bm, floor_plan_data, footprint_hull,
                                wall_height):
    """Generate visible door-leaf panels flush with the exterior wall face.

    Only creates panels for doors on PERIMETER walls (close to the hull
    edge).  Interior doors between rooms are skipped — they wouldn't be
    visible from outside and would clip through interior partitions.

    Exterior walls sit at:  hull_edge + 0.10 (gap) + 0.20 (thickness)
    So the panel outer face is placed at ~0.30 m outward from the hull edge.
    """
    panel_depth = 0.05  # 5 cm door leaf
    ext_wall_offset = 0.10
    ext_wall_thick = 0.20
    face_offset = ext_wall_offset + ext_wall_thick - panel_depth / 2

    doors = floor_plan_data.get("doors", [])
    walls = floor_plan_data.get("walls", [])
    rooms = floor_plan_data.get("rooms", [])
    if not doors or not walls:
        return

    # Building centroid — used to determine outward direction
    n_h = len(footprint_hull)
    bcx = sum(p[0] for p in footprint_hull) / n_h
    bcy = sum(p[1] for p in footprint_hull) / n_h

    for door in doors:
        di = door.get("wall_index")
        if di is None or di >= len(walls):
            continue

        # Only show panels for doors that lead outside
        if rooms and not _door_leads_outside(door, walls, rooms):
            continue

        wall = walls[di]
        s = wall["start"]
        e = wall["end"]
        dx = e[0] - s[0]
        dy = e[1] - s[1]
        wlen = math.sqrt(dx * dx + dy * dy)
        if wlen < 0.1:
            continue

        wu = (dx / wlen, dy / wlen)
        wn = (-dy / wlen, dx / wlen)

        # Door centre position
        pos = door.get("position", [0, 0])
        if isinstance(pos, list) and len(pos) >= 2:
            dcx, dcy = pos[0], pos[1]
        else:
            dist = door.get("distance_along", wlen / 2.0)
            dcx = s[0] + wu[0] * dist
            dcy = s[1] + wu[1] * dist

        # Ensure wn points OUTWARD (away from building centroid)
        test_x = dcx + wn[0]
        test_y = dcy + wn[1]
        if ((test_x - bcx) ** 2 + (test_y - bcy) ** 2 <
                (dcx - bcx) ** 2 + (dcy - bcy) ** 2):
            wn = (-wn[0], -wn[1])

        door_w = door.get("width", 0.9)
        door_h = door.get("height", 2.1)

        # Place panel on the exterior wall outer face
        ox = dcx + wn[0] * face_offset
        oy = dcy + wn[1] * face_offset

        _add_oriented_box(bm,
                          ox, oy, door_h / 2,
                          wu, wn,
                          door_w / 2,
                          panel_depth / 2,
                          door_h / 2)


# -- Exterior walls -----------------------------------------------------------

def _hull_outward_normals(footprint_hull):
    """Compute outward-facing unit normals for each edge of the hull."""
    n = len(footprint_hull)
    cx = sum(p[0] for p in footprint_hull) / n
    cy = sum(p[1] for p in footprint_hull) / n
    normals = []
    for i in range(n):
        j = (i + 1) % n
        p0, p1 = footprint_hull[i], footprint_hull[j]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        elen = math.sqrt(dx * dx + dy * dy)
        if elen < 0.01:
            normals.append((0, 0))
            continue
        nx, ny = -dy / elen, dx / elen
        mid = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2]
        if (mid[0] + nx - cx) ** 2 + (mid[1] + ny - cy) ** 2 < \
           (mid[0] - nx - cx) ** 2 + (mid[1] - ny - cy) ** 2:
            nx, ny = -nx, -ny
        normals.append((nx, ny))
    return normals


def _generate_exterior_walls(bm, footprint_hull, wall_height, wall_thickness=0.20):
    """Generate thick exterior wall panels pushed outward from the building hull.

    Creates a visible outer shell — 0.20m thick wall sections offset 0.1m
    outside the convex hull. Each segment is an oriented box so it works
    on non-axis-aligned edges.
    """
    n = len(footprint_hull)
    if n < 3:
        return

    normals = _hull_outward_normals(footprint_hull)
    offset = 0.10  # gap between hull edge and inner face of ext wall

    for i in range(n):
        j = (i + 1) % n
        p0, p1 = footprint_hull[i], footprint_hull[j]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        elen = math.sqrt(dx * dx + dy * dy)
        if elen < 0.01:
            continue

        nx, ny = normals[i]
        eu = (dx / elen, dy / elen)
        ev = (nx, ny)

        # Centre of the wall panel (pushed outward)
        wcx = (p0[0] + p1[0]) / 2 + nx * (offset + wall_thickness / 2)
        wcy = (p0[1] + p1[1]) / 2 + ny * (offset + wall_thickness / 2)

        _add_oriented_box(bm,
                          wcx, wcy, wall_height / 2,
                          eu, ev,
                          elen / 2 + 0.01,       # half-length along edge
                          wall_thickness / 2,      # half-thickness
                          wall_height / 2)         # half-height


def _generate_gable_fill_walls(bm, rect_corners, config, wall_height):
    """Generate triangular gable fill walls at the short ends of a gable roof.

    Only applies to gable roofs — hip roofs create their own triangular faces
    as part of the hip builder, and shed/flat roofs don't need fill walls.
    """
    roof_type = config.get("type", "gable").lower()
    if roof_type != "gable":
        return  # only gable roofs need triangular fill walls

    pitch = math.radians(config.get("pitch_degrees", 30))
    long_len, short_len, lu, pu, eave_a, eave_b, centre = _roof_rect_helpers(rect_corners)

    half_span = short_len / 2.0
    ridge_rise = math.tan(pitch) * half_span
    ridge_z = wall_height + ridge_rise

    # Two triangular gable walls at the short ends
    for ea_pt, eb_pt in [(eave_a[0], eave_b[0]), (eave_a[1], eave_b[1])]:
        v0 = bm.verts.new((ea_pt[0], ea_pt[1], wall_height))
        v1 = bm.verts.new((eb_pt[0], eb_pt[1], wall_height))
        rx = (ea_pt[0] + eb_pt[0]) / 2
        ry = (ea_pt[1] + eb_pt[1]) / 2
        vr = bm.verts.new((rx, ry, ridge_z))
        bm.faces.new((v0, v1, vr))


# -- Architectural details ---------------------------------------------------

def _generate_cornice_mesh(bm, footprint_hull, detail, wall_height):
    """Add a decorative cornice band running around the building at roofline.

    A horizontal projection just below the roof line — one of the most
    visually defining features of any building exterior.
    """
    projection = detail.get("projection", 0.12)
    height = detail.get("height", 0.15)
    z_bot = wall_height - height
    z_top = wall_height

    n = len(footprint_hull)
    if n < 3:
        return

    cx = sum(p[0] for p in footprint_hull) / n
    cy = sum(p[1] for p in footprint_hull) / n

    for i in range(n):
        j = (i + 1) % n
        p0, p1 = footprint_hull[i], footprint_hull[j]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        elen = math.sqrt(dx * dx + dy * dy)
        if elen < 0.01:
            continue
        eu = (dx / elen, dy / elen)
        nx, ny = -dy / elen, dx / elen
        mid = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2]
        if (mid[0] + nx - cx) ** 2 + (mid[1] + ny - cy) ** 2 < \
           (mid[0] - nx - cx) ** 2 + (mid[1] - ny - cy) ** 2:
            nx, ny = -nx, -ny

        scx = mid[0] + nx * projection / 2
        scy = mid[1] + ny * projection / 2
        _add_oriented_box(bm, scx, scy, (z_bot + z_top) / 2,
                          eu, (nx, ny),
                          elen / 2 + projection,
                          projection / 2 + 0.02,
                          height / 2)


def _generate_window_sills_mesh(bm, floor_plan_data, detail, wall_height):
    """Add protruding sills / ledges beneath each window."""
    sill_depth = detail.get("depth", 0.15)
    sill_thickness = detail.get("thickness", 0.05)
    windows = floor_plan_data.get("windows", [])
    walls = floor_plan_data.get("walls", [])

    for win in windows:
        wi = win.get("wall_index")
        if wi is None or wi >= len(walls):
            continue
        wall = walls[wi]
        s, e = wall["start"], wall["end"]
        dx, dy = e[0] - s[0], e[1] - s[1]
        wlen = math.sqrt(dx * dx + dy * dy)
        if wlen < 0.1:
            continue
        wu = (dx / wlen, dy / wlen)
        wn = (-dy / wlen, dx / wlen)

        pos = win.get("position", [0, 0])
        if isinstance(pos, list) and len(pos) >= 2:
            wcx, wcy = pos[0], pos[1]
        else:
            dist = win.get("distance_along", wlen / 2.0)
            wcx = s[0] + wu[0] * dist
            wcy = s[1] + wu[1] * dist

        win_w = win.get("width", 1.2)
        sill_h = win.get("sill_height", 0.9)

        sx = wcx + wn[0] * sill_depth / 2
        sy = wcy + wn[1] * sill_depth / 2
        _add_oriented_box(bm, sx, sy, sill_h - sill_thickness / 2,
                          wu, wn,
                          win_w / 2 + 0.05,
                          sill_depth / 2,
                          sill_thickness / 2)


def _generate_pilasters_mesh(bm, footprint_hull, detail, wall_height):
    """Add vertical pilasters (flat columns) at building corners.

    Pilasters define the building edges and add visual rhythm.
    """
    width = detail.get("width", 0.20)
    depth = detail.get("depth", 0.08)

    n = len(footprint_hull)
    if n < 3:
        return

    normals = _hull_outward_normals(footprint_hull)

    for i in range(n):
        pt = footprint_hull[i]
        # Average the normals of the two edges meeting at this corner
        prev = (i - 1) % n
        nx = (normals[prev][0] + normals[i][0])
        ny = (normals[prev][1] + normals[i][1])
        nlen = math.sqrt(nx * nx + ny * ny) or 1.0
        nx, ny = nx / nlen, ny / nlen

        # u direction = perpendicular to outward normal (along wall face)
        ux, uy = -ny, nx
        pcx = pt[0] + nx * depth / 2
        pcy = pt[1] + ny * depth / 2

        _add_oriented_box(bm, pcx, pcy, wall_height / 2,
                          (ux, uy), (nx, ny),
                          width / 2, depth / 2, wall_height / 2)


def _generate_canopy_mesh(bm, floor_plan_data, detail, wall_height):
    """Add a flat canopy/awning over entrance door(s).

    Creates a horizontal slab above door openings with thin support columns.
    """
    canopy_depth = detail.get("depth", 1.2)
    canopy_thickness = detail.get("thickness", 0.12)
    door_indices = detail.get("door_indices", None)
    doors = floor_plan_data.get("doors", [])
    walls = floor_plan_data.get("walls", [])

    if not doors:
        return

    targets = []
    if door_indices:
        targets = [doors[i] for i in door_indices if i < len(doors)]
    else:
        # Default: canopy over the first door
        targets = [doors[0]]

    for door in targets:
        di = door.get("wall_index")
        if di is None or di >= len(walls):
            continue
        wall = walls[di]
        s, e = wall["start"], wall["end"]
        dx, dy = e[0] - s[0], e[1] - s[1]
        wlen = math.sqrt(dx * dx + dy * dy)
        if wlen < 0.1:
            continue
        wu = (dx / wlen, dy / wlen)
        wn = (-dy / wlen, dx / wlen)

        pos = door.get("position", [0, 0])
        if isinstance(pos, list) and len(pos) >= 2:
            dcx, dcy = pos[0], pos[1]
        else:
            dist = door.get("distance_along", wlen / 2.0)
            dcx = s[0] + wu[0] * dist
            dcy = s[1] + wu[1] * dist

        door_w = door.get("width", 0.9)
        door_h = door.get("height", 2.1)
        canopy_z = door_h + 0.15

        # Canopy slab
        canopy_w = door_w + 0.6  # wider than the door
        cx_off = dcx + wn[0] * canopy_depth / 2
        cy_off = dcy + wn[1] * canopy_depth / 2
        _add_oriented_box(bm, cx_off, cy_off, canopy_z,
                          wu, wn,
                          canopy_w / 2, canopy_depth / 2, canopy_thickness / 2)

        # Two thin support columns
        col_r = 0.04
        col_h = canopy_z - 0.1
        for side in (-1, 1):
            colx = dcx + wu[0] * (door_w / 2 + 0.15) * side + wn[0] * (canopy_depth - 0.1)
            coly = dcy + wu[1] * (door_w / 2 + 0.15) * side + wn[1] * (canopy_depth - 0.1)
            _add_oriented_box(bm, colx, coly, col_h / 2,
                              wu, wn,
                              col_r, col_r, col_h / 2)


def _generate_accent_band_mesh(bm, footprint_hull, detail, wall_height):
    """Add a thin horizontal accent band/belt course around the building.

    Breaks up large wall surfaces and adds visual interest.
    """
    z_pos = detail.get("z_position", wall_height * 0.4)
    band_h = detail.get("height", 0.05)
    projection = detail.get("projection", 0.03)

    n = len(footprint_hull)
    if n < 3:
        return

    cx = sum(p[0] for p in footprint_hull) / n
    cy = sum(p[1] for p in footprint_hull) / n

    for i in range(n):
        j = (i + 1) % n
        p0, p1 = footprint_hull[i], footprint_hull[j]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        elen = math.sqrt(dx * dx + dy * dy)
        if elen < 0.01:
            continue
        eu = (dx / elen, dy / elen)
        nx, ny = -dy / elen, dx / elen
        mid = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2]
        if (mid[0] + nx - cx) ** 2 + (mid[1] + ny - cy) ** 2 < \
           (mid[0] - nx - cx) ** 2 + (mid[1] - ny - cy) ** 2:
            nx, ny = -nx, -ny

        scx = mid[0] + nx * projection / 2
        scy = mid[1] + ny * projection / 2
        _add_oriented_box(bm, scx, scy, z_pos,
                          eu, (nx, ny),
                          elen / 2 + 0.02,
                          projection / 2 + 0.01,
                          band_h / 2)


def _generate_chimney_mesh(bm, footprint_hull, detail, wall_height, roof_cfg):
    """Add a chimney rising above the roofline."""
    chim_w = detail.get("width", 0.6)
    chim_d = detail.get("depth", 0.4)
    # Place chimney near the back of the building

    n = len(footprint_hull)
    if n < 3:
        return


    # Pick the back edge (furthest from index 0)
    best_i = 0
    best_dist = 0
    for i in range(n):
        j = (i + 1) % n
        mid = [(footprint_hull[i][0] + footprint_hull[j][0]) / 2,
               (footprint_hull[i][1] + footprint_hull[j][1]) / 2]
        d = (mid[0] - footprint_hull[0][0]) ** 2 + (mid[1] - footprint_hull[0][1]) ** 2
        if d > best_dist:
            best_dist = d
            best_i = i

    j = (best_i + 1) % n
    p0, p1 = footprint_hull[best_i], footprint_hull[j]
    mid = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    elen = math.sqrt(dx * dx + dy * dy) or 1.0
    eu = (dx / elen, dy / elen)
    ev = (-dy / elen, dx / elen)

    # Chimney height: above roof peak
    # Approximate
    roof_peak_z = wall_height + 2.0  # safe above most roofs
    chim_top = roof_peak_z + 0.8
    chim_bot = wall_height - 0.5  # starts below roofline

    _add_oriented_box(bm, mid[0], mid[1], (chim_bot + chim_top) / 2,
                      eu, ev,
                      chim_w / 2, chim_d / 2, (chim_top - chim_bot) / 2)
    # Chimney cap (wider thin slab on top)
    _add_oriented_box(bm, mid[0], mid[1], chim_top + 0.03,
                      eu, ev,
                      chim_w / 2 + 0.05, chim_d / 2 + 0.05, 0.03)


def _generate_balcony_mesh(bm, floor_plan_data, footprint_hull, detail, wall_height):
    """Add balconies — cantilevered platforms with glass/metal railings.

    Places a balcony on the longest hull edge (most visible facade).
    """
    bal_depth = detail.get("depth", 1.5)
    bal_width = detail.get("width", 3.0)
    bal_z = detail.get("z_position", wall_height * 0.55)
    slab_thick = 0.15
    railing_h = detail.get("railing_height", 1.0)
    railing_thick = 0.04

    n = len(footprint_hull)
    if n < 3:
        return

    cx = sum(p[0] for p in footprint_hull) / n
    cy = sum(p[1] for p in footprint_hull) / n

    # Find the longest hull edge for placement
    best_i = 0
    best_len = 0
    for i in range(n):
        j = (i + 1) % n
        p0, p1 = footprint_hull[i], footprint_hull[j]
        d = math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
        if d > best_len:
            best_len = d
            best_i = i

    j = (best_i + 1) % n
    p0, p1 = footprint_hull[best_i], footprint_hull[j]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    elen = math.sqrt(dx * dx + dy * dy) or 1.0
    eu = (dx / elen, dy / elen)
    nx, ny = -dy / elen, dx / elen
    mid = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2]
    if (mid[0] + nx - cx) ** 2 + (mid[1] + ny - cy) ** 2 < \
       (mid[0] - nx - cx) ** 2 + (mid[1] - ny - cy) ** 2:
        nx, ny = -nx, -ny

    bw = min(bal_width, elen * 0.6)

    # Slab — extends outward from wall face
    scx = mid[0] + nx * bal_depth / 2
    scy = mid[1] + ny * bal_depth / 2
    _add_oriented_box(bm, scx, scy, bal_z,
                      eu, (nx, ny),
                      bw / 2, bal_depth / 2, slab_thick / 2)

    # Front railing (along outer edge)
    fcx = mid[0] + nx * bal_depth
    fcy = mid[1] + ny * bal_depth
    _add_oriented_box(bm, fcx, fcy, bal_z + railing_h / 2 + slab_thick / 2,
                      eu, (nx, ny),
                      bw / 2, railing_thick / 2, railing_h / 2)

    # Two side railings
    for side in (-1, 1):
        srx = mid[0] + eu[0] * bw / 2 * side + nx * bal_depth / 2
        sry = mid[1] + eu[1] * bw / 2 * side + ny * bal_depth / 2
        _add_oriented_box(bm, srx, sry, bal_z + railing_h / 2 + slab_thick / 2,
                          eu, (nx, ny),
                          railing_thick / 2, bal_depth / 2, railing_h / 2)


def _generate_pergola_mesh(bm, floor_plan_data, footprint_hull, detail, wall_height):
    """Add a pergola — open-frame structure with columns and cross-beams.

    Placed along one side of the building. Very architectural/modern.
    """
    pg_depth = detail.get("depth", 2.5)
    pg_width = detail.get("width", 4.0)
    col_size = 0.12
    beam_h = 0.15
    beam_w = 0.08
    n_beams = detail.get("beam_count", 6)
    pg_height = detail.get("height", wall_height * 0.85)

    n = len(footprint_hull)
    if n < 3:
        return

    cx = sum(p[0] for p in footprint_hull) / n
    cy = sum(p[1] for p in footprint_hull) / n

    # Pick longest edge
    best_i = 0
    best_len = 0
    for i in range(n):
        j = (i + 1) % n
        p0, p1 = footprint_hull[i], footprint_hull[j]
        d = math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
        if d > best_len:
            best_len = d
            best_i = i

    j = (best_i + 1) % n
    p0, p1 = footprint_hull[best_i], footprint_hull[j]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    elen = math.sqrt(dx * dx + dy * dy) or 1.0
    eu = (dx / elen, dy / elen)
    nx, ny = -dy / elen, dx / elen
    mid = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2]
    if (mid[0] + nx - cx) ** 2 + (mid[1] + ny - cy) ** 2 < \
       (mid[0] - nx - cx) ** 2 + (mid[1] - ny - cy) ** 2:
        nx, ny = -nx, -ny

    pw = min(pg_width, elen * 0.7)

    # 4 columns (corners of pergola rectangle)
    for su in (-1, 1):
        for sv in (0.1, 1.0):
            colx = mid[0] + eu[0] * pw / 2 * su + nx * pg_depth * sv
            coly = mid[1] + eu[1] * pw / 2 * su + ny * pg_depth * sv
            _add_oriented_box(bm, colx, coly, pg_height / 2,
                              eu, (nx, ny),
                              col_size / 2, col_size / 2, pg_height / 2)

    # Two main beams along the long axis (on top of columns)
    beam_z = pg_height + beam_h / 2
    for sv in (0.1, 1.0):
        bx = mid[0] + nx * pg_depth * sv
        by = mid[1] + ny * pg_depth * sv
        _add_oriented_box(bm, bx, by, beam_z,
                          eu, (nx, ny),
                          pw / 2 + 0.1, beam_w / 2, beam_h / 2)

    # Cross beams (perpendicular)
    for k in range(n_beams):
        t = (k + 0.5) / n_beams
        cbx = mid[0] + eu[0] * (t - 0.5) * pw + nx * pg_depth * 0.55
        cby = mid[1] + eu[1] * (t - 0.5) * pw + ny * pg_depth * 0.55
        _add_oriented_box(bm, cbx, cby, beam_z + beam_h,
                          (nx, ny), eu,
                          pg_depth * 0.5 + 0.1, beam_w / 2, beam_h / 2)


def _generate_louver_screen_mesh(bm, floor_plan_data, footprint_hull, detail, wall_height):
    """Add a louver/fin screen — series of vertical fins for sun-shading.

    Distinctive modern/tropical look. Placed on a building face.
    """
    fin_count = detail.get("fin_count", 12)
    fin_depth = detail.get("fin_depth", 0.4)
    fin_thick = detail.get("fin_thickness", 0.03)
    fin_height = detail.get("height", wall_height * 0.6)
    z_base = detail.get("z_position", wall_height * 0.25)
    standoff = detail.get("standoff", 0.15)

    n = len(footprint_hull)
    if n < 3:
        return

    cx = sum(p[0] for p in footprint_hull) / n
    cy = sum(p[1] for p in footprint_hull) / n

    # Find second-longest edge (to differentiate from balcony/pergola side)
    edges = []
    for i in range(n):
        j = (i + 1) % n
        p0, p1 = footprint_hull[i], footprint_hull[j]
        d = math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
        edges.append((d, i))
    edges.sort(reverse=True)
    ei = edges[min(1, len(edges) - 1)][1]

    j = (ei + 1) % n
    p0, p1 = footprint_hull[ei], footprint_hull[j]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    elen = math.sqrt(dx * dx + dy * dy) or 1.0
    eu = (dx / elen, dy / elen)
    nx, ny = -dy / elen, dx / elen
    mid = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2]
    if (mid[0] + nx - cx) ** 2 + (mid[1] + ny - cy) ** 2 < \
       (mid[0] - nx - cx) ** 2 + (mid[1] - ny - cy) ** 2:
        nx, ny = -nx, -ny

    screen_w = elen * 0.7
    fc = min(fin_count, int(screen_w / 0.15))
    fz = z_base + fin_height / 2

    for k in range(fc):
        t = (k + 0.5) / fc
        fx = mid[0] + eu[0] * (t - 0.5) * screen_w + nx * (standoff + fin_depth / 2)
        fy = mid[1] + eu[1] * (t - 0.5) * screen_w + ny * (standoff + fin_depth / 2)
        _add_oriented_box(bm, fx, fy, fz,
                          eu, (nx, ny),
                          fin_thick / 2, fin_depth / 2, fin_height / 2)


def _generate_planter_box_mesh(bm, footprint_hull, detail, wall_height):
    """Add raised planter boxes along building base.

    Greenery containers that sit against the foundation — softens the
    building edge and adds life.
    """
    box_h = detail.get("height", 0.5)
    box_d = detail.get("depth", 0.5)
    box_w = detail.get("width", 2.0)

    n = len(footprint_hull)
    if n < 3:
        return

    cx = sum(p[0] for p in footprint_hull) / n
    cy = sum(p[1] for p in footprint_hull) / n

    # Place a planter on each edge long enough
    for i in range(n):
        j = (i + 1) % n
        p0, p1 = footprint_hull[i], footprint_hull[j]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        elen = math.sqrt(dx * dx + dy * dy)
        if elen < box_w * 1.2:
            continue  # edge too short

        eu = (dx / elen, dy / elen)
        nx, ny = -dy / elen, dx / elen
        mid = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2]
        if (mid[0] + nx - cx) ** 2 + (mid[1] + ny - cy) ** 2 < \
           (mid[0] - nx - cx) ** 2 + (mid[1] - ny - cy) ** 2:
            nx, ny = -nx, -ny

        bx = mid[0] + nx * box_d / 2
        by = mid[1] + ny * box_d / 2
        pw = min(box_w, elen * 0.4)
        _add_oriented_box(bm, bx, by, box_h / 2,
                          eu, (nx, ny),
                          pw / 2, box_d / 2, box_h / 2)


def _generate_feature_wall_mesh(bm, footprint_hull, detail, wall_height):
    """Add a feature/accent wall that rises above the roofline.

    A signature element of modern architecture — a single plane that
    extends vertically past the main building volume.
    """
    extra_h = detail.get("extra_height", 1.5)
    wall_thick = detail.get("thickness", 0.15)

    n = len(footprint_hull)
    if n < 3:
        return

    cx = sum(p[0] for p in footprint_hull) / n
    cy = sum(p[1] for p in footprint_hull) / n

    # Use the shortest hull edge (typically a side/end wall)
    edges = []
    for i in range(n):
        j = (i + 1) % n
        p0, p1 = footprint_hull[i], footprint_hull[j]
        d = math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
        edges.append((d, i))
    edges.sort()
    ei = edges[0][1]

    j = (ei + 1) % n
    p0, p1 = footprint_hull[ei], footprint_hull[j]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    elen = math.sqrt(dx * dx + dy * dy) or 1.0
    eu = (dx / elen, dy / elen)
    nx, ny = -dy / elen, dx / elen
    mid = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2]
    if (mid[0] + nx - cx) ** 2 + (mid[1] + ny - cy) ** 2 < \
       (mid[0] - nx - cx) ** 2 + (mid[1] - ny - cy) ** 2:
        nx, ny = -nx, -ny

    total_h = wall_height + extra_h
    _add_oriented_box(bm, mid[0], mid[1], total_h / 2,
                      eu, (nx, ny),
                      elen / 2 + 0.1, wall_thick / 2, total_h / 2)


# ---------------------------------------------------------------------------
# Raised volume — interlocking box volumes at different heights (MCM signature)
# ---------------------------------------------------------------------------

def _sub_rect_along_long_axis(rect_corners, coverage):
    """Return 4 corners of a sub-rectangle covering *coverage* from one end.

    Splits the bounding rectangle along its long axis and returns the
    portion at the "far" end (near corners 1/2 if 0→1 is long, or near
    corners 2/3 if 1→2 is long).
    """
    coverage = max(0.15, min(0.75, coverage))
    t = 1.0 - coverage

    def _lerp(a, b, f):
        return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])]

    d01 = math.sqrt((rect_corners[1][0] - rect_corners[0][0]) ** 2 +
                     (rect_corners[1][1] - rect_corners[0][1]) ** 2)
    d12 = math.sqrt((rect_corners[2][0] - rect_corners[1][0]) ** 2 +
                     (rect_corners[2][1] - rect_corners[1][1]) ** 2)

    if d01 >= d12:
        # Long axis: 0→1 and 3→2
        return [_lerp(rect_corners[0], rect_corners[1], t),
                list(rect_corners[1]),
                list(rect_corners[2]),
                _lerp(rect_corners[3], rect_corners[2], t)]
    else:
        # Long axis: 1→2 and 0→3
        return [list(rect_corners[0]),
                list(rect_corners[1]),
                _lerp(rect_corners[1], rect_corners[2], t),
                _lerp(rect_corners[0], rect_corners[3], t)]


def _generate_raised_volume_mesh(bm, floor_plan_data, footprint_hull,
                                  detail, wall_height, roof_config):
    """Create a raised roof slab with clerestory walls below it.

    Signature element of mid-century modern and contemporary architecture:
    interlocking rectangular volumes at different heights create dynamic,
    stepped massing.

    Geometry (bottom-up):
    1. Clerestory walls — short walls from main-roof parapet top up to the
       raised slab.  These follow the sub-rect (NO overhang) so the slab
       overhangs them creating a shadow reveal.  They fill the visible gap
       without covering the original building walls / windows below.
    2. Raised roof slab — thick slab at the elevated height with overhang.
    """
    extra_h = min(detail.get("extra_height", 1.2), 1.5)
    coverage = min(detail.get("coverage", 0.4), 0.5)
    oh = min(detail.get("overhang", 0.3), 0.4)
    slab_t = 0.25  # thicker slab for visual weight

    hull_pts = [(p[0], p[1]) for p in footprint_hull]
    hull = _convex_hull(hull_pts)
    rect, _, _ = _min_bounding_rect(hull)
    if not rect or len(rect) < 4:
        return

    sub = _sub_rect_along_long_axis(rect, coverage)
    expanded = _expand_corners(sub, oh)

    # Roof slab at elevated height
    slab_bot = wall_height + extra_h
    slab_top = slab_bot + slab_t

    n = len(expanded)
    v_bot = [bm.verts.new((p[0], p[1], slab_bot)) for p in expanded]
    v_top = [bm.verts.new((p[0], p[1], slab_top)) for p in expanded]

    new_faces = []
    # Top face (roof surface)
    new_faces.append(bm.faces.new(v_top))
    # Bottom face (soffit — visible from below)
    new_faces.append(bm.faces.new(list(reversed(v_bot))))
    # Edge faces (fascia strip around the slab perimeter)
    for i in range(n):
        j = (i + 1) % n
        new_faces.append(bm.faces.new((v_bot[i], v_bot[j], v_top[j], v_top[i])))

    bmesh.ops.recalc_face_normals(bm, faces=new_faces)

    # --- Clerestory walls (fill gap between main roof and raised slab) ---
    parapet_h = roof_config.get("parapet_height", 0.4)
    main_roof_z = wall_height + parapet_h
    clere_h = slab_bot - main_roof_z

    if clere_h > 0.1:
        # Walls follow the sub-rect (inset from slab by overhang → shadow reveal)
        wall_thick = 0.15
        n_sub = len(sub)
        for i in range(n_sub):
            j = (i + 1) % n_sub
            p0, p1 = sub[i], sub[j]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            seg_len = math.sqrt(dx * dx + dy * dy)
            if seg_len < 0.1:
                continue
            eu = (dx / seg_len, dy / seg_len)
            en = (-dy / seg_len, dx / seg_len)
            mid_x = (p0[0] + p1[0]) / 2
            mid_y = (p0[1] + p1[1]) / 2
            mid_z = main_roof_z + clere_h / 2

            _add_oriented_box(bm, mid_x, mid_y, mid_z,
                              eu, en,
                              seg_len / 2, wall_thick / 2, clere_h / 2)


# ---------------------------------------------------------------------------
# New high-impact architectural detail builders
# ---------------------------------------------------------------------------

def _find_longest_hull_edge(footprint_hull):
    """Return (edge_index, edge_length) for the longest hull edge."""
    n = len(footprint_hull)
    best_i, best_len = 0, 0
    for i in range(n):
        j = (i + 1) % n
        p0, p1 = footprint_hull[i], footprint_hull[j]
        d = math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
        if d > best_len:
            best_i, best_len = i, d
    return best_i, best_len


def _hull_edge_geometry(footprint_hull, edge_index):
    """Compute edge unit tangent (eu), outward normal (nx,ny), midpoint, and length."""
    n = len(footprint_hull)
    j = (edge_index + 1) % n
    p0 = footprint_hull[edge_index]
    p1 = footprint_hull[j]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    elen = math.sqrt(dx * dx + dy * dy) or 1.0
    eu = (dx / elen, dy / elen)
    nx, ny = -dy / elen, dx / elen
    mid = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2]

    # Ensure normal points outward (away from hull centroid)
    cx = sum(p[0] for p in footprint_hull) / n
    cy = sum(p[1] for p in footprint_hull) / n
    if (mid[0] + nx - cx) ** 2 + (mid[1] + ny - cy) ** 2 < \
       (mid[0] - nx - cx) ** 2 + (mid[1] - ny - cy) ** 2:
        nx, ny = -nx, -ny

    return eu, (nx, ny), mid, elen


def _generate_glass_wall_mesh(bm, floor_plan_data, footprint_hull, detail, wall_height):
    """Add a glass curtain wall with mullion grid on one building face.

    THE defining feature of mid-century modern architecture — an entire
    wall section replaced by floor-to-ceiling glass with thin mullion bars.

    Note: this builder puts ALL geometry (glass + mullions) into the same
    bmesh.  The orchestrator in generate_exterior() creates a SECOND object
    for the glass panel so it can receive a different (transparent) material.
    """
    n = len(footprint_hull)
    if n < 3:
        return

    mullion_spacing = detail.get("mullion_spacing", 1.2)
    mullion_w = detail.get("mullion_width", 0.04)
    frame_depth = 0.06
    sill_h = detail.get("sill_height", 0.0)
    glass_h = detail.get("height", wall_height - sill_h - 0.05)
    if glass_h <= 0:
        glass_h = wall_height * 0.85

    # Pick target edge (longest = main facade)
    ei, _ = _find_longest_hull_edge(footprint_hull)
    eu, (nx, ny), mid, elen = _hull_edge_geometry(footprint_hull, ei)

    glass_w = elen * 0.88  # Leave corner margins
    z_center = sill_h + glass_h / 2

    # --- Mullion frame (opaque material) ---
    # Outer frame — 4 bars around the glass area
    # Top bar
    _add_oriented_box(bm, mid[0] + nx * frame_depth / 2,
                      mid[1] + ny * frame_depth / 2,
                      sill_h + glass_h - mullion_w / 2,
                      eu, (nx, ny),
                      glass_w / 2, frame_depth / 2, mullion_w / 2)
    # Bottom bar
    _add_oriented_box(bm, mid[0] + nx * frame_depth / 2,
                      mid[1] + ny * frame_depth / 2,
                      sill_h + mullion_w / 2,
                      eu, (nx, ny),
                      glass_w / 2, frame_depth / 2, mullion_w / 2)

    # Vertical mullions at regular spacing
    n_mullions = max(1, int(glass_w / mullion_spacing)) + 1
    for k in range(n_mullions):
        t = k / max(n_mullions - 1, 1) - 0.5  # -0.5 to +0.5
        mx = mid[0] + eu[0] * t * glass_w + nx * frame_depth / 2
        my = mid[1] + eu[1] * t * glass_w + ny * frame_depth / 2
        _add_oriented_box(bm, mx, my, z_center,
                          eu, (nx, ny),
                          mullion_w / 2, frame_depth / 2, glass_h / 2)

    # Horizontal mullion at mid-height
    _add_oriented_box(bm, mid[0] + nx * frame_depth / 2,
                      mid[1] + ny * frame_depth / 2,
                      z_center,
                      eu, (nx, ny),
                      glass_w / 2, frame_depth / 2, mullion_w / 2)


def _generate_glass_wall_panel(bm, footprint_hull, detail, wall_height):
    """Create just the glass panel (separate object for transparent material).

    Called by the orchestrator alongside _generate_glass_wall_mesh.
    """
    n = len(footprint_hull)
    if n < 3:
        return

    sill_h = detail.get("sill_height", 0.0)
    glass_h = detail.get("height", wall_height - sill_h - 0.05)
    if glass_h <= 0:
        glass_h = wall_height * 0.85

    ei, _ = _find_longest_hull_edge(footprint_hull)
    eu, (nx, ny), mid, elen = _hull_edge_geometry(footprint_hull, ei)
    glass_w = elen * 0.88
    z_center = sill_h + glass_h / 2

    # Thin glass slab
    _add_oriented_box(bm, mid[0] + nx * 0.01,
                      mid[1] + ny * 0.01,
                      z_center,
                      eu, (nx, ny),
                      glass_w / 2, 0.005, glass_h / 2)


def _generate_exposed_beams_mesh(bm, footprint_hull, detail, wall_height):
    """Add exposed beam ends protruding from under roof overhang.

    Beam ends stick out perpendicular to each hull edge, spaced
    regularly along the wall length.  Classic MCM / Craftsman detail.
    """
    n = len(footprint_hull)
    if n < 3:
        return

    beam_w = detail.get("beam_width", 0.15)
    beam_h = detail.get("beam_height", 0.20)
    protrusion = detail.get("beam_protrusion", 0.4)
    spacing = detail.get("spacing", 0.8)

    normals = _hull_outward_normals(footprint_hull)

    for i in range(n):
        j = (i + 1) % n
        p0, p1 = footprint_hull[i], footprint_hull[j]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        elen = math.sqrt(dx * dx + dy * dy)
        if elen < spacing:
            continue
        eu = (dx / elen, dy / elen)
        nx, ny = normals[i]

        n_beams = max(1, int(elen / spacing))
        for k in range(n_beams):
            t = (k + 0.5) / n_beams
            bx = p0[0] + eu[0] * t * elen + nx * protrusion / 2
            by = p0[1] + eu[1] * t * elen + ny * protrusion / 2
            z_center = wall_height - beam_h / 2
            _add_oriented_box(bm, bx, by, z_center,
                              eu, (nx, ny),
                              beam_w / 2, protrusion / 2, beam_h / 2)


def _generate_columns_mesh(bm, footprint_hull, detail, wall_height):
    """Add structural columns along a building facade.

    Post-and-beam columns that support overhangs or mark entries.
    Supports square or round cross-sections.
    """
    n = len(footprint_hull)
    if n < 3:
        return

    col_d = detail.get("diameter", 0.25)
    shape = detail.get("shape", "square")
    count = detail.get("count", 4)
    # Standoff = distance from wall face.  Clamp to prevent columns
    # floating far from the building.  0 = flush with wall.
    standoff = min(detail.get("standoff", 0.3), 1.0)

    ei, _ = _find_longest_hull_edge(footprint_hull)
    eu, (nx, ny), mid, elen = _hull_edge_geometry(footprint_hull, ei)

    for k in range(count):
        t = (k + 0.5) / count - 0.5  # centred on edge
        cx = mid[0] + eu[0] * t * elen + nx * standoff
        cy = mid[1] + eu[1] * t * elen + ny * standoff

        if shape == "round":
            _add_cylinder(bm, cx, cy, wall_height / 2,
                          col_d / 2, wall_height)
        else:
            _add_oriented_box(bm, cx, cy, wall_height / 2,
                              eu, (nx, ny),
                              col_d / 2, col_d / 2, wall_height / 2)


def _generate_carport_mesh(bm, footprint_hull, detail, wall_height):
    """Add a carport — an open roof structure with columns but no walls.

    Extends from one side of the building.  The slab sits at wall_height.
    Four columns support it.  Iconic mid-century modern feature.
    """
    n = len(footprint_hull)
    if n < 3:
        return

    slab_thick = detail.get("slab_thickness", 0.15)
    col_size = detail.get("column_size", 0.20)

    # Use second-longest edge to avoid overlapping with glass_wall on longest
    edges = []
    for i in range(n):
        j = (i + 1) % n
        p0, p1 = footprint_hull[i], footprint_hull[j]
        d = math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
        edges.append((d, i))
    edges.sort(reverse=True)
    ei = edges[min(1, len(edges) - 1)][1]  # second-longest, or longest if only one
    eu, (nx, ny), mid, elen = _hull_edge_geometry(footprint_hull, ei)

    # Scale carport to look like a thin canopy, not a floating platform
    cp_width = min(detail.get("width", 6.0), elen * 0.35)   # max 35% of the wall
    cp_depth = min(detail.get("depth", 5.0), cp_width * 0.5)  # shallow depth
    cw = min(cp_width, elen * 0.35)
    slab_thick = 0.06  # thin roof slab — canopy, not platform
    slab_z = wall_height + slab_thick / 2

    # Thin canopy extending outward
    scx = mid[0] + nx * cp_depth / 2
    scy = mid[1] + ny * cp_depth / 2
    _add_oriented_box(bm, scx, scy, slab_z,
                      eu, (nx, ny),
                      cw / 2, cp_depth / 2, slab_thick / 2)

    # 4 support columns — two near wall, two at far edge
    for su in (-0.4, 0.4):
        for depth_frac in (0.1, 0.9):
            colx = mid[0] + eu[0] * cw * su + nx * cp_depth * depth_frac
            coly = mid[1] + eu[1] * cw * su + ny * cp_depth * depth_frac
            _add_oriented_box(bm, colx, coly, wall_height / 2,
                              eu, (nx, ny),
                              col_size / 2, col_size / 2, wall_height / 2)


def _generate_clerestory_mesh(bm, footprint_hull, detail, wall_height):
    """Add a row of clerestory windows high on the wall.

    Small horizontal window openings near the roofline, represented
    as glass panels.  Very common in mid-century modern homes.
    """
    n = len(footprint_hull)
    if n < 3:
        return

    win_w = detail.get("window_width", 1.2)
    win_h = detail.get("window_height", 0.6)
    count = detail.get("count", 5)
    spacing = detail.get("spacing", 0.3)
    z_frac = detail.get("z_position", 0.9)
    z_center = wall_height * z_frac

    ei, _ = _find_longest_hull_edge(footprint_hull)
    eu, (nx, ny), mid, elen = _hull_edge_geometry(footprint_hull, ei)

    total_w = count * win_w + (count - 1) * spacing
    if total_w > elen * 0.9:
        count = max(1, int((elen * 0.9 + spacing) / (win_w + spacing)))

    for k in range(count):
        offset = (k - (count - 1) / 2) * (win_w + spacing)
        wx = mid[0] + eu[0] * offset + nx * 0.02
        wy = mid[1] + eu[1] * offset + ny * 0.02
        _add_oriented_box(bm, wx, wy, z_center,
                          eu, (nx, ny),
                          win_w / 2, 0.03, win_h / 2)


def _generate_fascia_board_mesh(bm, footprint_hull, detail, wall_height, roof_cfg):
    """Add a fascia board along the roof edge.

    A thin horizontal board that runs along the building perimeter
    at the roof-wall junction, pushed outward to the overhang edge.
    Gives the clean horizontal line that defines modern architecture.
    """
    n = len(footprint_hull)
    if n < 3:
        return

    fascia_h = detail.get("height", 0.20)
    fascia_t = detail.get("thickness", 0.03)
    overhang = roof_cfg.get("overhang", 0.5)
    normals = _hull_outward_normals(footprint_hull)

    for i in range(n):
        j = (i + 1) % n
        p0, p1 = footprint_hull[i], footprint_hull[j]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        elen = math.sqrt(dx * dx + dy * dy)
        if elen < 0.5:
            continue
        eu = (dx / elen, dy / elen)
        nx, ny = normals[i]
        mid = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2]

        # Position at outer edge of overhang
        fcx = mid[0] + nx * overhang
        fcy = mid[1] + ny * overhang
        z_center = wall_height - fascia_h / 2

        _add_oriented_box(bm, fcx, fcy, z_center,
                          eu, (nx, ny),
                          elen / 2 + overhang,
                          fascia_t / 2, fascia_h / 2)


# Detail dispatcher — maps detail type strings to generator functions
_DETAIL_BUILDERS = {
    "cornice": lambda bm, fp, hull, det, wh, rc: _generate_cornice_mesh(bm, hull, det, wh),
    "window_sills": lambda bm, fp, hull, det, wh, rc: _generate_window_sills_mesh(bm, fp, det, wh),
    "pilasters": lambda bm, fp, hull, det, wh, rc: _generate_pilasters_mesh(bm, hull, det, wh),
    "canopy": lambda bm, fp, hull, det, wh, rc: _generate_canopy_mesh(bm, fp, det, wh),
    "accent_band": lambda bm, fp, hull, det, wh, rc: _generate_accent_band_mesh(bm, hull, det, wh),
    "chimney": lambda bm, fp, hull, det, wh, rc: _generate_chimney_mesh(bm, hull, det, wh, rc),
    "balcony": lambda bm, fp, hull, det, wh, rc: _generate_balcony_mesh(bm, fp, hull, det, wh),
    "pergola": lambda bm, fp, hull, det, wh, rc: _generate_pergola_mesh(bm, fp, hull, det, wh),
    "louver_screen": lambda bm, fp, hull, det, wh, rc: _generate_louver_screen_mesh(bm, fp, hull, det, wh),
    "planter_box": lambda bm, fp, hull, det, wh, rc: _generate_planter_box_mesh(bm, hull, det, wh),
    # "feature_wall" disabled — the thin vertical slab rising above the
    # roofline looks poor across all styles.  Geometry needs a redesign
    # before re-enabling (e.g. proper integration with the building volume).
    # "feature_wall": lambda bm, fp, hull, det, wh, rc: _generate_feature_wall_mesh(bm, hull, det, wh),
    # New high-impact detail types
    "glass_wall": lambda bm, fp, hull, det, wh, rc: _generate_glass_wall_mesh(bm, fp, hull, det, wh),
    "exposed_beams": lambda bm, fp, hull, det, wh, rc: _generate_exposed_beams_mesh(bm, hull, det, wh),
    "columns": lambda bm, fp, hull, det, wh, rc: _generate_columns_mesh(bm, hull, det, wh),
    "carport": lambda bm, fp, hull, det, wh, rc: _generate_carport_mesh(bm, hull, det, wh),
    "clerestory_windows": lambda bm, fp, hull, det, wh, rc: _generate_clerestory_mesh(bm, hull, det, wh),
    "fascia_board": lambda bm, fp, hull, det, wh, rc: _generate_fascia_board_mesh(bm, hull, det, wh, rc),
    "raised_volume": lambda bm, fp, hull, det, wh, rc: _generate_raised_volume_mesh(bm, fp, hull, det, wh, rc),
}


# -- Exterior orchestrator ---------------------------------------------------

def _get_or_create_exterior_collection(parent_collection):
    """Get or create the 'Exterior' sub-collection under the parent."""
    ext_name = "Exterior"
    ext_col = None
    for child in parent_collection.children:
        if child.name == ext_name:
            ext_col = child
            break
    if ext_col:
        for obj in list(ext_col.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
    else:
        ext_col = bpy.data.collections.new(ext_name)
        parent_collection.children.link(ext_col)
    return ext_col


def generate_exterior(floor_plan_data, exterior_config, collection, wall_height,
                      story_data=None):
    """Generate full exterior shell (walls, roof, foundation, frames, surrounds).

    Args:
        floor_plan_data: the floor plan JSON dict (ground floor / primary).
        exterior_config: AI-returned exterior configuration (merged with defaults).
        collection: parent FloorPlan3D collection.
        wall_height: total building height in metres (wall_height × num_stories).
        story_data: optional dict mapping story_index → floor plan dict.
                    When provided, the roof follows the TOP story's footprint
                    while the foundation follows the ground floor's.

    Returns:
        dict with stats: {"roof": str, "walls": bool, "foundation": bool, ...}
    """
    from . import materials

    config = _validate_exterior_config(exterior_config)

    # For multi-story, the roof follows the top story's footprint (which is
    # often smaller than the ground floor).  However the raw top-story outline
    # can be very jagged and may extend beyond the ground floor due to
    # rasterisation noise.  We clip it to the ground floor's bounding rect
    # and use the top story's bounding rect for a clean roof shape.
    roof_plan = floor_plan_data
    if story_data and len(story_data) > 1:
        top_story_idx = max(story_data.keys())
        roof_plan = story_data[top_story_idx]

    footprint = compute_building_footprint(floor_plan_data)  # ground floor
    if not footprint:
        return {"error": "Could not compute building footprint"}

    roof_footprint_data = compute_building_footprint(roof_plan)  # top story
    if not roof_footprint_data:
        roof_footprint_data = footprint
    elif story_data and len(story_data) > 1:
        # For multi-story: clamp the top story's hull points to the ground
        # floor bounds (to remove rasterisation artefacts like balconies
        # extending beyond the building), but keep the hull shape — do NOT
        # replace with a bounding rect, which loses the L-shape.
        ground_rect = footprint["rect_corners"]
        gx = [c[0] for c in ground_rect]
        gy = [c[1] for c in ground_rect]
        g_min_x, g_max_x = min(gx), max(gx)
        g_min_y, g_max_y = min(gy), max(gy)
        raw_hull = roof_footprint_data["hull"]
        clamped_hull = [
            [max(g_min_x, min(g_max_x, p[0])),
             max(g_min_y, min(g_max_y, p[1]))]
            for p in raw_hull
        ]
        # Remove duplicate consecutive points that clamping may create
        deduped = [clamped_hull[0]]
        for p in clamped_hull[1:]:
            if abs(p[0] - deduped[-1][0]) > 0.05 or \
               abs(p[1] - deduped[-1][1]) > 0.05:
                deduped.append(p)
        if len(deduped) >= 3:
            roof_footprint_data = dict(roof_footprint_data)
            roof_footprint_data["hull"] = deduped

    # Debug: write footprint info to file for inspection
    try:
        import json as _json
        _rf = roof_footprint_data
        _dbg = {
            "ground_hull_pts": len(footprint.get("hull", [])),
            "ground_hull": footprint.get("hull", []),
            "roof_hull_pts": len(_rf.get("hull", [])),
            "roof_hull": _rf.get("hull", []),
            "rect_corners": footprint.get("rect_corners", []),
            "rooms_count": len(floor_plan_data.get("rooms", [])),
            "roof_rooms_count": len(roof_plan.get("rooms", [])),
            "multi_story": story_data is not None and len(story_data or {}) > 1,
        }
        _dbg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "model", "data", "debug_footprint.json")
        with open(_dbg_path, 'w') as _f:
            _json.dump(_dbg, _f, indent=2)
    except Exception:
        pass

    ext_col = _get_or_create_exterior_collection(collection)
    stats = {}
    facade_cfg = config.get("facade", {})
    found_cfg = facade_cfg.get("foundation", {})
    found_height = found_cfg.get("height", 0.3)

    # --- Restyle existing walls with cladding material ---
    cladding_cfg = facade_cfg.get("cladding", {})
    cladding_mat = cladding_cfg.get("material", "white_stucco")
    # Optional RGB color override (e.g. [0.9, 0.7, 0.7] for pink)
    cladding_color = cladding_cfg.get("color")
    if cladding_color and isinstance(cladding_color, (list, tuple)) and len(cladding_color) >= 3:
        cladding_color = tuple(cladding_color[:3]) + (1.0,)
    else:
        cladding_color = None
    materials.set_active_cladding(cladding_mat)  # enable roof↔wall harmony

    # Collect all objects from collection AND its child sub-collections
    # (multi-story builds have Story_0, Story_1, etc. sub-collections)
    def _all_objects(col):
        for obj in col.objects:
            yield obj
        for child in col.children:
            if child.name != "Exterior":  # don't recurse into Exterior
                yield from _all_objects(child)

    restyled = 0
    for obj in _all_objects(collection):
        if obj.get("fp3d_type") == "wall":
            materials.assign_exterior_material(
                obj, cladding_mat, color_override=cladding_color)
            restyled += 1
    if restyled:
        stats["restyled_walls"] = restyled

    # --- Parse roof color override ---
    roof_color_raw = config.get("roof", {}).get("color")
    if roof_color_raw and isinstance(roof_color_raw, (list, tuple)) and len(roof_color_raw) >= 3:
        roof_color_override = tuple(roof_color_raw[:3]) + (1.0,)
    else:
        roof_color_override = None

    # --- Restyle ceiling objects to match roof color ---
    # The ceiling planes from the initial 3D model sit at wall_height and are
    # visible from above, appearing as the "roof" surface.  Without restyling
    # they keep their default grey material and look like an ugly grey roof.
    roof_mat_key = config.get("roof", {}).get("material", "metal")
    for obj in _all_objects(collection):
        if obj.get("fp3d_type") == "ceiling":
            materials.assign_roof_material(
                obj, roof_mat_key, cladding_override=cladding_mat,
                color_override=roof_color_override)

    # --- Roof ---
    roof_cfg = config.get("roof", {})
    roof_type = roof_cfg.get("type", "gable").lower()
    roof_builder = _ROOF_BUILDERS.get(roof_type, _generate_modern_flat_roof)

    # Use top-story footprint for roof, ground-floor for foundation.
    # The hull can be very detailed (many small jogs from room boundaries)
    # which looks terrible as a roof outline.  Simplify aggressively:
    # 1. Apply Douglas-Peucker with a large epsilon (1.5m) to remove
    #    small room-boundary jogs while preserving the overall shape.
    # 2. Remove near-collinear points that add visual noise.
    if roof_type in _HULL_COMPATIBLE_ROOFS:
        raw_hull = roof_footprint_data["hull"]
        # For multi-story, use a small epsilon to preserve wing/setback
        # geometry.  For single-story, use aggressive simplification to
        # remove room-boundary jogs.
        is_multi = story_data and len(story_data) > 1
        simplify_eps = 0.4 if is_multi else 1.5
        roof_footprint = _simplify_roof_outline(raw_hull, epsilon=simplify_eps)
    else:
        roof_footprint = roof_footprint_data["rect_corners"]

    # Debug: append roof info to debug file
    try:
        import json as _json
        _dbg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "model", "data", "debug_footprint.json")
        with open(_dbg_path, 'r') as _f:
            _dbg = _json.load(_f)
        _dbg["roof_type"] = roof_type
        _dbg["hull_compatible"] = roof_type in _HULL_COMPATIBLE_ROOFS
        _dbg["roof_footprint_pts"] = len(roof_footprint)
        _dbg["roof_footprint"] = [[round(p[0], 3), round(p[1], 3)] for p in roof_footprint]
        with open(_dbg_path, 'w') as _f:
            _json.dump(_dbg, _f, indent=2)
    except Exception:
        pass

    roof_mesh = bpy.data.meshes.new("Ext_Roof")
    roof_bm = bmesh.new()
    roof_builder(roof_bm, roof_footprint, roof_cfg, wall_height)
    roof_bm.to_mesh(roof_mesh)
    roof_bm.free()
    roof_mesh.update()
    roof_obj = bpy.data.objects.new("Ext_Roof", roof_mesh)
    roof_obj["fp3d_type"] = "exterior"
    roof_obj["fp3d_exterior_part"] = "roof"
    _link_to_collection(roof_obj, ext_col)
    materials.assign_roof_material(
        roof_obj, roof_cfg.get("material"), cladding_override=cladding_mat,
        color_override=roof_color_override)
    stats["roof"] = roof_type

    # --- Lower roof for exposed ground floor (multi-story with setback) ---
    # When the top story is smaller than the ground floor, the ground-floor
    # area not covered by the upper story needs its own roof at single-story
    # height.  Compute this by grid subtraction: rasterize both floors and
    # trace the outline of (ground − upper).
    if story_data and len(story_data) > 1:
        num_stories = len(story_data)
        per_story_h = wall_height / num_stories
        top_story_idx = max(story_data.keys())
        top_plan = story_data[top_story_idx]

        # Use the SIMPLIFIED roof footprint (the one actually rendered as the
        # main roof) for the subtraction — not the raw hull.  This ensures the
        # lower roof perfectly fills the gap left by the main roof, with no
        # overlap or uncovered strips.
        upper_subtract_hull = roof_footprint if (
            roof_type in _HULL_COMPATIBLE_ROOFS and roof_footprint
        ) else roof_footprint_data.get("hull")

        lower_outline = _compute_exposed_ground_outline(
            floor_plan_data, top_plan,
            upper_hull=upper_subtract_hull)

        # Debug: log lower roof computation details
        try:
            import json as _json
            _dbg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "model", "data", "debug_footprint.json")
            with open(_dbg_path, 'r') as _f:
                _dbg = _json.load(_f)
            _dbg["lower_roof_debug"] = {
                "upper_subtract_hull_pts": len(upper_subtract_hull) if upper_subtract_hull else 0,
                "upper_subtract_hull": [[round(p[0], 2), round(p[1], 2)]
                                         for p in upper_subtract_hull] if upper_subtract_hull else [],
                "lower_outline_raw_pts": len(lower_outline) if lower_outline else 0,
                "lower_outline_raw": [[round(p[0], 2), round(p[1], 2)]
                                       for p in lower_outline] if lower_outline else [],
                "per_story_h": per_story_h,
                "wall_height": wall_height,
            }
            with open(_dbg_path, 'w') as _f:
                _json.dump(_dbg, _f, indent=2)
        except Exception:
            pass

        if lower_outline and len(lower_outline) >= 3:
            lower_outline_simplified = _simplify_roof_outline(lower_outline,
                                                              epsilon=0.4)

            # Debug: log simplified outline
            try:
                with open(_dbg_path, 'r') as _f:
                    _dbg = _json.load(_f)
                _dbg["lower_roof_debug"]["simplified_pts"] = len(lower_outline_simplified)
                _dbg["lower_roof_debug"]["simplified"] = [
                    [round(p[0], 2), round(p[1], 2)]
                    for p in lower_outline_simplified]
                with open(_dbg_path, 'w') as _f:
                    _json.dump(_dbg, _f, indent=2)
            except Exception:
                pass

            if len(lower_outline_simplified) >= 3:
                lr_mesh = bpy.data.meshes.new("Ext_LowerRoof")
                lr_bm = bmesh.new()
                # Use the simple slab builder (no parapets) — the concave
                # lower-roof polygon would create distorted geometry with
                # _generate_modern_flat_roof's radial _expand_corners.
                _generate_lower_roof_slab(
                    lr_bm, lower_outline_simplified, roof_cfg, per_story_h)
                lr_bm.to_mesh(lr_mesh)
                lr_bm.free()
                lr_mesh.update()
                lr_obj = bpy.data.objects.new("Ext_LowerRoof", lr_mesh)
                lr_obj["fp3d_type"] = "exterior"
                lr_obj["fp3d_exterior_part"] = "roof"
                _link_to_collection(lr_obj, ext_col)
                materials.assign_roof_material(
                    lr_obj, roof_cfg.get("material"),
                    cladding_override=cladding_mat,
                    color_override=roof_color_override)
                stats["lower_roofs"] = 1

    # --- Gable / hip fill walls ---
    fill_mesh = bpy.data.meshes.new("Ext_GableFill")
    fill_bm = bmesh.new()
    _generate_gable_fill_walls(fill_bm, roof_footprint_data["rect_corners"], roof_cfg, wall_height)
    fill_bm.to_mesh(fill_mesh)
    fill_bm.free()
    fill_mesh.update()
    if fill_mesh.vertices:
        fill_obj = bpy.data.objects.new("Ext_GableFill", fill_mesh)
        fill_obj["fp3d_type"] = "exterior"
        fill_obj["fp3d_exterior_part"] = "gable_fill"
        _link_to_collection(fill_obj, ext_col)
        materials.assign_exterior_material(
            fill_obj, cladding_mat, color_override=cladding_color)
        stats["gable_fill"] = True

    # --- Foundation ---
    if found_height > 0:
        found_mesh = bpy.data.meshes.new("Ext_Foundation")
        found_bm = bmesh.new()
        _generate_foundation_mesh(found_bm, footprint["hull"], found_cfg)
        found_bm.to_mesh(found_mesh)
        found_bm.free()
        found_mesh.update()
        found_obj = bpy.data.objects.new("Ext_Foundation", found_mesh)
        found_obj["fp3d_type"] = "exterior"
        found_obj["fp3d_exterior_part"] = "foundation"
        _link_to_collection(found_obj, ext_col)
        materials.assign_foundation_material(found_obj, found_cfg.get("material"))
        stats["foundation"] = True

    # --- Apply AI-recommended window sizing (e.g. floor-to-ceiling for modern) ---
    win_style = config.get("window_style")
    if win_style and floor_plan_data.get("windows"):
        new_h = win_style.get("height")
        new_sill = win_style.get("sill_height")
        new_w = win_style.get("width")
        # If height is specified but width isn't, derive width from height
        # to maintain reasonable proportions (avoids tall narrow slits)
        if new_h is not None and new_w is None:
            new_w = max(1.2, new_h * 0.7)  # ~70% of height, min 1.2m
        for w in floor_plan_data["windows"]:
            if new_h is not None:
                w["height"] = new_h
            if new_sill is not None:
                w["sill_height"] = new_sill
            if new_w is not None:
                w["width"] = new_w

    # --- Window frames ---
    win_cfg = facade_cfg.get("window_frames", {})
    if floor_plan_data.get("windows"):
        wf_mesh = bpy.data.meshes.new("Ext_WindowFrames")
        wf_bm = bmesh.new()
        _generate_window_frame_meshes(wf_bm, floor_plan_data, win_cfg, wall_height)
        wf_bm.to_mesh(wf_mesh)
        wf_bm.free()
        wf_mesh.update()
        wf_obj = bpy.data.objects.new("Ext_WindowFrames", wf_mesh)
        wf_obj["fp3d_type"] = "exterior"
        wf_obj["fp3d_exterior_part"] = "window_frames"
        _link_to_collection(wf_obj, ext_col)
        materials.assign_exterior_material(wf_obj, win_cfg.get("material"))
        stats["window_frames"] = len(floor_plan_data["windows"])

    # --- Door surrounds ---
    door_cfg = facade_cfg.get("door_surround", {})
    if floor_plan_data.get("doors"):
        ds_mesh = bpy.data.meshes.new("Ext_DoorSurrounds")
        ds_bm = bmesh.new()
        _generate_door_surround_meshes(ds_bm, floor_plan_data, door_cfg,
                                      wall_height, footprint["hull"])
        ds_bm.to_mesh(ds_mesh)
        ds_bm.free()
        ds_mesh.update()
        ds_obj = bpy.data.objects.new("Ext_DoorSurrounds", ds_mesh)
        ds_obj["fp3d_type"] = "exterior"
        ds_obj["fp3d_exterior_part"] = "door_surrounds"
        _link_to_collection(ds_obj, ext_col)
        materials.assign_exterior_material(ds_obj, door_cfg.get("material"))
        stats["door_surrounds"] = len(floor_plan_data["doors"])

    # --- Door panels (visible door leaves on exterior face) ---
    if floor_plan_data.get("doors"):
        dp_mesh = bpy.data.meshes.new("Ext_DoorPanels")
        dp_bm = bmesh.new()
        _generate_door_panel_meshes(dp_bm, floor_plan_data,
                                    footprint["hull"], wall_height)
        dp_bm.to_mesh(dp_mesh)
        dp_bm.free()
        dp_mesh.update()
        if dp_mesh.vertices:
            dp_obj = bpy.data.objects.new("Ext_DoorPanels", dp_mesh)
            dp_obj["fp3d_type"] = "exterior"
            dp_obj["fp3d_exterior_part"] = "door_panels"
            _link_to_collection(dp_obj, ext_col)
            materials.assign_door_material(dp_obj)
            stats["door_panels"] = len(floor_plan_data["doors"])

    # --- Architectural details (canopy, cornice, pilasters, etc.) ---
    # For multi-story buildings, roof-level details (raised_volume, chimney,
    # fascia, exposed beams, cornice, clerestory) must use the top story's
    # footprint — not the ground floor's, which can be much wider.
    _ROOF_LEVEL_DETAILS = {
        "raised_volume", "chimney", "fascia_board", "exposed_beams",
        "cornice", "clerestory_windows",
    }
    roof_hull = roof_footprint_data["hull"]

    details = config.get("details", [])
    detail_count = 0
    for idx, detail in enumerate(details):
        dtype = detail.get("type", "").lower()

        # raised_volume only makes sense on flat/modern_flat roofs (MCM style).
        # On pitched roofs (hip/gable/shed) it creates an ugly floating slab.
        if dtype == "raised_volume" and roof_type not in ("flat", "modern_flat"):
            continue

        builder = _DETAIL_BUILDERS.get(dtype)
        if not builder:
            continue

        # Roof-level details follow top story footprint; ground-level
        # details (canopy, planter, glass wall, etc.) follow ground floor.
        is_multi = story_data and len(story_data) > 1
        detail_hull = roof_hull if (is_multi and dtype in _ROOF_LEVEL_DETAILS) \
            else footprint["hull"]

        det_mesh = bpy.data.meshes.new(f"Ext_Detail_{dtype}_{idx}")
        det_bm = bmesh.new()
        try:
            builder(det_bm, floor_plan_data, detail_hull,
                    detail, wall_height, roof_cfg)
        except Exception:
            det_bm.free()
            continue
        det_bm.to_mesh(det_mesh)
        det_bm.free()
        det_mesh.update()

        if det_mesh.vertices:
            det_obj = bpy.data.objects.new(f"Ext_{dtype.title()}_{idx}", det_mesh)
            det_obj["fp3d_type"] = "exterior"
            det_obj["fp3d_exterior_part"] = f"detail_{dtype}"
            _link_to_collection(det_obj, ext_col)
            det_mat = detail.get("material", cladding_mat)
            # Raised volume is a roof slab — always use the ROOF material
            # (not the detail's own material) so it matches the main roof.
            if dtype == "raised_volume":
                materials.assign_roof_material(
                    det_obj, roof_cfg.get("material", "metal"),
                    cladding_override=cladding_mat,
                    color_override=roof_color_override)
                print(f"[FP3D] raised_volume mat assigned: "
                      f"obj={det_obj.name}, slots={len(det_obj.data.materials)}")
            else:
                materials.assign_exterior_material(det_obj, det_mat)
            detail_count += 1

        # Glass wall special case: create a second object for the glass panel
        # so it can have a separate transparent material.
        if dtype == "glass_wall":
            glass_mesh = bpy.data.meshes.new(f"Ext_GlassPanel_{idx}")
            glass_bm = bmesh.new()
            try:
                _generate_glass_wall_panel(
                    glass_bm, footprint["hull"], detail, wall_height)
            except Exception:
                glass_bm.free()
                continue
            glass_bm.to_mesh(glass_mesh)
            glass_bm.free()
            glass_mesh.update()
            if glass_mesh.vertices:
                glass_obj = bpy.data.objects.new(
                    f"Ext_GlassPanel_{idx}", glass_mesh)
                glass_obj["fp3d_type"] = "exterior"
                glass_obj["fp3d_exterior_part"] = "glass_panel"
                _link_to_collection(glass_obj, ext_col)
                materials.assign_exterior_material(glass_obj, "glass")

    if detail_count:
        stats["details"] = detail_count

    return stats


def remove_exterior(collection):
    """Remove all exterior objects, materials, and restore default wall materials."""
    from . import materials

    if not collection:
        return 0

    # Restore wall materials to default (including Story_N sub-collections)
    def _iter_all(col):
        for obj in col.objects:
            yield obj
        for child in col.children:
            if child.name != "Exterior":
                yield from _iter_all(child)

    for obj in _iter_all(collection):
        if obj.get("fp3d_type") == "wall":
            materials.assign_wall_material(obj)

    # Remove the Exterior sub-collection
    removed = 0
    for child in list(collection.children):
        if child.name == "Exterior":
            for obj in list(child.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
                removed += 1
            bpy.data.collections.remove(child)
            break

    # Purge orphaned exterior materials so they're rebuilt fresh next time
    for mat in list(bpy.data.materials):
        if mat.name.startswith(("FP3D_Roof_", "FP3D_Ext_", "FP3D_Foundation_")):
            if mat.users == 0:
                bpy.data.materials.remove(mat)

    return removed
