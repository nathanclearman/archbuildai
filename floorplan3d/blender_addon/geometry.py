"""
Geometry generation from parsed floor plan JSON data.

Creates Blender meshes for walls, floors, ceilings, door/window openings,
and room labels from the structured JSON output of the floor plan parser.
"""

import bpy
import bmesh
import math
from mathutils import Vector, Matrix


def create_floorplan_collection(context):
    """Create or clear the FloorPlan3D collection."""
    collection = bpy.data.collections.get("FloorPlan3D")
    if collection:
        for obj in list(collection.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
    else:
        collection = bpy.data.collections.new("FloorPlan3D")
        context.scene.collection.children.link(collection)
    return collection


def _link_to_collection(obj, collection):
    """Link an object to the given collection only."""
    collection.objects.link(obj)


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

    Why this matters: `bm.faces.new(verts)` computes the face normal
    from vertex order — CCW → +Z (face looks up), CW → -Z (face looks
    down). For a floor polygon we want +Z so the floor is visible from
    above. synthesize.py's `_apply_augmentation` flips the winding of
    every polygon whenever `flip_x=True`, which hits ~50% of samples,
    so we can't rely on a consistent winding from upstream JSON.

    Without this normalization, roughly half of all loaded plans
    produced invisible floors (and ceilings flipped the WRONG way after
    `face.normal_flip()`, which assumes a CCW input). Bug was silent
    in all 16 prior geometry tests because the sample fixtures
    happened to be CCW.
    """
    if _signed_polygon_area(polygon) < 0:
        return list(reversed(polygon))
    return polygon


def _clamp_along_wall(dist_along, wall_length, half_width):
    """Clamp `dist_along` so a door/window of total width `2*half_width`
    fits entirely inside the [0, wall_length] wall span.

    Why: the VLM sometimes emits a door `position=[x, y]` that sits
    slightly off the wall line. Projecting onto the wall axis (see
    `_project_position_to_wall`) gives a `dist_along` that may land
    past either end of the wall segment. The boolean cutter then sits
    outside the wall — the modifier succeeds but cuts nothing visible,
    producing a "door didn't open" failure with no error message.

    Clamping keeps the cutter inside the wall. If `wall_length <
    2*half_width` (door wider than wall), clamp collapses to a single
    centered point; the cutter will overshoot both ends but at least
    hit the wall. Caller may choose to skip instead via the returned
    value vs wall_length comparison.
    """
    lo = half_width
    hi = max(half_width, wall_length - half_width)
    return max(lo, min(hi, dist_along))


def _project_position_to_wall(wall_start, wall_direction, position):
    """Return distance along wall from `wall_start` for a door/window.

    Accepts either [x, y] absolute coordinates (canonical schema — see
    `schema.serialize`) or a scalar distance-along-wall (legacy format
    some older sample fixtures use). The [x, y] case dot-products onto
    the wall direction; the scalar case passes through.
    """
    if isinstance(position, (list, tuple)) and len(position) == 2:
        dp = Vector(position) - Vector(wall_start)
        return dp.dot(wall_direction)
    return float(position)


def _label_size_for_polygon(polygon, min_size=0.2, max_size=1.0, ratio=0.12):
    """Return a font size for a room label, scaled to the room's smaller
    bounding-box dimension.

    The previous hardcoded 0.3m was illegible in a 20m great room (text
    disappears at that camera distance) and overflowed the width of a
    1.2m hallway (text spilled out into adjacent rooms). Scaling by the
    polygon's minor axis keeps labels proportional to what they label.

    Clamped so:
      - a degenerate / zero-area polygon can't produce a 0m label
        (font_curve.size=0 is a Blender error)
      - an outsize polygon can't produce a 5m label that dominates the
        viewport
    """
    if not polygon:
        return min_size
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    minor = min(max(xs) - min(xs), max(ys) - min(ys))
    size = minor * ratio
    return max(min_size, min(max_size, size))


def _apply_cutter_boolean(wall_obj, cutter_obj, modifier_name):
    """Attach `cutter_obj` to `wall_obj` as a DIFFERENCE boolean and apply.

    Returns True if the modifier applied cleanly, False if apply failed.

    On failure: removes the modifier and deletes the cutter so the wall
    is left in its pre-call state and the scene has no orphaned cutter
    boxes. Leaving a half-applied modifier stack on the wall corrupts
    later cutter applies on the same wall (GEO-6 from the code review).

    Uses `bpy.context.temp_override` to set the active object rather than
    mutating `view_layer.objects.active`. The mutation idiom:
      - leaks scene state (the user's selection and active object stay
        changed after the operator returns)
      - fails silently in some modal-operator contexts where the view
        layer's active setter is a no-op
    `temp_override` restores state automatically and is the documented
    Blender 3.2+ / 4.x way to run an op on a specific object.
    """
    mod = wall_obj.modifiers.new(name=modifier_name, type='BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.object = cutter_obj
    mod.solver = 'FLOAT'

    try:
        with bpy.context.temp_override(object=wall_obj, active_object=wall_obj):
            bpy.ops.object.modifier_apply(modifier=mod.name)
    except (RuntimeError, AttributeError) as e:
        # RuntimeError: modifier_apply refused (non-manifold input, bad
        #   context, solver failure). AttributeError: temp_override
        #   missing on very old Blender — we target 4.x but be defensive.
        # Either way, roll back so subsequent cutters on this wall
        # don't inherit a broken modifier stack.
        print(f"FloorPlan3D: modifier_apply failed for {modifier_name}: {e}")
        try:
            wall_obj.modifiers.remove(mod)
        except Exception:
            pass
        try:
            bpy.data.objects.remove(cutter_obj, do_unlink=True)
        except Exception:
            pass
        return False

    bpy.data.objects.remove(cutter_obj, do_unlink=True)
    return True


def generate_walls(floor_plan_data, collection, wall_height):
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
            Vector((start.x - perp.x * half_t, start.y - perp.y * half_t, 0)),
            Vector((start.x + perp.x * half_t, start.y + perp.y * half_t, 0)),
            Vector((end.x + perp.x * half_t, end.y + perp.y * half_t, 0)),
            Vector((end.x - perp.x * half_t, end.y - perp.y * half_t, 0)),
        ]

        mesh = bpy.data.meshes.new(f"Wall_{i}")
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

        obj = bpy.data.objects.new(f"Wall_{i}", mesh)
        obj["fp3d_type"] = "wall"
        obj["fp3d_wall_index"] = i
        obj["fp3d_original_height"] = wall_height
        _link_to_collection(obj, collection)

        materials.assign_wall_material(obj)
        count += 1

    return count


def generate_door_openings(floor_plan_data, collection, wall_height):
    """Cut door openings into walls using boolean modifiers.

    Doors are full-height openings (default 2.1m) cut from the bottom of the wall.
    Returns the number of door openings created.
    """
    count = 0
    doors = floor_plan_data.get("doors", [])
    walls = floor_plan_data.get("walls", [])

    for i, door_data in enumerate(doors):
        wall_idx = door_data.get("wall_index", 0)
        # Skip doors with `wall_index=-1` (schema's "no wall attached"
        # sentinel). Python's `-1 >= len(walls)` is False, so the old
        # single-sided check let -1 slip through — walls[-1] then silently
        # picked the LAST wall in the list and cut a door through it at
        # the wrong position.
        if wall_idx < 0 or wall_idx >= len(walls):
            continue

        wall_data = walls[wall_idx]
        start = Vector(wall_data["start"])
        end = Vector(wall_data["end"])
        thickness = wall_data.get("thickness", 0.15)

        direction, wall_length = _wall_direction(start, end)
        if wall_length < 1e-6:
            continue

        half_t = thickness / 2.0

        door_width = door_data.get("width", 0.9)
        door_height = door_data.get("height", 2.1)
        half_w = door_width / 2.0

        # Door position along wall. See _project_position_to_wall + _clamp_along_wall
        # for the off-wall-projection and overshoot story (GEO-3).
        door_pos = door_data.get("position", [0, 0])
        dist_along = _project_position_to_wall(start, direction, door_pos)
        center_along = _clamp_along_wall(dist_along, wall_length, half_w)

        # Create cutter box
        cutter_center = (
            start.x + direction.x * center_along,
            start.y + direction.y * center_along,
            door_height / 2.0,
        )

        mesh = bpy.data.meshes.new(f"DoorCutter_{i}")
        bm = bmesh.new()

        # Cutter box slightly larger than wall thickness to ensure clean cut
        margin = thickness * 1.5
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

        cutter_obj = bpy.data.objects.new(f"DoorCutter_{i}", mesh)
        cutter_obj.location = Vector(cutter_center)

        # Rotate cutter to align with wall direction
        angle = math.atan2(direction.y, direction.x)
        cutter_obj.rotation_euler.z = angle

        _link_to_collection(cutter_obj, collection)

        wall_obj = bpy.data.objects.get(f"Wall_{wall_idx}")
        if wall_obj and _apply_cutter_boolean(wall_obj, cutter_obj, f"Door_{i}"):
            count += 1

    return count


def generate_window_openings(floor_plan_data, collection, wall_height):
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
        # Same negative-index guard as generate_door_openings; -1 is the
        # schema sentinel for "not attached to a wall".
        if wall_idx < 0 or wall_idx >= len(walls):
            continue

        wall_data = walls[wall_idx]
        start = Vector(wall_data["start"])
        end = Vector(wall_data["end"])
        thickness = wall_data.get("thickness", 0.15)

        direction, wall_length = _wall_direction(start, end)
        if wall_length < 1e-6:
            continue

        win_width = win_data.get("width", 1.2)
        win_height = win_data.get("height", 1.2)
        sill_height = win_data.get("sill_height", 0.9)
        half_w = win_width / 2.0

        # Same off-wall-projection + overshoot guard as doors.
        win_pos = win_data.get("position", [0, 0])
        dist_along = _project_position_to_wall(start, direction, win_pos)
        center_along = _clamp_along_wall(dist_along, wall_length, half_w)

        cutter_center = (
            start.x + direction.x * center_along,
            start.y + direction.y * center_along,
            sill_height + win_height / 2.0,
        )

        mesh = bpy.data.meshes.new(f"WindowCutter_{i}")
        bm = bmesh.new()

        margin = thickness * 1.5
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

        cutter_obj = bpy.data.objects.new(f"WindowCutter_{i}", mesh)
        cutter_obj.location = Vector(cutter_center)

        angle = math.atan2(direction.y, direction.x)
        cutter_obj.rotation_euler.z = angle

        _link_to_collection(cutter_obj, collection)

        wall_obj = bpy.data.objects.get(f"Wall_{wall_idx}")
        if wall_obj and _apply_cutter_boolean(wall_obj, cutter_obj, f"Window_{i}"):
            count += 1

    return count


def generate_floors(floor_plan_data, collection):
    """Generate floor planes from room polygons. Returns the number of floors generated."""
    from . import materials

    count = 0
    rooms = floor_plan_data.get("rooms", [])
    for i, room_data in enumerate(rooms):
        polygon = room_data.get("polygon", [])
        if len(polygon) < 3:
            continue
        # Normalize to CCW so bm.faces.new gives the floor a +Z normal
        # regardless of the winding the upstream JSON landed in. See
        # _ensure_ccw docstring for the augmentation-flip story.
        polygon = _ensure_ccw(polygon)

        label = room_data.get("label", f"Room_{i}")

        mesh = bpy.data.meshes.new(f"Floor_{label}_{i}")
        bm = bmesh.new()

        verts = [bm.verts.new(Vector((p[0], p[1], 0))) for p in polygon]
        try:
            bm.faces.new(verts)
        except ValueError:
            # Face creation can fail if verts are collinear
            bm.free()
            continue

        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        obj = bpy.data.objects.new(f"Floor_{label}_{i}", mesh)
        obj["fp3d_type"] = "floor"
        obj["fp3d_room_label"] = label
        _link_to_collection(obj, collection)

        materials.assign_floor_material(obj)
        count += 1

    return count


def generate_ceilings(floor_plan_data, collection, wall_height):
    """Generate ceiling planes from room polygons. Returns the number of ceilings generated."""
    from . import materials

    count = 0
    rooms = floor_plan_data.get("rooms", [])
    for i, room_data in enumerate(rooms):
        polygon = room_data.get("polygon", [])
        if len(polygon) < 3:
            continue
        # Same CCW normalization as the floor. The `face.normal_flip()`
        # below flips +Z to -Z so the ceiling faces down (visible from
        # inside the room). Before normalization, a CW polygon produced
        # a -Z normal that the flip bumped BACK to +Z — ceiling then
        # looked up instead of down, wrong from any viewing angle inside
        # the house.
        polygon = _ensure_ccw(polygon)

        label = room_data.get("label", f"Room_{i}")

        mesh = bpy.data.meshes.new(f"Ceiling_{label}_{i}")
        bm = bmesh.new()

        verts = [bm.verts.new(Vector((p[0], p[1], wall_height))) for p in polygon]
        try:
            face = bm.faces.new(verts)
            # Flip normal to face downward
            face.normal_flip()
        except ValueError:
            bm.free()
            continue

        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        obj = bpy.data.objects.new(f"Ceiling_{label}_{i}", mesh)
        obj["fp3d_type"] = "ceiling"
        obj["fp3d_room_label"] = label
        _link_to_collection(obj, collection)

        materials.assign_ceiling_material(obj)
        count += 1

    return count


def generate_room_labels(floor_plan_data, collection):
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

        # Create text object. Font size scales with room bbox so labels
        # are legible in both 1m hallways and 20m great rooms — see
        # _label_size_for_polygon for the sizing heuristic (GEO-8).
        font_curve = bpy.data.curves.new(name=f"Label_{label}_{i}", type='FONT')
        font_curve.body = label.replace("_", " ").title()
        font_curve.size = _label_size_for_polygon(polygon)
        font_curve.align_x = 'CENTER'
        font_curve.align_y = 'CENTER'

        obj = bpy.data.objects.new(f"Label_{label}_{i}", font_curve)
        obj.location = Vector((cx, cy, 0.01))  # Slightly above floor
        obj.rotation_euler.x = 0  # Flat on the floor
        obj["fp3d_type"] = "label"
        obj["fp3d_room_label"] = label
        _link_to_collection(obj, collection)

        area = room_data.get("area")
        if area is not None:
            obj["fp3d_room_area"] = area
        count += 1

    return count
