"""
Interactive 2D correction layer for floor plan parsing results.

Provides a CorrectionState that creates Blender Empty objects for wall
endpoints, doors, and windows. Users manipulate these with standard Blender
tools (G to grab, X to delete), then confirm to generate corrected 3D geometry.
"""

import math

import bpy
import bmesh
from mathutils import Vector

# ── Module-level state ────────────────────────────────────────────────

_state = None


def get_state():
    """Return the current CorrectionState, or None."""
    return _state


def init_state(floor_plan_data, image_path, scale_factor):
    """Create and store a new CorrectionState."""
    global _state
    _state = CorrectionState(floor_plan_data, image_path, scale_factor)
    return _state


def clear_state():
    """Discard the current CorrectionState."""
    global _state
    _state = None


# ── Merge tolerance for shared wall corners ───────────────────────────

MERGE_TOL = 0.1  # 10 cm


def _snap_key(pos, tol=MERGE_TOL):
    """Round a 2D position to the merge grid."""
    return (round(pos[0] / tol) * tol, round(pos[1] / tol) * tol)


# ── Element data classes ──────────────────────────────────────────────

class CorrectionWall:
    """A wall segment with two draggable endpoints."""

    __slots__ = (
        "uid", "start", "end", "thickness",
        "start_empty", "end_empty", "edge_obj",
    )

    def __init__(self, uid, start, end, thickness):
        self.uid = uid
        self.start = list(start)
        self.end = list(end)
        self.thickness = thickness
        self.start_empty = None
        self.end_empty = None
        self.edge_obj = None


class CorrectionDoor:
    """A door marker snapped to the nearest wall."""

    __slots__ = ("uid", "position", "width", "door_type", "wall_index", "empty")

    def __init__(self, uid, position, width, door_type, wall_index):
        self.uid = uid
        self.position = list(position)
        self.width = width
        self.door_type = door_type
        self.wall_index = wall_index
        self.empty = None


class CorrectionWindow:
    """A window marker snapped to the nearest wall."""

    __slots__ = ("uid", "position", "width", "wall_index", "full_wall", "empty")

    def __init__(self, uid, position, width, wall_index, full_wall=False):
        self.uid = uid
        self.position = list(position)
        self.width = width
        self.wall_index = wall_index
        self.full_wall = full_wall
        self.empty = None


# ── Main state container ──────────────────────────────────────────────

class CorrectionState:
    """Holds the mutable floor plan data during interactive correction."""

    def __init__(self, floor_plan_data, image_path, scale_factor):
        self.original_data = floor_plan_data
        self.image_path = image_path
        self.scale_factor = scale_factor  # pixels_per_meter

        self.walls = []
        self.doors = []
        self.windows = []

        self.collection = None
        self.bg_object = None
        self.draw_handler = None
        self.active = False

        self._next_uid = 0
        # Maps snap_key -> Empty for shared wall corners
        self._corner_empties = {}

    # ── UID helper ────────────────────────────────────────────────────

    def _uid(self, prefix="e"):
        self._next_uid += 1
        return f"{prefix}_{self._next_uid}"

    # ── JSON import ───────────────────────────────────────────────────

    def populate_from_json(self, data):
        """Parse floor plan JSON into correction elements."""
        for i, w in enumerate(data.get("walls", [])):
            self.walls.append(CorrectionWall(
                uid=self._uid("wall"),
                start=w["start"],
                end=w["end"],
                thickness=w.get("thickness", 0.15),
            ))

        for i, d in enumerate(data.get("doors", [])):
            self.doors.append(CorrectionDoor(
                uid=self._uid("door"),
                position=d["position"],
                width=d.get("width", 0.9),
                door_type=d.get("type", "hinged"),
                wall_index=d.get("wall_index", 0),
            ))

        for i, w in enumerate(data.get("windows", [])):
            self.windows.append(CorrectionWindow(
                uid=self._uid("win"),
                position=w["position"],
                width=w.get("width", 1.2),
                wall_index=w.get("wall_index", 0),
                full_wall=w.get("full_wall", False),
            ))

    # ── JSON export ───────────────────────────────────────────────────

    def to_json(self):
        """Export corrected state back to floor plan JSON schema."""
        self.read_back_positions()

        result = {
            "scale": self.original_data.get("scale", {"pixels_per_meter": 50}),
            "walls": [],
            "doors": [],
            "windows": [],
            "rooms": self.original_data.get("rooms", []),
        }

        for wall in self.walls:
            result["walls"].append({
                "start": [round(wall.start[0], 3), round(wall.start[1], 3)],
                "end": [round(wall.end[0], 3), round(wall.end[1], 3)],
                "thickness": wall.thickness,
            })

        for door in self.doors:
            result["doors"].append({
                "position": [round(door.position[0], 3), round(door.position[1], 3)],
                "width": door.width,
                "type": door.door_type,
                "wall_index": door.wall_index,
            })

        for window in self.windows:
            win_entry = {
                "position": [round(window.position[0], 3), round(window.position[1], 3)],
                "width": window.width,
                "wall_index": window.wall_index,
            }
            if window.full_wall:
                win_entry["full_wall"] = True
            result["windows"].append(win_entry)

        return result

    # ── Blender object creation ───────────────────────────────────────

    def create_blender_objects(self, context):
        """Create all correction-mode Blender objects."""
        # Collection
        col = bpy.data.collections.get("FP3D_Correction")
        if col:
            _remove_collection_recursive(col)
        col = bpy.data.collections.new("FP3D_Correction")
        context.scene.collection.children.link(col)
        self.collection = col

        # Background image plane
        self.bg_object = _create_background_plane(
            self.image_path, self.scale_factor, col,
        )

        # Wall empties (with corner merging) and edge meshes
        self._corner_empties = {}
        for wall in self.walls:
            wall.start_empty = self._get_or_create_corner_empty(
                wall.start, col,
            )
            wall.end_empty = self._get_or_create_corner_empty(
                wall.end, col,
            )
            wall.edge_obj = _create_edge_mesh(wall, col)

        # Door empties
        for door in self.doors:
            door.empty = _create_marker_empty(
                f"Door_{door.uid}", door.position, "SINGLE_ARROW",
                door.width / 2, "door", door.uid, col,
            )
            door.empty["fp3d_door_width"] = door.width
            door.empty["fp3d_door_type"] = door.door_type

        # Window empties
        for window in self.windows:
            window.empty = _create_marker_empty(
                f"Win_{window.uid}", window.position, "CUBE",
                window.width / 2, "window", window.uid, col,
            )
            window.empty["fp3d_win_width"] = window.width

        self.active = True

    def _get_or_create_corner_empty(self, pos, collection):
        """Return a shared corner Empty, creating one if needed."""
        key = _snap_key(pos)
        if key in self._corner_empties:
            return self._corner_empties[key]

        empty = bpy.data.objects.new(f"WP_{len(self._corner_empties)}", None)
        empty.empty_display_type = 'SPHERE'
        empty.empty_display_size = 0.15
        empty.location = (pos[0], pos[1], 0.0)
        empty["fp3d_correction_type"] = "wall_point"
        collection.objects.link(empty)
        self._corner_empties[key] = empty
        return empty

    # ── Read back positions from Blender objects ──────────────────────

    def read_back_positions(self):
        """Update internal data from current Empty locations."""
        for wall in self.walls:
            if wall.start_empty:
                wall.start = [wall.start_empty.location.x,
                              wall.start_empty.location.y]
            if wall.end_empty:
                wall.end = [wall.end_empty.location.x,
                            wall.end_empty.location.y]

        for door in self.doors:
            if door.empty:
                door.position = [door.empty.location.x,
                                 door.empty.location.y]

        for window in self.windows:
            if window.empty:
                window.position = [window.empty.location.x,
                                   window.empty.location.y]

    # ── Sync visual helpers ───────────────────────────────────────────

    def sync_edge_meshes(self):
        """Update wall edge meshes to match current Empty positions."""
        for wall in self.walls:
            if not wall.edge_obj or not wall.start_empty or not wall.end_empty:
                continue
            mesh = wall.edge_obj.data
            if len(mesh.vertices) >= 2:
                mesh.vertices[0].co = wall.start_empty.location
                mesh.vertices[1].co = wall.end_empty.location
                mesh.update()

    def recompute_wall_assignments(self):
        """Reassign each door/window to the nearest wall."""
        for door in self.doors:
            if not door.empty:
                continue
            pos = (door.empty.location.x, door.empty.location.y)
            best_idx, _ = self._find_nearest_wall(pos)
            if best_idx is not None:
                door.wall_index = best_idx

        for window in self.windows:
            if not window.empty:
                continue
            pos = (window.empty.location.x, window.empty.location.y)
            best_idx, _ = self._find_nearest_wall(pos)
            if best_idx is not None:
                window.wall_index = best_idx

    def _find_nearest_wall(self, pos):
        """Return (wall_list_index, distance) for the nearest wall to pos."""
        best_idx = None
        best_dist = float("inf")
        for i, wall in enumerate(self.walls):
            ws = _empty_xy(wall.start_empty) if wall.start_empty else wall.start
            we = _empty_xy(wall.end_empty) if wall.end_empty else wall.end
            d = _point_to_segment_dist(pos, ws, we)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx, best_dist

    def snap_to_nearest_wall(self, pos, max_dist=1.0):
        """Snap a position to the nearest wall. Returns (wall_index, snapped_pos) or (None, None)."""
        best_idx, best_dist = self._find_nearest_wall(pos)
        if best_idx is None or best_dist > max_dist:
            return None, None

        wall = self.walls[best_idx]
        ws = _empty_xy(wall.start_empty) if wall.start_empty else wall.start
        we = _empty_xy(wall.end_empty) if wall.end_empty else wall.end
        snapped = _project_onto_segment(pos, ws, we)
        return best_idx, snapped

    # ── Add / remove elements ─────────────────────────────────────────

    def add_wall(self, start, end, thickness=0.15):
        """Add a new wall and create its Blender objects."""
        wall = CorrectionWall(self._uid("wall"), start, end, thickness)
        self.walls.append(wall)
        if self.collection:
            wall.start_empty = self._get_or_create_corner_empty(start, self.collection)
            wall.end_empty = self._get_or_create_corner_empty(end, self.collection)
            wall.edge_obj = _create_edge_mesh(wall, self.collection)

    def add_door(self, position, width=0.9, door_type="hinged", wall_index=0):
        """Add a new door and create its Empty."""
        door = CorrectionDoor(self._uid("door"), position, width, door_type, wall_index)
        self.doors.append(door)
        if self.collection:
            door.empty = _create_marker_empty(
                f"Door_{door.uid}", position, "SINGLE_ARROW",
                width / 2, "door", door.uid, self.collection,
            )
            door.empty["fp3d_door_width"] = width
            door.empty["fp3d_door_type"] = door_type

    def add_window(self, position, width=1.2, wall_index=0, full_wall=False):
        """Add a new window and create its Empty."""
        win = CorrectionWindow(self._uid("win"), position, width, wall_index,
                               full_wall=full_wall)
        self.windows.append(win)
        if self.collection:
            display_size = max(width / 2, 0.3)
            win.empty = _create_marker_empty(
                f"Win_{win.uid}", position, "CUBE",
                display_size, "window", win.uid, self.collection,
            )
            win.empty["fp3d_win_width"] = width
            if full_wall:
                win.empty["fp3d_full_wall"] = True

    def remove_element_by_object(self, obj):
        """Remove the correction element associated with a Blender object."""
        ctype = obj.get("fp3d_correction_type", "")
        uid = obj.get("fp3d_correction_uid", "")

        if ctype == "door":
            self.doors = [d for d in self.doors if d.uid != uid]
            bpy.data.objects.remove(obj, do_unlink=True)
            return True

        if ctype == "window":
            self.windows = [w for w in self.windows if w.uid != uid]
            bpy.data.objects.remove(obj, do_unlink=True)
            return True

        if ctype == "wall_point":
            # Remove all walls that use this corner empty
            to_remove = [w for w in self.walls
                         if w.start_empty == obj or w.end_empty == obj]
            for wall in to_remove:
                if wall.edge_obj:
                    bpy.data.objects.remove(wall.edge_obj, do_unlink=True)
                self.walls.remove(wall)
            bpy.data.objects.remove(obj, do_unlink=True)
            # Clean up from _corner_empties dict
            self._corner_empties = {
                k: v for k, v in self._corner_empties.items() if v != obj
            }
            return True

        return False

    # ── Cleanup ───────────────────────────────────────────────────────

    def cleanup(self, context):
        """Remove all correction objects, handlers, and collection."""
        if self.draw_handler:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(
                    self.draw_handler, 'WINDOW',
                )
            except Exception:
                pass
            self.draw_handler = None

        if self.collection:
            _remove_collection_recursive(self.collection)
            self.collection = None

        self.bg_object = None
        self._corner_empties = {}
        self.active = False


# ── Helper: create background image plane ─────────────────────────────

def _create_background_plane(image_path, pixels_per_meter, collection):
    """Create a textured plane showing the floor plan image at correct scale."""
    try:
        img = bpy.data.images.load(image_path, check_existing=True)
    except Exception:
        return None

    w_px, h_px = img.size
    if w_px == 0 or h_px == 0:
        return None

    w_m = w_px / pixels_per_meter
    h_m = h_px / pixels_per_meter

    mesh = bpy.data.meshes.new("FP3D_CorrectionBG")
    bm = bmesh.new()
    verts = [
        bm.verts.new((0, 0, -0.01)),
        bm.verts.new((w_m, 0, -0.01)),
        bm.verts.new((w_m, h_m, -0.01)),
        bm.verts.new((0, h_m, -0.01)),
    ]
    face = bm.faces.new(verts)

    uv_layer = bm.loops.layers.uv.new()
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for loop, uv in zip(face.loops, uvs):
        loop[uv_layer].uv = uv

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("FP3D_CorrectionBG", mesh)
    obj["fp3d_correction_type"] = "background"
    obj.hide_select = True
    collection.objects.link(obj)

    # Emission material so the image is visible in Solid viewport mode
    mat = bpy.data.materials.new("FP3D_CorrectionBG_Mat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in list(nodes):
        nodes.remove(node)

    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    tex = nodes.new('ShaderNodeTexImage')
    tex.image = img
    emission.inputs['Strength'].default_value = 0.5
    links.new(tex.outputs['Color'], emission.inputs['Color'])
    links.new(emission.outputs['Emission'], output.inputs['Surface'])

    obj.data.materials.append(mat)
    return obj


# ── Helper: create edge mesh for wall visualization ───────────────────

def _create_edge_mesh(wall, collection):
    """Create a thin edge-only mesh between wall start/end."""
    mesh = bpy.data.meshes.new(f"WallEdge_{wall.uid}")
    bm = bmesh.new()
    v1 = bm.verts.new((wall.start[0], wall.start[1], 0.0))
    v2 = bm.verts.new((wall.end[0], wall.end[1], 0.0))
    bm.edges.new((v1, v2))
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(f"WallEdge_{wall.uid}", mesh)
    obj.display_type = 'WIRE'
    obj["fp3d_correction_type"] = "wall_edge"
    obj.hide_select = True
    collection.objects.link(obj)
    return obj


# ── Helper: create marker empty for doors/windows ────────────────────

def _create_marker_empty(name, position, display_type, size, ctype, uid, collection):
    """Create an Empty to represent a door or window."""
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = display_type
    empty.empty_display_size = size
    empty.location = (position[0], position[1], 0.0)
    empty["fp3d_correction_type"] = ctype
    empty["fp3d_correction_uid"] = uid
    collection.objects.link(empty)
    return empty


# ── Helper: remove a collection and all its objects ───────────────────

def _remove_collection_recursive(col):
    """Remove a collection and every object/child inside it."""
    for child in list(col.children):
        _remove_collection_recursive(child)
    for obj in list(col.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(col)


# ── Geometry helpers ──────────────────────────────────────────────────

def _empty_xy(empty):
    """Return (x, y) tuple from an Empty's location."""
    return (empty.location.x, empty.location.y)


def _point_to_segment_dist(p, a, b):
    """Minimum distance from point p to line segment a-b (2D)."""
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    px, py = p[0], p[1]

    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return math.hypot(px - ax, py - ay)

    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _project_onto_segment(p, a, b):
    """Project point p onto segment a-b, clamped to endpoints. Returns [x, y]."""
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    px, py = p[0], p[1]

    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return [ax, ay]

    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return [ax + t * dx, ay + t * dy]
