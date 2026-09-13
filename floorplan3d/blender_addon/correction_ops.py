"""
Operators for the 2D interactive correction layer.

Provides EnterCorrection (modal), ConfirmCorrection, CancelCorrection,
AddWall, AddDoor, AddWindow, and DeleteElement operators.
"""

import json
import os

import bpy
from bpy_extras import view3d_utils
from mathutils import Vector

from .correction import get_state, init_state, clear_state


# ── Viewport helpers ──────────────────────────────────────────────────

def _setup_topdown_view(context):
    """Switch the active 3D viewport to top-down orthographic."""
    for area in context.screen.areas:
        if area.type != 'VIEW_3D':
            continue
        for space in area.spaces:
            if space.type != 'VIEW_3D':
                continue
            space.region_3d.view_perspective = 'ORTHO'
            space.region_3d.view_rotation = (1, 0, 0, 0)  # Top-down (Numpad 7)
            # Frame the correction objects
            override = context.copy()
            override['area'] = area
            override['region'] = area.regions[-1]
            with context.temp_override(**override):
                bpy.ops.view3d.view_all(center=True)
            break
        break


def _mouse_to_world_xy(context, event):
    """Convert mouse position to XY world coordinates on the Z=0 plane."""
    region = context.region
    rv3d = context.space_data.region_3d

    coord = (event.mouse_region_x, event.mouse_region_y)
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)

    # Intersect with Z=0 plane
    if abs(direction.z) < 1e-8:
        return [origin.x, origin.y]

    t = -origin.z / direction.z
    hit = origin + direction * t
    return [hit.x, hit.y]


# ── Enter Correction Mode ────────────────────────────────────────────

class FP3D_OT_EnterCorrection(bpy.types.Operator):
    bl_idname = "fp3d.enter_correction"
    bl_label = "Enter Correction Mode"
    bl_description = "Review and adjust AI-detected elements before 3D generation"
    bl_options = {'REGISTER'}

    def execute(self, context):
        state = get_state()
        if state and state.active:
            self.report({'WARNING'}, "Already in correction mode")
            return {'CANCELLED'}

        # If state was pre-initialized by GenerateModel, use it;
        # otherwise try to load from last_run_output.json
        if not state:
            from .operators import _load_floor_plan_data
            data = _load_floor_plan_data()
            if not data:
                self.report({'ERROR'}, "No floor plan data to correct")
                return {'CANCELLED'}
            image_path = bpy.path.abspath(context.scene.fp3d_image_path)
            scale = context.scene.fp3d_scale_factor
            state = init_state(data, image_path, scale)

        if not state.walls and not state.doors and not state.windows:
            state.populate_from_json(state.original_data)

        state.create_blender_objects(context)
        _setup_topdown_view(context)

        # Register GPU draw handler
        from .correction_draw import draw_correction_overlay
        state.draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_correction_overlay, (state,), 'WINDOW', 'POST_PIXEL',
        )

        # Start modal timer for syncing
        self._timer = context.window_manager.event_timer_add(
            0.1, window=context.window,
        )
        context.window_manager.modal_handler_add(self)

        context.scene.fp3d_correction_active = True
        context.scene.fp3d_status = "Correction mode \u2014 G to move, X to delete, then Confirm"
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        state = get_state()
        if not state or not state.active:
            self._cleanup(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            state.sync_edge_meshes()
            state.recompute_wall_assignments()
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

        return {'PASS_THROUGH'}

    def _cleanup(self, context):
        if hasattr(self, '_timer') and self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None


# ── Confirm Corrections ──────────────────────────────────────────────

class FP3D_OT_ConfirmCorrection(bpy.types.Operator):
    bl_idname = "fp3d.confirm_correction"
    bl_label = "Confirm & Generate 3D"
    bl_description = "Apply corrections and generate 3D model"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        state = get_state()
        if not state or not state.active:
            self.report({'WARNING'}, "Not in correction mode")
            return {'CANCELLED'}

        # Read back corrected positions and export JSON
        corrected_data = state.to_json()

        # Save corrected data
        from .operators import _save_floor_plan_data
        _save_floor_plan_data(corrected_data)

        # Clean up correction objects
        state.cleanup(context)
        clear_state()
        context.scene.fp3d_correction_active = False

        # Build 3D geometry from corrected data
        from . import geometry
        from .operators import (
            _build_stories, _load_all_story_results, _load_story_result,
        )

        try:
            num_stories = getattr(context.scene, "fp3d_num_stories", 1)

            # Build per-story data list.
            # Story 0 uses the corrected data; upper stories use
            # inference results saved to disk by GenerateModel.
            story_data_list = [corrected_data]
            if num_stories > 1:
                items = getattr(context.scene, "fp3d_story_items", [])
                for i in range(1, num_stories):
                    # Load this story's inference result from disk
                    story_data = _load_story_result(i)
                    if story_data:
                        story_data_list.append(story_data)
                        continue

                    # Check JSON overrides
                    if i < len(items):
                        jp = bpy.path.abspath(items[i].json_path) if items[i].json_path else ""
                        if jp and os.path.isfile(jp):
                            try:
                                with open(jp, 'r') as f:
                                    story_data_list.append(json.load(f))
                                continue
                            except (json.JSONDecodeError, IOError):
                                pass

                    # Fallback to corrected ground floor
                    story_data_list.append(corrected_data)

            collection = geometry.create_floorplan_collection(context)
            _build_stories(context, corrected_data, collection,
                           story_data_list=story_data_list)
            context.scene.fp3d_status = "Done \u2014 corrections applied"
            self.report({'INFO'}, "3D model generated from corrected data")
            return {'FINISHED'}
        except Exception as e:
            context.scene.fp3d_status = f"Error: {e}"
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}


# ── Cancel Corrections ────────────────────────────────────────────────

class FP3D_OT_CancelCorrection(bpy.types.Operator):
    bl_idname = "fp3d.cancel_correction"
    bl_label = "Cancel Corrections"
    bl_description = "Discard corrections and return to normal mode"

    def execute(self, context):
        state = get_state()
        if state:
            state.cleanup(context)
        clear_state()
        context.scene.fp3d_correction_active = False
        context.scene.fp3d_status = "Correction cancelled"
        return {'FINISHED'}


# ── Add Wall (two-click modal) ────────────────────────────────────────

class FP3D_OT_AddWall(bpy.types.Operator):
    bl_idname = "fp3d.correction_add_wall"
    bl_label = "Add Wall"
    bl_description = "Click two points to add a new wall segment"
    bl_options = {'REGISTER', 'UNDO'}

    _start_pos = None

    def invoke(self, context, event):
        self._start_pos = None
        context.scene.fp3d_status = "Click to place wall start point..."
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            pos = _mouse_to_world_xy(context, event)
            if self._start_pos is None:
                self._start_pos = pos
                context.scene.fp3d_status = "Click to place wall end point..."
                return {'RUNNING_MODAL'}
            else:
                state = get_state()
                if state:
                    state.add_wall(self._start_pos, pos)
                context.scene.fp3d_status = "Wall added"
                return {'FINISHED'}

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            context.scene.fp3d_status = "Wall placement cancelled"
            return {'CANCELLED'}

        return {'PASS_THROUGH'}


# ── Add Door (single-click, snaps to wall) ────────────────────────────

class FP3D_OT_AddDoor(bpy.types.Operator):
    bl_idname = "fp3d.correction_add_door"
    bl_label = "Add Door"
    bl_description = "Click on a wall to place a door"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        context.scene.fp3d_status = "Click on a wall to place a door..."
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            pos = _mouse_to_world_xy(context, event)
            state = get_state()
            if not state:
                return {'CANCELLED'}
            wall_idx, snap_pos = state.snap_to_nearest_wall(pos)
            if wall_idx is not None:
                state.add_door(snap_pos, wall_index=wall_idx)
                context.scene.fp3d_status = "Door added"
                return {'FINISHED'}
            else:
                self.report({'WARNING'}, "No wall found near click position")
                return {'RUNNING_MODAL'}

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            context.scene.fp3d_status = "Door placement cancelled"
            return {'CANCELLED'}

        return {'PASS_THROUGH'}


# ── Add Window (single-click, snaps to wall) ──────────────────────────

class FP3D_OT_AddWindow(bpy.types.Operator):
    bl_idname = "fp3d.correction_add_window"
    bl_label = "Add Window"
    bl_description = "Click on a wall to place a window"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        context.scene.fp3d_status = "Click on a wall to place a window..."
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            pos = _mouse_to_world_xy(context, event)
            state = get_state()
            if not state:
                return {'CANCELLED'}
            wall_idx, snap_pos = state.snap_to_nearest_wall(pos)
            if wall_idx is not None:
                state.add_window(snap_pos, wall_index=wall_idx)
                context.scene.fp3d_status = "Window added"
                return {'FINISHED'}
            else:
                self.report({'WARNING'}, "No wall found near click position")
                return {'RUNNING_MODAL'}

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            context.scene.fp3d_status = "Window placement cancelled"
            return {'CANCELLED'}

        return {'PASS_THROUGH'}


# ── Add Full-Wall Window (single-click on a wall) ─────────────────────

class FP3D_OT_AddFullWallWindow(bpy.types.Operator):
    bl_idname = "fp3d.correction_add_full_window"
    bl_label = "Full-Wall Window"
    bl_description = "Click a wall to make it a floor-to-ceiling glass wall"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        context.scene.fp3d_status = "Click on a wall for a full-wall window..."
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            pos = _mouse_to_world_xy(context, event)
            state = get_state()
            if not state:
                return {'CANCELLED'}
            wall_idx, snap_pos = state.snap_to_nearest_wall(pos)
            if wall_idx is not None:
                # Place the window at the wall midpoint with full_wall flag
                wall = state.walls[wall_idx]
                ws = _wall_midpoint(wall)
                state.add_window(ws, width=9999, wall_index=wall_idx,
                                 full_wall=True)
                context.scene.fp3d_status = "Full-wall window added"
                return {'FINISHED'}
            else:
                self.report({'WARNING'}, "No wall found near click position")
                return {'RUNNING_MODAL'}

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            context.scene.fp3d_status = "Window placement cancelled"
            return {'CANCELLED'}

        return {'PASS_THROUGH'}


def _wall_midpoint(wall):
    """Return the midpoint [x, y] of a CorrectionWall."""
    if wall.start_empty and wall.end_empty:
        sx, sy = wall.start_empty.location.x, wall.start_empty.location.y
        ex, ey = wall.end_empty.location.x, wall.end_empty.location.y
    else:
        sx, sy = wall.start
        ex, ey = wall.end
    return [(sx + ex) / 2, (sy + ey) / 2]


# ── Delete Selected Element ───────────────────────────────────────────

class FP3D_OT_DeleteElement(bpy.types.Operator):
    bl_idname = "fp3d.correction_delete"
    bl_label = "Delete Selected Element"
    bl_description = "Remove the selected correction element"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        state = get_state()
        if not state:
            return {'CANCELLED'}

        removed = False
        for obj in list(context.selected_objects):
            if obj.get("fp3d_correction_type"):
                if state.remove_element_by_object(obj):
                    removed = True

        if removed:
            context.scene.fp3d_status = "Element deleted"
        else:
            self.report({'WARNING'}, "No correction element selected")

        return {'FINISHED'}
