import bpy
import json
import os
import threading
from pathlib import Path


class FP3D_OT_LoadFloorPlan(bpy.types.Operator):
    bl_idname = "fp3d.load_floor_plan"
    bl_label = "Load Floor Plan"
    bl_description = "Load a floor plan image for processing"

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')

    def execute(self, context):
        context.scene.fp3d_image_path = self.filepath
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class FP3D_OT_GenerateModel(bpy.types.Operator):
    bl_idname = "fp3d.generate_model"
    bl_label = "Generate 3D Model"
    bl_description = "Generate a 3D model from the floor plan"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _thread = None
    _result = None
    _error = None

    def execute(self, context):
        scene = context.scene
        json_path = bpy.path.abspath(scene.fp3d_json_path)
        image_path = bpy.path.abspath(scene.fp3d_image_path)

        # Determine input source
        if json_path and os.path.isfile(json_path):
            return self._generate_from_json(context, json_path)
        elif image_path and os.path.isfile(image_path):
            return self._start_model_inference(context, image_path)
        else:
            self.report({'ERROR'}, "No floor plan image or JSON file specified")
            return {'CANCELLED'}

    def _generate_from_json(self, context, json_path):
        """Generate directly from a JSON file."""
        try:
            with open(json_path, 'r') as f:
                floor_plan_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.report({'ERROR'}, f"Failed to load JSON: {e}")
            return {'CANCELLED'}

        return self._build_geometry(context, floor_plan_data)

    def _start_model_inference(self, context, image_path):
        """Start model inference in a background thread."""
        context.scene.fp3d_status = "Running model inference..."

        self._result = None
        self._error = None
        self._thread = threading.Thread(
            target=self._run_inference,
            args=(image_path,),
            daemon=True,
        )
        self._thread.start()

        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _run_inference(self, image_path):
        """Run model inference in background thread."""
        try:
            from .api.local_model import LocalModelClient
            client = LocalModelClient()
            self._result = client.predict(image_path)
        except Exception as e:
            self._error = str(e)

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        if self._thread and self._thread.is_alive():
            return {'PASS_THROUGH'}

        # Thread finished — clean up timer
        context.window_manager.event_timer_remove(self._timer)
        self._timer = None

        if self._error:
            context.scene.fp3d_status = f"Error: {self._error}"
            self.report({'ERROR'}, self._error)
            return {'CANCELLED'}

        if self._result is None:
            context.scene.fp3d_status = "Error: No result from model"
            self.report({'ERROR'}, "Model returned no result")
            return {'CANCELLED'}

        context.scene.fp3d_status = "Building geometry..."
        result = self._build_geometry(context, self._result)
        if result == {'FINISHED'}:
            context.scene.fp3d_status = "Done"
        return result

    def _build_geometry(self, context, floor_plan_data):
        """Build 3D geometry from parsed floor plan data."""
        from . import geometry

        try:
            wall_height = context.scene.fp3d_wall_height
            generate_ceiling = context.scene.fp3d_generate_ceiling

            collection = geometry.create_floorplan_collection(context)
            geometry.generate_walls(floor_plan_data, collection, wall_height)
            geometry.generate_door_openings(floor_plan_data, collection, wall_height)
            geometry.generate_window_openings(floor_plan_data, collection, wall_height)
            geometry.generate_floors(floor_plan_data, collection)
            if generate_ceiling:
                geometry.generate_ceilings(floor_plan_data, collection, wall_height)
            geometry.generate_room_labels(floor_plan_data, collection)

            context.scene.fp3d_status = "Done"
            self.report({'INFO'}, "3D model generated successfully")
            return {'FINISHED'}
        except Exception as e:
            context.scene.fp3d_status = f"Error: {e}"
            self.report({'ERROR'}, f"Geometry generation failed: {e}")
            return {'CANCELLED'}


class FP3D_OT_AdjustWallHeight(bpy.types.Operator):
    bl_idname = "fp3d.adjust_wall_height"
    bl_label = "Apply Wall Height"
    bl_description = "Update wall heights to the current setting"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        target_height = scene.fp3d_wall_height

        collection = bpy.data.collections.get("FloorPlan3D")
        if not collection:
            self.report({'WARNING'}, "No FloorPlan3D model found")
            return {'CANCELLED'}

        adjusted = 0
        for obj in collection.objects:
            if obj.get("fp3d_type") == "wall":
                obj.scale.z = target_height / obj.get("fp3d_original_height", 2.7)
                adjusted += 1

        self.report({'INFO'}, f"Adjusted {adjusted} wall(s) to {target_height}m")
        return {'FINISHED'}


class FP3D_OT_ExportModel(bpy.types.Operator):
    bl_idname = "fp3d.export_model"
    bl_label = "Export Model"
    bl_description = "Export the generated model"
    bl_options = {'REGISTER'}

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(default="*.fbx;*.obj;*.glb", options={'HIDDEN'})

    def execute(self, context):
        ext = Path(self.filepath).suffix.lower()

        if ext == '.fbx':
            bpy.ops.export_scene.fbx(filepath=self.filepath, use_selection=False)
        elif ext == '.obj':
            bpy.ops.wm.obj_export(filepath=self.filepath)
        elif ext in ('.glb', '.gltf'):
            bpy.ops.export_scene.gltf(filepath=self.filepath)
        else:
            self.report({'ERROR'}, f"Unsupported format: {ext}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Exported to {self.filepath}")
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
