import bpy
import json
import os
import threading
from pathlib import Path


class FP3D_OT_GenerateSample(bpy.types.Operator):
    bl_idname = "fp3d.generate_sample"
    bl_label = "Generate Sample Model"
    bl_description = "Generate a sample 3D model using built-in mock data (no model or image needed)"
    bl_options = {'REGISTER', 'UNDO'}

    sample: bpy.props.EnumProperty(
        name="Sample",
        items=[
            ('APARTMENT', "Simple Apartment", "Two-room apartment with doors and windows"),
            ('STUDIO', "Studio", "Studio apartment with bathroom"),
            ('THREE_BEDROOM', "Three Bedroom", "Three-bedroom house with kitchen and living room"),
        ],
        default='APARTMENT',
    )

    def execute(self, context):
        from . import geometry
        from .api.local_model import get_mock_output, get_mock_studio, get_mock_three_bedroom

        if self.sample == 'STUDIO':
            data = get_mock_studio()
        elif self.sample == 'THREE_BEDROOM':
            data = get_mock_three_bedroom()
        else:
            data = get_mock_output()

        try:
            wall_height = context.scene.fp3d_wall_height
            generate_ceiling = context.scene.fp3d_generate_ceiling

            collection = geometry.create_floorplan_collection(context)
            stats = {}
            stats["walls"] = geometry.generate_walls(data, collection, wall_height)
            stats["doors"] = geometry.generate_door_openings(data, collection, wall_height)
            stats["windows"] = geometry.generate_window_openings(data, collection, wall_height)
            stats["floors"] = geometry.generate_floors(data, collection)
            if generate_ceiling:
                stats["ceilings"] = geometry.generate_ceilings(data, collection, wall_height)
            stats["labels"] = geometry.generate_room_labels(data, collection)

            summary = ", ".join(f"{v} {k}" for k, v in stats.items() if v)
            context.scene.fp3d_status = f"Sample generated: {summary}"
            self.report({'INFO'}, f"Sample model generated: {summary}")
            return {'FINISHED'}
        except Exception as e:
            context.scene.fp3d_status = f"Error: {e}"
            self.report({'ERROR'}, f"Sample generation failed: {e}")
            return {'CANCELLED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


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
        """Run model inference in background thread.

        Routing:
          * If the user explicitly enabled Claude Vision and an API key is set,
            use Claude Opus 4.6 as the primary parser.
          * Otherwise run the local CV model. If it fails AND an API key is
            available, auto-fall back to Claude Vision.
        """
        scene = bpy.context.scene
        api_key = (scene.fp3d_claude_api_key or "").strip()
        use_claude = bool(scene.fp3d_use_claude_vision) and bool(api_key)

        if use_claude:
            try:
                self._result = self._run_claude_vision(image_path, api_key)
                return
            except Exception as e:
                # If the user asked for Claude Vision and it failed, surface the
                # error — do not silently fall back to a possibly-untrained
                # local model.
                self._error = f"Claude Vision parse failed: {e}"
                return

        try:
            from .api.local_model import LocalModelClient
            client = LocalModelClient()
            self._result = client.predict(image_path)
        except Exception as local_err:
            if api_key:
                try:
                    self._result = self._run_claude_vision(image_path, api_key)
                    return
                except Exception as claude_err:
                    self._error = (
                        f"Local model failed ({local_err}); "
                        f"Claude Vision fallback also failed ({claude_err})"
                    )
                    return
            self._error = str(local_err)

    @staticmethod
    def _run_claude_vision(image_path, api_key):
        """Parse a floor plan image via Claude Opus 4.6 vision + tool use."""
        from .api.claude_client import ClaudeClient
        client = ClaudeClient(api_key)
        return client.parse_floor_plan_from_image(image_path)

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


class FP3D_OT_RunVisionAccuracyTest(bpy.types.Operator):
    """Run the Claude Vision accuracy loop on the currently-loaded image.

    Reuses the scene's image path + API key (entered in the Claude Vision
    panel) and iterates verify_and_repair against the bundled ground truth
    until the score meets the threshold or the iteration budget is spent.
    All API work happens on a background thread; results are reported to
    the status line on the main thread.
    """

    bl_idname = "fp3d.run_vision_accuracy_test"
    bl_label = "Run Vision Accuracy Test"
    bl_description = (
        "Parse the loaded floor plan with Claude Opus 4.6, score against the "
        "reference ground truth, and iterate until accurate"
    )

    _timer = None
    _thread = None
    _score = None
    _iterations = None
    _error = None
    _report_path = None

    def execute(self, context):
        scene = context.scene

        api_key = (scene.fp3d_claude_api_key or "").strip()
        if not api_key:
            self.report({'ERROR'}, "Enter your Anthropic API key in the Claude Vision panel")
            return {'CANCELLED'}

        image_path = bpy.path.abspath(scene.fp3d_image_path or "")
        if not image_path or not os.path.isfile(image_path):
            self.report({'ERROR'}, "Load a floor plan image first")
            return {'CANCELLED'}

        gt_path = Path(__file__).resolve().parent.parent / "tests" / "sample_plans" / "large_house_ground_truth.json"
        if not gt_path.is_file():
            self.report({'ERROR'}, f"Ground truth not found: {gt_path}")
            return {'CANCELLED'}

        threshold = float(scene.fp3d_vision_threshold)
        max_iter = int(scene.fp3d_vision_max_iterations)

        scene.fp3d_status = "Claude Vision: running parse..."

        self._score = None
        self._iterations = None
        self._error = None
        self._report_path = None
        self._thread = threading.Thread(
            target=self._run,
            args=(api_key, image_path, str(gt_path), threshold, max_iter),
            daemon=True,
        )
        self._thread.start()

        self._timer = context.window_manager.event_timer_add(0.25, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _run(self, api_key, image_path, gt_path, threshold, max_iter):
        try:
            from .api.claude_client import ClaudeClient
            ground_truth = json.loads(Path(gt_path).read_text())

            client = ClaudeClient(api_key=api_key, use_thinking=True, thinking_budget=8192)
            attempt = client.parse_floor_plan_from_image(
                image_path,
                user_notes=(
                    "Trace the exterior perimeter precisely. Include every "
                    "jog and notch. The plan has a triangular balcony on the "
                    "west side meeting the living room at an angle."
                ),
            )

            best_score = _score_plan(attempt, ground_truth)
            best_attempt = attempt
            iterations = 0

            while best_score < threshold and iterations < max_iter:
                iterations += 1
                repaired = client.verify_and_repair(image_path, best_attempt)
                s = _score_plan(repaired, ground_truth)
                if s > best_score:
                    best_score = s
                    best_attempt = repaired
                else:
                    break

            # Write the final attempt next to the image for inspection.
            report_path = Path(image_path).with_suffix(".vision_result.json")
            report_path.write_text(json.dumps(best_attempt, indent=2))

            self._score = best_score
            self._iterations = iterations
            self._report_path = str(report_path)
        except Exception as e:
            self._error = f"{type(e).__name__}: {e}"

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        if self._thread and self._thread.is_alive():
            return {'PASS_THROUGH'}

        context.window_manager.event_timer_remove(self._timer)
        self._timer = None

        if self._error:
            context.scene.fp3d_status = f"Vision test error: {self._error}"
            self.report({'ERROR'}, self._error)
            return {'CANCELLED'}

        score = self._score or 0.0
        iters = self._iterations if self._iterations is not None else 0
        threshold = float(context.scene.fp3d_vision_threshold)
        passed = score >= threshold

        verdict = "PASS" if passed else "FAIL"
        msg = (
            f"Vision test {verdict}: score={score:.3f} "
            f"(threshold {threshold:.2f}), repair passes={iters}"
        )
        if self._report_path:
            msg += f"  →  {self._report_path}"

        context.scene.fp3d_status = msg
        self.report({'INFO'} if passed else {'WARNING'}, msg)
        return {'FINISHED'}


def _score_plan(predicted, ground_truth):
    """Inline accuracy scoring — mirrors tools/vision_accuracy_test.py.

    Bounding-box IoU 50%, wall-count 20%, perimeter-length ratio 20%,
    room-count 10%. Inlined here so the operator has no cross-package
    dependency on the /tools directory when the add-on is installed.
    """
    pw = predicted.get("walls", []) or []
    gw = ground_truth.get("walls", []) or []

    def bbox(walls):
        if not walls:
            return (0.0, 0.0, 0.0, 0.0)
        xs, ys = [], []
        for w in walls:
            xs.extend([w["start"][0], w["end"][0]])
            ys.extend([w["start"][1], w["end"][1]])
        return (min(xs), min(ys), max(xs), max(ys))

    def iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        aa = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
        ab = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
        u = aa + ab - inter
        return inter / u if u > 0 else 0.0

    def length(w):
        dx = w["end"][0] - w["start"][0]
        dy = w["end"][1] - w["start"][1]
        return (dx * dx + dy * dy) ** 0.5

    bbox_iou = iou(bbox(pw), bbox(gw))
    count_s = 1.0 - min(1.0, abs(len(pw) - len(gw)) / max(1, len(gw)))
    pl = sum(length(w) for w in pw)
    gl = sum(length(w) for w in gw) or 1e-6
    length_s = 1.0 - min(1.0, abs(pl - gl) / gl)
    pr = len(predicted.get("rooms", []) or [])
    gr = len(ground_truth.get("rooms", []) or []) or 1
    room_s = 1.0 - min(1.0, abs(pr - gr) / gr)

    return 0.5 * bbox_iou + 0.2 * count_s + 0.2 * length_s + 0.1 * room_s


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
