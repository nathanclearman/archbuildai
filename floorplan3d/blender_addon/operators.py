import bpy
import json
import os
import threading
from pathlib import Path


# ── Helpers ────────────────────────────────────────────────────────────

def _get_floor_plan_data_path():
    """Return the path to the last_run_output.json file."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "model", "data", "last_run_output.json",
    )


def _load_floor_plan_data():
    """Load the most recent floor plan data from disk."""
    path = _get_floor_plan_data_path()
    if os.path.isfile(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def _get_ai_client(scene):
    """Return the appropriate AI client based on the user's backend choice.

    Returns either a ClaudeClient or LocalLLMClient, both sharing the
    same public interface (suggest_furniture, critique_layout, etc.).
    """
    backend = getattr(scene, "fp3d_ai_backend", "CLAUDE")

    if backend == "LOCAL":
        from .api.local_llm_client import LocalLLMClient
        model = scene.fp3d_local_model
        url = scene.fp3d_local_url
        return LocalLLMClient(model=model, base_url=url)
    else:
        from .api.claude_client import ClaudeClient
        api_key = scene.fp3d_claude_api_key.strip()
        if not api_key:
            raise ValueError("API key is required. Enter it in the Premium AI panel.")
        model = scene.fp3d_claude_model
        return ClaudeClient(api_key=api_key, model=model)


def _done_status(data):
    """Status line after a successful build, with the hybrid label report."""
    rep = (data or {}).get("_hybrid")
    if not rep:
        return "Done"
    if "error" in rep:
        return f"Done — labels from YOLO only (VLM unavailable: {rep['error'][:60]})"
    named = rep.get('labeled_from_ocr', 0) + rep.get('labeled_from_crop', 0)
    msg = f"Done — {named}/{rep.get('rooms', 0)} room names read from the plan"
    scale = rep.get("scale") or {}
    if scale.get("applied"):
        msg += f"; scale {scale.get('ppm')} px/m from {scale.get('used')} dimensions"
    elif scale:
        msg += "; scale kept from Scale Factor"
    return msg


def _save_floor_plan_data(data):
    """Save floor plan data to disk."""
    path = _get_floor_plan_data_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def _get_story_data_dir():
    """Return the directory for per-story inference data files."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "model", "data",
    )


def _save_story_results(results):
    """Save per-story inference results to disk.

    Writes each story's data to a separate JSON file so it survives
    addon reloads (module-level variables get wiped on reload).

    Args:
        results: dict mapping story_index (int) → floor plan dict.
    """
    if not results:
        return
    data_dir = _get_story_data_dir()
    os.makedirs(data_dir, exist_ok=True)

    # Clean up old story files first
    for f in os.listdir(data_dir):
        if f.startswith("story_") and f.endswith("_output.json"):
            os.remove(os.path.join(data_dir, f))

    for story_idx, data in results.items():
        path = os.path.join(data_dir, f"story_{story_idx}_output.json")
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


def _load_story_result(story_idx):
    """Load a single story's inference result from disk.

    Returns:
        dict or None: Floor plan data for the given story index.
    """
    path = os.path.join(_get_story_data_dir(),
                        f"story_{story_idx}_output.json")
    if os.path.isfile(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def _load_all_story_results():
    """Load all per-story inference results from disk.

    Returns:
        dict: story_index (int) → floor plan dict.
    """
    data_dir = _get_story_data_dir()
    results = {}
    if not os.path.isdir(data_dir):
        return results
    for f in os.listdir(data_dir):
        if f.startswith("story_") and f.endswith("_output.json"):
            try:
                idx = int(f.split("_")[1])
                with open(os.path.join(data_dir, f), 'r') as fp:
                    results[idx] = json.load(fp)
            except (ValueError, json.JSONDecodeError, IOError):
                pass
    return results


def _run_inference_sync(image_path, model_type, scale_factor, conf_threshold,
                        api_key="", claude_model="",
                        refine_with_claude=False, auto_scale=True):
    """Run model inference synchronously on a single image.

    This is a standalone helper so it can be called from both the main
    GenerateModel operator and the per-story loading pipeline.

    Args:
        image_path: Path to the floor plan image.
        model_type: 'HYBRID', 'YOLO', 'QWEN', or 'CLAUDE_VISION'.
        scale_factor: Pixels per meter.
        conf_threshold: Detection confidence threshold.
        api_key: Claude API key (required for CLAUDE_VISION).
        claude_model: Claude model name (for CLAUDE_VISION).
        refine_with_claude: If True and api_key is set, refine local
            model output using Claude Vision for better accuracy.

    Returns:
        dict: Parsed floor plan data.
    """
    if model_type == 'CLAUDE_VISION':
        from .api.claude_vision_client import ClaudeVisionClient
        client = ClaudeVisionClient(api_key=api_key, model=claude_model)
        return client.predict(
            image_path, pixels_per_meter=scale_factor,
            conf_threshold=conf_threshold,
        )
    elif model_type == 'YOLO':
        from .api.yolo_model import YOLOModelClient
        from .api import cleanup
        client = YOLOModelClient()
        result = client.predict(
            image_path, pixels_per_meter=scale_factor,
            conf_threshold=conf_threshold,
        )
        result, clean_report = cleanup.clean_cv_plan(result)
        print(f"[FP3D] cleanup: {clean_report}")
    elif model_type == 'QWEN':
        from .api.qwen_client import QwenModelClient
        client = QwenModelClient()
        result = client.predict(
            image_path, pixels_per_meter=scale_factor,
            conf_threshold=conf_threshold,
        )
    elif model_type == 'HYBRID':
        # Geometry from YOLO (walls / openings / room polygons), room names
        # read off the image by the VLM. If the VLM environment is missing
        # the geometry still comes through with YOLO's heuristic labels and
        # the reason is surfaced in the status line.
        from .api.yolo_model import YOLOModelClient
        from .api import qwen_client, cleanup
        result = YOLOModelClient().predict(
            image_path, pixels_per_meter=scale_factor,
            conf_threshold=conf_threshold,
        )
        # Make the outside shape match before naming rooms: drop walls read
        # off dimension lines / balcony hatching, remove sliver polygons,
        # close open stretches of the outline.
        result, clean_report = cleanup.clean_cv_plan(result)
        print(f"[FP3D] cleanup: {clean_report}")
        try:
            result, report = qwen_client.label_rooms(image_path, result, scale_factor)
            result["_hybrid"] = report
        except Exception as e:  # noqa: BLE001 — degrade, don't lose the geometry
            print(f"[FP3D] hybrid: VLM label pass unavailable, keeping YOLO labels: {e}")
            result["_hybrid"] = {"error": str(e)}
        if auto_scale and "error" not in result["_hybrid"]:
            # Scale from the printed dimension strings; the user's Scale
            # Factor is only the starting guess. Failure keeps the guess.
            try:
                result, srep = qwen_client.auto_scale(image_path, result, scale_factor)
                result["_hybrid"]["scale"] = srep
                print(f"[FP3D] auto-scale: {srep}")
            except Exception as e:  # noqa: BLE001
                print(f"[FP3D] auto-scale unavailable, keeping Scale Factor: {e}")
                result["_hybrid"]["scale"] = {"applied": False, "error": str(e)}
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Optionally refine local model output with Claude Vision
    if refine_with_claude and api_key and api_key.strip():
        from .api.claude_vision_client import ClaudeVisionClient
        vision = ClaudeVisionClient(api_key=api_key, model=claude_model)
        result = vision.refine_with_vision(
            image_path, result, pixels_per_meter=scale_factor)

    return result


def _load_story_data_list(scene, primary_data=None, run_inference=False):
    """Build a list of floor plan dicts, one per story.

    For each story, checks the per-story CollectionProperty for a custom
    JSON path or image path.  Falls back to the primary floor plan data
    when no per-story override is set.

    Args:
        scene: Blender scene.
        primary_data: Default floor plan data (used as fallback).
        run_inference: If True, run model inference on per-story image
            paths that don't have a JSON override.  This should only be
            True when called from a background thread (inference is slow).

    Returns:
        list[dict]: Floor plan data for each story (length == num_stories).
    """
    num = getattr(scene, "fp3d_num_stories", 1)
    items = getattr(scene, "fp3d_story_items", [])

    if primary_data is None:
        primary_data = _load_floor_plan_data()

    # Read scene inference settings once (safe — these are simple values)
    model_type = scene.fp3d_model_type
    scale_factor = scene.fp3d_scale_factor
    sensitivity = scene.fp3d_sensitivity
    conf_map = {'LOW': 0.6, 'MEDIUM': 0.5, 'HIGH': 0.35, 'VERY_HIGH': 0.2}
    conf_threshold = conf_map.get(sensitivity, 0.35)

    # Claude Vision needs API key + model (read once from scene)
    api_key = getattr(scene, "fp3d_claude_api_key", "")
    claude_model = getattr(scene, "fp3d_claude_model", "")
    refine = (getattr(scene, "fp3d_refine_with_claude", False)
              and model_type != 'CLAUDE_VISION')

    result = []
    for i in range(num):
        # Story 0 (ground floor) always uses the primary data from the
        # main "Floor Plan" input — no per-story override for it.
        if i == 0:
            result.append(primary_data)
            continue

        if i < len(items):
            item = items[i]
            # Try JSON override first
            jp = bpy.path.abspath(item.json_path) if item.json_path else ""
            if jp and os.path.isfile(jp):
                try:
                    with open(jp, 'r') as f:
                        result.append(json.load(f))
                    continue
                except (json.JSONDecodeError, IOError):
                    pass  # fall through

            # Try per-story image (requires inference)
            ip = bpy.path.abspath(item.image_path) if item.image_path else ""
            if ip and os.path.isfile(ip) and run_inference:
                try:
                    story_data = _run_inference_sync(
                        ip, model_type, scale_factor, conf_threshold,
                        api_key=api_key, claude_model=claude_model,
                        refine_with_claude=refine)
                    result.append(story_data)
                    continue
                except Exception:
                    pass  # fall through to primary

        # Fall back to the primary (shared) floor plan data
        result.append(primary_data)

    return result


def _build_stories(context, floor_plan_data, collection, story_data_list=None):
    """Build geometry for all stories (single or multi-story).

    When num_stories == 1, objects go directly into the parent collection
    with no name prefix (identical to pre-multi-story behaviour).
    When num_stories > 1, each story gets a Story_N sub-collection and
    an S{N}_ name prefix.

    Args:
        context: Blender context.
        floor_plan_data: Default floor plan data (used when story_data_list
            is None or a story has no override).
        collection: Parent Blender collection.
        story_data_list: Optional list of per-story floor plan dicts.

    Returns a dict summarising what was generated.
    """
    from . import geometry

    wall_height = context.scene.fp3d_wall_height
    generate_ceiling = context.scene.fp3d_generate_ceiling
    num_stories = getattr(context.scene, "fp3d_num_stories", 1)

    stats = {}
    total_walls = 0
    total_doors = 0
    total_door_panels = 0
    total_windows = 0
    total_floors = 0
    total_ceilings = 0
    total_labels = 0

    # Reference data for alignment (story 0's data)
    ref_data = (story_data_list[0] if story_data_list
                else floor_plan_data)

    for story in range(num_stories):
        # Use per-story data if available, otherwise fall back
        data = floor_plan_data
        if story_data_list and story < len(story_data_list):
            data = story_data_list[story]

        # Align upper stories to the ground floor's coordinate system.
        # Each image is parsed independently, so their origins differ.
        if story > 0 and data is not ref_data:
            data = geometry.align_story_to_reference(data, ref_data)

        # Sanitize noisy detections (adaptive confidence filtering,
        # room count caps, phantom room removal)
        data = geometry.sanitize_floor_plan_data(data)

        # Deduplicate per-story (each story may have different walls)
        data = geometry.deduplicate_walls(data)

        z_offset = story * wall_height
        if num_stories > 1:
            col = geometry.get_or_create_story_collection(collection, story)
            prefix = f"S{story}_"
        else:
            col = collection
            prefix = ""

        total_walls += geometry.generate_walls(
            data, col, wall_height,
            z_offset=z_offset, name_prefix=prefix)
        total_doors += geometry.generate_door_openings(
            data, col, wall_height,
            z_offset=z_offset, name_prefix=prefix)
        total_door_panels += geometry.generate_door_panels(
            data, col, wall_height,
            z_offset=z_offset, name_prefix=prefix)
        total_windows += geometry.generate_window_openings(
            data, col, wall_height,
            z_offset=z_offset, name_prefix=prefix)
        geometry.generate_window_panes(
            data, col, wall_height,
            z_offset=z_offset, name_prefix=prefix)
        total_floors += geometry.generate_floors(
            data, col,
            z_offset=z_offset, name_prefix=prefix)
        if generate_ceiling:
            total_ceilings += geometry.generate_ceilings(
                data, col, wall_height,
                z_offset=z_offset, name_prefix=prefix)
        total_labels += geometry.generate_room_labels(
            data, col,
            z_offset=z_offset, name_prefix=prefix)

    stats["walls"] = total_walls
    stats["doors"] = total_doors
    stats["door_panels"] = total_door_panels
    stats["windows"] = total_windows
    stats["floors"] = total_floors
    if total_ceilings:
        stats["ceilings"] = total_ceilings
    stats["labels"] = total_labels

    # Generate staircases when multi-story
    if num_stories > 1:
        stair_list = story_data_list or [floor_plan_data] * num_stories
        stair_count = geometry.generate_staircases(
            context, stair_list, collection)
        if stair_count:
            stats["staircases"] = stair_count

    return stats


# ── Core operators ─────────────────────────────────────────────────────

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
        from .api.sample_plans import get_mock_output, get_mock_studio, get_mock_three_bedroom

        if self.sample == 'STUDIO':
            data = get_mock_studio()
        elif self.sample == 'THREE_BEDROOM':
            data = get_mock_three_bedroom()
        else:
            data = get_mock_output()

        try:
            collection = geometry.create_floorplan_collection(context)
            stats = _build_stories(context, data, collection)

            # Save so Claude features can use it
            _save_floor_plan_data(data)

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


class FP3D_OT_LoadStoryFloorPlan(bpy.types.Operator):
    bl_idname = "fp3d.load_story_floor_plan"
    bl_label = "Load Story Floor Plan"
    bl_description = "Load a floor plan image or JSON for a specific story"

    story_index: bpy.props.IntProperty(default=0)
    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(
        default="*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.json",
        options={'HIDDEN'},
    )

    def execute(self, context):
        items = context.scene.fp3d_story_items
        if self.story_index < len(items):
            ext = os.path.splitext(self.filepath)[1].lower()
            if ext == '.json':
                items[self.story_index].json_path = self.filepath
            else:
                items[self.story_index].image_path = self.filepath
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

    # Stores per-story inference results when multi-story.
    _story_results = None

    def execute(self, context):
        scene = context.scene
        json_path = bpy.path.abspath(scene.fp3d_json_path)
        image_path = bpy.path.abspath(scene.fp3d_image_path)
        num_stories = getattr(scene, "fp3d_num_stories", 1)

        # Validate: if multi-story, upper stories need images or JSON
        if num_stories > 1:
            items = getattr(scene, "fp3d_story_items", [])
            for i in range(1, num_stories):
                if i < len(items):
                    item = items[i]
                    ip = bpy.path.abspath(item.image_path) if item.image_path else ""
                    jp = bpy.path.abspath(item.json_path) if item.json_path else ""
                    has_input = ((ip and os.path.isfile(ip)) or
                                 (jp and os.path.isfile(jp)))
                else:
                    has_input = False
                if not has_input:
                    label = items[i].label if i < len(items) else f"Story {i}"
                    self.report(
                        {'ERROR'},
                        f"No floor plan loaded for {label}. "
                        f"Load an image in the '{label}' slot under "
                        f"'Upper Story Floor Plans'.")
                    return {'CANCELLED'}

        # Validate Claude Vision requires API key
        if scene.fp3d_model_type == 'CLAUDE_VISION':
            api_key = getattr(scene, "fp3d_claude_api_key", "")
            if not api_key or not api_key.strip():
                self.report(
                    {'ERROR'},
                    "Premium Vision requires an API key. "
                    "Enter it in the Premium AI panel below.")
                return {'CANCELLED'}

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

        # Save so Claude features can reference it
        _save_floor_plan_data(floor_plan_data)
        return self._build_geometry(context, floor_plan_data)

    # Map sensitivity enum to confidence thresholds
    _SENSITIVITY_MAP = {
        'LOW': 0.6,
        'MEDIUM': 0.5,
        'HIGH': 0.35,
        'VERY_HIGH': 0.2,
    }

    def _start_model_inference(self, context, image_path):
        """Start model inference in a background thread.

        For multi-story builds, runs inference on ALL story images
        sequentially in the same background thread.
        """
        scene = context.scene
        num_stories = getattr(scene, "fp3d_num_stories", 1)

        model_type = scene.fp3d_model_type
        scale_factor = scene.fp3d_scale_factor
        sensitivity = scene.fp3d_sensitivity
        conf_threshold = self._SENSITIVITY_MAP.get(sensitivity, 0.35)

        # Claude Vision needs API credentials (read from scene on main thread)
        api_key = getattr(scene, "fp3d_claude_api_key", "") or ""
        claude_model = getattr(scene, "fp3d_claude_model", "") or ""

        # Collect all image paths to process: [(story_index, image_path), ...]
        # Story 0 always uses the main image.
        image_jobs = [(0, image_path)]

        if num_stories > 1:
            items = getattr(scene, "fp3d_story_items", [])
            for i in range(1, num_stories):
                if i < len(items):
                    item = items[i]
                    # JSON overrides are loaded later (no inference needed)
                    jp = bpy.path.abspath(item.json_path) if item.json_path else ""
                    if jp and os.path.isfile(jp):
                        continue  # skip — will be loaded from JSON in _build_geometry
                    ip = bpy.path.abspath(item.image_path) if item.image_path else ""
                    if ip and os.path.isfile(ip):
                        image_jobs.append((i, ip))

            n_images = len(image_jobs)
            context.scene.fp3d_status = (
                f"Running model inference on {n_images} floor plan"
                f"{'s' if n_images > 1 else ''}...")
        else:
            context.scene.fp3d_status = "Running model inference..."

        self._result = None
        self._story_results = None
        self._error = None
        refine = (getattr(scene, "fp3d_refine_with_claude", False)
                  and model_type != 'CLAUDE_VISION')
        auto_scale = getattr(scene, "fp3d_auto_scale", True)
        self._thread = threading.Thread(
            target=self._run_inference,
            args=(image_jobs, model_type, scale_factor, conf_threshold,
                  api_key, claude_model, refine, auto_scale),
            daemon=True,
        )
        self._thread.start()

        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _run_inference(self, image_jobs, model_type, scale_factor, conf_threshold,
                       api_key="", claude_model="",
                       refine_with_claude=False, auto_scale=True):
        """Run model inference on all story images in background thread.

        Args:
            image_jobs: list of (story_index, image_path) tuples.
        """
        try:
            results = {}
            for story_idx, img_path in image_jobs:
                results[story_idx] = _run_inference_sync(
                    img_path, model_type, scale_factor, conf_threshold,
                    api_key=api_key, claude_model=claude_model,
                    refine_with_claude=refine_with_claude, auto_scale=auto_scale)
            # Primary result is always story 0
            self._result = results.get(0)
            self._story_results = results
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

        # Save result for Claude features
        _save_floor_plan_data(self._result)

        # Save ALL per-story results to disk so ConfirmCorrection can
        # access them even after an addon reload (module vars get wiped).
        _save_story_results(self._story_results)

        # Enter correction mode if enabled, otherwise build directly
        if context.scene.fp3d_correction_enabled:
            from .correction import init_state
            image_path = bpy.path.abspath(context.scene.fp3d_image_path)
            scale = context.scene.fp3d_scale_factor
            init_state(self._result, image_path, scale)
            bpy.ops.fp3d.enter_correction('INVOKE_DEFAULT')
            return {'FINISHED'}

        context.scene.fp3d_status = "Building geometry..."
        result = self._build_geometry(context, self._result)
        if result == {'FINISHED'}:
            context.scene.fp3d_status = _done_status(self._result)
        return result

    def _build_geometry(self, context, floor_plan_data):
        """Build 3D geometry from parsed floor plan data."""
        from . import geometry

        try:
            num_stories = getattr(context.scene, "fp3d_num_stories", 1)

            # Build per-story data list from pre-computed inference results
            # and/or JSON overrides.
            story_data_list = self._build_story_data_list(
                context.scene, floor_plan_data, num_stories)

            collection = geometry.create_floorplan_collection(context)
            stats = _build_stories(
                context, floor_plan_data, collection,
                story_data_list=story_data_list)

            summary = ", ".join(f"{v} {k}" for k, v in stats.items() if v)
            context.scene.fp3d_status = f"Done — {summary}"
            self.report({'INFO'}, f"3D model generated: {summary}")
            return {'FINISHED'}
        except Exception as e:
            context.scene.fp3d_status = f"Error: {e}"
            self.report({'ERROR'}, f"Geometry generation failed: {e}")
            return {'CANCELLED'}

    def _build_story_data_list(self, scene, primary_data, num_stories):
        """Assemble per-story floor plan data from inference results + JSON.

        Uses self._story_results (populated by the background thread) for
        image-based stories, and loads JSON overrides directly.
        """
        if num_stories <= 1:
            return [primary_data]

        items = getattr(scene, "fp3d_story_items", [])
        # Use in-memory results if available, otherwise load from disk
        story_results = self._story_results or _load_all_story_results()
        result = []

        for i in range(num_stories):
            # Check if inference result exists for this story
            if i in story_results:
                result.append(story_results[i])
                continue

            # Check for JSON override (these weren't sent to the thread)
            if i < len(items):
                jp = bpy.path.abspath(items[i].json_path) if items[i].json_path else ""
                if jp and os.path.isfile(jp):
                    try:
                        with open(jp, 'r') as f:
                            result.append(json.load(f))
                        continue
                    except (json.JSONDecodeError, IOError):
                        pass

            # Fallback to primary
            result.append(primary_data)

        return result


class FP3D_OT_AdjustWallHeight(bpy.types.Operator):
    bl_idname = "fp3d.adjust_wall_height"
    bl_label = "Apply Wall Height"
    bl_description = "Update wall heights to the current setting"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        target_height = scene.fp3d_wall_height

        collection = bpy.data.collections.get("ArchbuildAI")
        if not collection:
            self.report({'WARNING'}, "No ArchbuildAI model found")
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


# ── Premium AI operators ───────────────────────────────────────────────

class FP3D_OT_SuggestFurniture(bpy.types.Operator):
    bl_idname = "fp3d.suggest_furniture"
    bl_label = "Suggest Furniture"
    bl_description = "Use Premium AI to suggest furniture placement for each room"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _thread = None
    _result = None
    _error = None

    def execute(self, context):
        scene = context.scene

        try:
            client = _get_ai_client(scene)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        floor_plan = _load_floor_plan_data()
        if not floor_plan or not floor_plan.get("rooms"):
            self.report({'ERROR'}, "No floor plan data found. Generate a model first.")
            return {'CANCELLED'}

        backend = getattr(scene, "fp3d_ai_backend", "CLAUDE")
        label = "local LLM" if backend == "LOCAL" else "Premium AI"
        context.scene.fp3d_status = f"Asking {label} for furniture suggestions..."

        self._result = None
        self._error = None
        self._thread = threading.Thread(
            target=self._call_ai,
            args=(client, floor_plan["rooms"]),
            daemon=True,
        )
        self._thread.start()

        self._timer = context.window_manager.event_timer_add(0.2, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _call_ai(self, client, rooms_data):
        try:
            self._result = client.suggest_furniture(rooms_data)
        except Exception as e:
            self._error = str(e)

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        if self._thread and self._thread.is_alive():
            return {'PASS_THROUGH'}

        context.window_manager.event_timer_remove(self._timer)
        self._timer = None

        if self._error:
            context.scene.fp3d_status = f"Error: {self._error}"
            self.report({'ERROR'}, f"AI error: {self._error}")
            return {'CANCELLED'}

        if not self._result:
            context.scene.fp3d_status = "Error: No furniture suggestions received"
            self.report({'ERROR'}, "No furniture suggestions received")
            return {'CANCELLED'}

        # Save debug output
        try:
            debug_path = os.path.join(
                os.path.dirname(_get_floor_plan_data_path()),
                "last_furniture_output.json",
            )
            with open(debug_path, 'w') as f:
                json.dump(self._result, f, indent=2)
        except Exception:
            pass

        # Generate furniture geometry
        from . import geometry

        collection = bpy.data.collections.get("ArchbuildAI")
        if not collection:
            context.scene.fp3d_status = "Error: No ArchbuildAI collection"
            self.report({'ERROR'}, "No ArchbuildAI collection found")
            return {'CANCELLED'}

        # Remove existing furniture first
        geometry.remove_furniture(collection)

        num_stories = getattr(context.scene, "fp3d_num_stories", 1)
        wall_height = context.scene.fp3d_wall_height
        total_count = 0
        for story in range(num_stories):
            z_offset = story * wall_height
            if num_stories > 1:
                col = geometry.get_or_create_story_collection(collection, story)
                prefix = f"S{story}_"
            else:
                col = collection
                prefix = ""
            total_count += geometry.generate_furniture(
                self._result, col, z_offset=z_offset, name_prefix=prefix)

        context.scene.fp3d_status = f"Done — placed {total_count} furniture items"
        self.report({'INFO'}, f"Placed {total_count} furniture items")
        return {'FINISHED'}


class FP3D_OT_RemoveFurniture(bpy.types.Operator):
    bl_idname = "fp3d.remove_furniture"
    bl_label = "Remove Furniture"
    bl_description = "Remove all AI-placed furniture from the model"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from . import geometry

        collection = bpy.data.collections.get("ArchbuildAI")
        if not collection:
            self.report({'WARNING'}, "No ArchbuildAI model found")
            return {'CANCELLED'}

        removed = geometry.remove_furniture(collection)
        self.report({'INFO'}, f"Removed {removed} furniture items")
        return {'FINISHED'}


class FP3D_OT_GenerateExterior(bpy.types.Operator):
    bl_idname = "fp3d.generate_exterior"
    bl_label = "Generate Exterior"
    bl_description = "Use AI to generate roof, facade, and exterior details from an architectural style"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _thread = None
    _result = None
    _error = None

    def execute(self, context):
        scene = context.scene
        style = getattr(scene, "fp3d_exterior_style", "")
        if not style or not style.strip():
            self.report({'ERROR'}, "Enter an architectural style (e.g. 'modern Korean house')")
            return {'CANCELLED'}

        try:
            client = _get_ai_client(scene)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        floor_plan = _load_floor_plan_data()
        if not floor_plan:
            self.report({'ERROR'}, "No floor plan data found. Generate a model first.")
            return {'CANCELLED'}

        backend = getattr(scene, "fp3d_ai_backend", "CLAUDE")
        label = "local LLM" if backend == "LOCAL" else "Premium AI"
        context.scene.fp3d_status = f"Asking {label} to design exterior..."

        # Optional reference image for Premium AI vision
        ref_image = bpy.path.abspath(
            getattr(scene, "fp3d_reference_image", "") or ""
        )

        self._result = None
        self._error = None
        self._thread = threading.Thread(
            target=self._call_ai,
            args=(client, floor_plan, style, ref_image),
            daemon=True,
        )
        self._thread.start()

        self._timer = context.window_manager.event_timer_add(0.2, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _call_ai(self, client, floor_plan, style, ref_image_path=""):
        try:
            self._result = client.generate_exterior(
                floor_plan, style, reference_image_path=ref_image_path
            )
        except Exception as e:
            self._error = str(e)

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        if self._thread and self._thread.is_alive():
            return {'PASS_THROUGH'}

        context.window_manager.event_timer_remove(self._timer)
        self._timer = None

        if self._error:
            context.scene.fp3d_status = f"Error: {self._error}"
            self.report({'ERROR'}, f"AI error: {self._error}")
            return {'CANCELLED'}

        if not self._result:
            context.scene.fp3d_status = "Error: No exterior config received"
            self.report({'ERROR'}, "No exterior configuration received")
            return {'CANCELLED'}

        # Save debug output
        try:
            debug_path = os.path.join(
                os.path.dirname(_get_floor_plan_data_path()),
                "last_exterior_output.json",
            )
            with open(debug_path, 'w') as f:
                json.dump(self._result, f, indent=2)
        except Exception:
            pass

        from . import geometry

        floor_plan = _load_floor_plan_data()
        collection = bpy.data.collections.get("ArchbuildAI")
        if not collection:
            context.scene.fp3d_status = "Error: No ArchbuildAI collection"
            self.report({'ERROR'}, "No ArchbuildAI collection found")
            return {'CANCELLED'}

        wall_height = context.scene.fp3d_wall_height
        num_stories = getattr(context.scene, "fp3d_num_stories", 1)
        total_height = wall_height * num_stories

        # For multi-story buildings, load per-story data so the roof
        # follows the top story's footprint (not the ground floor).
        # Align upper stories to story 0's coordinate system — the raw
        # JSON files have independent origins since each image is parsed
        # separately.  Without this the roof footprint is offset from
        # the actual 3D walls (which were aligned during _build_stories).
        story_data = None
        if num_stories > 1:
            story_data = _load_all_story_results()
            if story_data and 0 in story_data:
                ref_data = story_data[0]
                for idx in story_data:
                    if idx > 0:
                        story_data[idx] = geometry.align_story_to_reference(
                            story_data[idx], ref_data)

        try:
            stats = geometry.generate_exterior(
                floor_plan, self._result, collection, total_height,
                story_data=story_data,
            )
            parts = []
            if stats.get("restyled_walls"):
                parts.append(f"{stats['restyled_walls']} walls restyled")
            roof_type = stats.get("roof", "unknown")
            parts.append(f"{roof_type} roof")
            if stats.get("foundation"):
                parts.append("foundation")
            if stats.get("window_frames"):
                parts.append(f"{stats['window_frames']} window frames")
            if stats.get("door_surrounds"):
                parts.append(f"{stats['door_surrounds']} door surrounds")
            if stats.get("details"):
                parts.append(f"{stats['details']} details")
            summary = ", ".join(parts)
            context.scene.fp3d_status = f"Done — Exterior: {summary}"
            self.report({'INFO'}, f"Exterior generated: {summary}")
            return {'FINISHED'}
        except Exception as e:
            context.scene.fp3d_status = f"Error building exterior: {e}"
            self.report({'ERROR'}, f"Failed to build exterior: {e}")
            return {'CANCELLED'}


class FP3D_OT_RemoveExterior(bpy.types.Operator):
    bl_idname = "fp3d.remove_exterior"
    bl_label = "Remove Exterior"
    bl_description = "Remove all exterior objects (roof, facade, details)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from . import geometry

        collection = bpy.data.collections.get("ArchbuildAI")
        if not collection:
            self.report({'WARNING'}, "No ArchbuildAI model found")
            return {'CANCELLED'}

        removed = geometry.remove_exterior(collection)
        self.report({'INFO'}, f"Removed {removed} exterior objects")
        return {'FINISHED'}


class FP3D_OT_CritiqueLayout(bpy.types.Operator):
    bl_idname = "fp3d.critique_layout"
    bl_label = "Critique Layout"
    bl_description = "Use AI to review and score the floor plan layout"
    bl_options = {'REGISTER'}

    _timer = None
    _thread = None
    _result = None
    _error = None

    def execute(self, context):
        scene = context.scene

        try:
            client = _get_ai_client(scene)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        floor_plan = _load_floor_plan_data()
        if not floor_plan:
            self.report({'ERROR'}, "No floor plan data found. Generate a model first.")
            return {'CANCELLED'}

        backend = getattr(scene, "fp3d_ai_backend", "CLAUDE")
        label = "local LLM" if backend == "LOCAL" else "Premium AI"
        context.scene.fp3d_status = f"Asking {label} to critique layout..."

        self._result = None
        self._error = None
        self._thread = threading.Thread(
            target=self._call_ai,
            args=(client, floor_plan),
            daemon=True,
        )
        self._thread.start()

        self._timer = context.window_manager.event_timer_add(0.2, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _call_ai(self, client, floor_plan):
        try:
            self._result = client.critique_layout(floor_plan)
        except Exception as e:
            self._error = str(e)

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        if self._thread and self._thread.is_alive():
            return {'PASS_THROUGH'}

        context.window_manager.event_timer_remove(self._timer)
        self._timer = None

        if self._error:
            context.scene.fp3d_status = f"Error: {self._error}"
            self.report({'ERROR'}, f"AI error: {self._error}")
            return {'CANCELLED'}

        if not self._result:
            context.scene.fp3d_status = "Error: No critique received"
            self.report({'ERROR'}, "No critique received")
            return {'CANCELLED'}

        # Save debug output
        try:
            debug_path = os.path.join(
                os.path.dirname(_get_floor_plan_data_path()),
                "last_critique_output.json",
            )
            with open(debug_path, 'w') as f:
                json.dump(self._result, f, indent=2)
        except Exception:
            pass

        # Write critique to a Blender text block
        critique = self._result
        text_name = "ArchbuildAI Critique"
        text_block = bpy.data.texts.get(text_name)
        if text_block:
            text_block.clear()
        else:
            text_block = bpy.data.texts.new(text_name)

        lines = []
        score = critique.get("score", "N/A")
        lines.append("=== ArchbuildAI Layout Critique ===")
        lines.append(f"Overall Score: {score}/10")
        lines.append("")

        strengths = critique.get("strengths", [])
        if strengths:
            lines.append("STRENGTHS:")
            for s in strengths:
                lines.append(f"  + {s}")
            lines.append("")

        issues = critique.get("issues", [])
        if issues:
            lines.append("ISSUES:")
            for issue in issues:
                if isinstance(issue, dict):
                    sev = issue.get("severity", "")
                    desc = issue.get("description", issue.get("issue", str(issue)))
                    lines.append(f"  [{sev.upper()}] {desc}")
                else:
                    lines.append(f"  - {issue}")
            lines.append("")

        suggestions = critique.get("suggestions", [])
        if suggestions:
            lines.append("SUGGESTIONS:")
            for s in suggestions:
                lines.append(f"  > {s}")

        text_block.write("\n".join(lines))

        context.scene.fp3d_status = f"Done — Layout score: {score}/10 (see Text Editor)"
        self.report({'INFO'}, f"Layout score: {score}/10 — see '{text_name}' in Text Editor")
        return {'FINISHED'}


class FP3D_OT_ModifyLayout(bpy.types.Operator):
    bl_idname = "fp3d.modify_layout"
    bl_label = "Modify Layout"
    bl_description = "Use AI to modify the floor plan based on natural language"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _thread = None
    _result = None
    _error = None

    def execute(self, context):
        scene = context.scene
        prompt = scene.fp3d_claude_prompt

        try:
            client = _get_ai_client(scene)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        if not prompt or not prompt.strip():
            self.report({'ERROR'}, "Enter a modification prompt (e.g. 'make the kitchen bigger')")
            return {'CANCELLED'}

        floor_plan = _load_floor_plan_data()
        if not floor_plan:
            self.report({'ERROR'}, "No floor plan data found. Generate a model first.")
            return {'CANCELLED'}

        backend = getattr(scene, "fp3d_ai_backend", "CLAUDE")
        label = "local LLM" if backend == "LOCAL" else "Premium AI"
        context.scene.fp3d_status = f"Asking {label} to modify layout..."

        self._result = None
        self._error = None
        self._thread = threading.Thread(
            target=self._call_ai,
            args=(client, floor_plan, prompt),
            daemon=True,
        )
        self._thread.start()

        self._timer = context.window_manager.event_timer_add(0.2, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _call_ai(self, client, floor_plan, prompt):
        try:
            self._result = client.interpret_modification(floor_plan, prompt)
        except Exception as e:
            self._error = str(e)

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        if self._thread and self._thread.is_alive():
            return {'PASS_THROUGH'}

        context.window_manager.event_timer_remove(self._timer)
        self._timer = None

        if self._error:
            context.scene.fp3d_status = f"Error: {self._error}"
            self.report({'ERROR'}, f"AI error: {self._error}")
            return {'CANCELLED'}

        if not self._result:
            context.scene.fp3d_status = "Error: No modified plan received"
            self.report({'ERROR'}, "No modified plan received")
            return {'CANCELLED'}

        # Validate the response has required keys
        if "walls" not in self._result or "rooms" not in self._result:
            context.scene.fp3d_status = "Error: AI returned invalid floor plan data"
            self.report({'ERROR'}, "Modified plan missing 'walls' or 'rooms'")
            return {'CANCELLED'}

        # Save the modified plan
        _save_floor_plan_data(self._result)

        # Rebuild geometry from the modified plan
        from . import geometry

        try:
            collection = geometry.create_floorplan_collection(context)
            _build_stories(context, self._result, collection)

            n_rooms = len(self._result.get("rooms", []))
            n_walls = len(self._result.get("walls", []))
            context.scene.fp3d_status = f"Done — Modified: {n_walls} walls, {n_rooms} rooms"
            self.report({'INFO'}, f"Layout modified: {n_walls} walls, {n_rooms} rooms")
            return {'FINISHED'}
        except Exception as e:
            context.scene.fp3d_status = f"Error rebuilding: {e}"
            self.report({'ERROR'}, f"Failed to rebuild geometry: {e}")
            return {'CANCELLED'}
