bl_info = {
    "name": "ArchbuildAI",
    "author": "Nathan Clearman",
    "version": (1, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > ArchbuildAI",
    "description": "Generate 3D architectural models from 2D floor plan images",
    "category": "3D View",
}

import importlib
import site
import sys

import bpy


def _enable_user_site_packages():
    """Make packages pip-installed into Blender's Python visible in-process.

    Blender starts its embedded interpreter with user site-packages
    disabled, so `<blender-python> -m pip install ultralytics` lands in
    ~/.local/lib/pythonX.Y/site-packages (or the Windows/macOS equivalent)
    where a plain `import ultralytics` inside Blender can't see it. The
    in-process YOLO backend and Shapely-based polygon cleanup need those
    packages, so append the user site dir (never prepend — Blender's own
    bundled numpy etc. keep priority).
    """
    try:
        user_site = site.getusersitepackages()
    except Exception:
        return
    if user_site and user_site not in sys.path and __import__("os").path.isdir(user_site):
        sys.path.append(user_site)


_enable_user_site_packages()

# Reload submodules when Blender reloads the add-on (F8 or toggle).
# On first import these names don't exist yet, so only reload if already loaded.
if "operators" in dir():
    importlib.reload(local_model)
    importlib.reload(hybrid)
    importlib.reload(cleanup)
    importlib.reload(autoscale)
    importlib.reload(sample_plans)
    importlib.reload(yolo_model)
    importlib.reload(qwen_client)
    importlib.reload(claude_client)
    importlib.reload(local_llm_client)
    importlib.reload(geometry)
    importlib.reload(materials)
    importlib.reload(correction)
    importlib.reload(correction_ops)
    importlib.reload(correction_draw)
    importlib.reload(setup_env)
    importlib.reload(preferences)
    importlib.reload(operators)
    importlib.reload(panels)

from . import operators, panels, geometry, materials, preferences
from . import correction, correction_ops, correction_draw
from .api import local_model, hybrid, cleanup, autoscale, sample_plans, setup_env, yolo_model, qwen_client, claude_client, local_llm_client


# ── Per-story PropertyGroup ──────────────────────────────────────────

class FP3D_StoryItem(bpy.types.PropertyGroup):
    """Per-story floor plan configuration."""
    image_path: bpy.props.StringProperty(
        name="Floor Plan Image",
        description="Floor plan image for this story (empty = use main image)",
        subtype='FILE_PATH',
        default="",
    )
    json_path: bpy.props.StringProperty(
        name="JSON Override",
        description="JSON file for this story (empty = use main / run model on image)",
        subtype='FILE_PATH',
        default="",
    )
    label: bpy.props.StringProperty(
        name="Label",
        default="",
    )


# ── Story count sync callback ────────────────────────────────────────

def _sync_story_items(self, context):
    """Keep fp3d_story_items length in sync with fp3d_num_stories."""
    items = context.scene.fp3d_story_items
    target = self.fp3d_num_stories
    while len(items) < target:
        item = items.add()
        idx = len(items) - 1
        item.label = f"Story {idx}" if idx > 0 else "Ground Floor"
    while len(items) > target:
        items.remove(len(items) - 1)


# ── Class registration list ──────────────────────────────────────────

classes = [
    FP3D_StoryItem,
    operators.FP3D_OT_GenerateSample,
    operators.FP3D_OT_GenerateModel,
    operators.FP3D_OT_AdjustWallHeight,
    operators.FP3D_OT_ExportModel,
    operators.FP3D_OT_LoadFloorPlan,
    operators.FP3D_OT_LoadStoryFloorPlan,
    operators.FP3D_OT_SuggestFurniture,
    operators.FP3D_OT_CritiqueLayout,
    operators.FP3D_OT_ModifyLayout,
    operators.FP3D_OT_RemoveFurniture,
    operators.FP3D_OT_GenerateExterior,
    operators.FP3D_OT_RemoveExterior,
    correction_ops.FP3D_OT_EnterCorrection,
    correction_ops.FP3D_OT_ConfirmCorrection,
    correction_ops.FP3D_OT_CancelCorrection,
    correction_ops.FP3D_OT_AddWall,
    correction_ops.FP3D_OT_AddDoor,
    correction_ops.FP3D_OT_AddWindow,
    correction_ops.FP3D_OT_AddFullWallWindow,
    correction_ops.FP3D_OT_DeleteElement,
    *preferences.classes,
    panels.FP3D_PT_MainPanel,
    panels.FP3D_PT_AdjustPanel,
    panels.FP3D_PT_ExportPanel,
    panels.FP3D_PT_ClaudePanel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    # Preferences overrides → env vars, then a non-blocking environment
    # probe so the panel can say what (if anything) still needs installing.
    try:
        preferences.apply_overrides()
    except Exception as e:  # noqa: BLE001 — never block registration
        print(f"[FP3D] preferences not applied: {e}")
    preferences.refresh_environment_async()

    bpy.types.Scene.fp3d_image_path = bpy.props.StringProperty(
        name="Floor Plan Image",
        description="Path to the floor plan image file",
        subtype='FILE_PATH',
        default="",
    )
    bpy.types.Scene.fp3d_wall_height = bpy.props.FloatProperty(
        name="Wall Height",
        description="Height of walls in meters",
        default=2.7,
        min=1.0,
        max=10.0,
        unit='LENGTH',
    )
    bpy.types.Scene.fp3d_model_type = bpy.props.EnumProperty(
        name="Model",
        description="Which AI model to use for floor plan parsing",
        items=[
            ('HYBRID', "Hybrid (recommended)", "YOLO reads the walls, doors, windows and room shapes; "
                                              "the local Qwen2.5-VL reads the printed room names"),
            ('YOLO', "YOLO only", "Object detection model — geometry only, heuristic room labels"),
            ('QWEN', "Qwen2.5-VL (trained)", "Fine-tuned Qwen2.5-VL-7B on synthetic floor plans — local, high accuracy, slow first run"),
            ('CLAUDE_VISION', "Premium Vision", "Premium AI vision — most accurate, reads labels & dimensions (requires API key)"),
        ],
        default='HYBRID',
    )
    bpy.types.Scene.fp3d_sensitivity = bpy.props.EnumProperty(
        name="Sensitivity",
        description="Detection sensitivity — higher catches more elements but may add noise",
        items=[
            ('LOW', "Low", "Conservative — only high-confidence detections"),
            ('MEDIUM', "Medium", "Balanced (original CubiCasa default)"),
            ('HIGH', "High", "Aggressive — catches more walls, doors, windows"),
            ('VERY_HIGH', "Very High", "Maximum recall — for faint or tricky floor plans"),
        ],
        default='VERY_HIGH',
    )
    bpy.types.Scene.fp3d_refine_with_claude = bpy.props.BoolProperty(
        name="Refine with Premium AI",
        description="After local model inference, use Premium AI to fix missing walls, "
                    "correct room labels, and improve accuracy (requires API key)",
        default=False,
    )
    bpy.types.Scene.fp3d_auto_scale = bpy.props.BoolProperty(
        name="Auto Scale",
        description="Read the dimension strings printed on the plan and set the scale "
                    "from them (Hybrid only). Scale Factor is then just the starting guess",
        default=True,
    )
    bpy.types.Scene.fp3d_scale_factor = bpy.props.FloatProperty(
        name="Scale Factor",
        description="Pixels per meter (auto-adjusts if too low for the image resolution)",
        default=50.0,
        min=1.0,
        max=2000.0,
    )
    bpy.types.Scene.fp3d_ai_backend = bpy.props.EnumProperty(
        name="AI Backend",
        description="Which AI backend to use for Premium features",
        items=[
            ('CLAUDE', "Premium Cloud", "Premium cloud AI — best quality, requires API key and credits"),
            ('LOCAL', "Local (Ollama/MLX)", "Local LLM — free, offline, requires a running model server"),
        ],
        default='CLAUDE',
    )
    bpy.types.Scene.fp3d_local_model = bpy.props.StringProperty(
        name="Model",
        description="Local model name (as shown by your server)",
        default="mlx-community/Qwen2.5-32B-Instruct-4bit",
    )
    bpy.types.Scene.fp3d_local_url = bpy.props.StringProperty(
        name="Server URL",
        description="Base URL of the local LLM server",
        default="http://localhost:8080",
    )
    bpy.types.Scene.fp3d_claude_api_key = bpy.props.StringProperty(
        name="API Key",
        description="API key for Premium AI features",
        subtype='PASSWORD',
        default="",
    )
    bpy.types.Scene.fp3d_claude_model = bpy.props.EnumProperty(
        name="Model",
        description="Which AI model to use for Premium features",
        items=[
            ('claude-sonnet-4-6', "Standard",
             "Best value — high quality for exterior, furniture, and layout (Recommended)"),
            ('claude-sonnet-4-5-20250929', "Standard (Previous Gen)",
             "Previous gen — still fast and cost-efficient"),
            ('claude-opus-4-6', "Advanced",
             "Most capable — better for complex layout modifications"),
        ],
        default='claude-sonnet-4-6',
    )
    bpy.types.Scene.fp3d_claude_prompt = bpy.props.StringProperty(
        name="Prompt",
        description="Natural language instruction for modifying the layout",
        default="",
    )
    bpy.types.Scene.fp3d_json_path = bpy.props.StringProperty(
        name="JSON Override",
        description="Optional: load geometry from a JSON file instead of running the model",
        subtype='FILE_PATH',
        default="",
    )
    bpy.types.Scene.fp3d_generate_ceiling = bpy.props.BoolProperty(
        name="Generate Ceiling",
        description="Generate ceiling planes for each room",
        default=False,
    )
    bpy.types.Scene.fp3d_num_stories = bpy.props.IntProperty(
        name="Stories",
        description="Number of stories to stack vertically",
        default=1,
        min=1,
        max=10,
        update=_sync_story_items,
    )
    bpy.types.Scene.fp3d_story_items = bpy.props.CollectionProperty(
        type=FP3D_StoryItem,
    )
    bpy.types.Scene.fp3d_active_story = bpy.props.IntProperty(
        name="Active Story",
        default=0,
        min=0,
    )
    # Staircase properties
    bpy.types.Scene.fp3d_stair_auto = bpy.props.BoolProperty(
        name="Auto-detect Stairs",
        description="Automatically place stairs where stairwell rooms are detected",
        default=True,
    )
    bpy.types.Scene.fp3d_stair_width = bpy.props.FloatProperty(
        name="Stair Width",
        description="Width of generated staircases in meters",
        default=1.0,
        min=0.5,
        max=3.0,
        unit='LENGTH',
    )
    bpy.types.Scene.fp3d_stair_position_x = bpy.props.FloatProperty(
        name="Stair X",
        description="X position for manual stair placement",
        default=0.0,
        unit='LENGTH',
    )
    bpy.types.Scene.fp3d_stair_position_y = bpy.props.FloatProperty(
        name="Stair Y",
        description="Y position for manual stair placement",
        default=0.0,
        unit='LENGTH',
    )
    bpy.types.Scene.fp3d_stair_direction = bpy.props.EnumProperty(
        name="Direction",
        description="Direction of stair run",
        items=[
            ('X_POS', "+X", "Stairs run in +X direction"),
            ('X_NEG', "-X", "Stairs run in -X direction"),
            ('Y_POS', "+Y", "Stairs run in +Y direction"),
            ('Y_NEG', "-Y", "Stairs run in -Y direction"),
        ],
        default='X_POS',
    )
    bpy.types.Scene.fp3d_exterior_style = bpy.props.StringProperty(
        name="Exterior Style",
        description="Architectural style for exterior generation (e.g. 'modern Korean house')",
        default="",
    )
    bpy.types.Scene.fp3d_reference_image = bpy.props.StringProperty(
        name="Reference Image",
        description="Optional reference photo for exterior style analysis (Premium AI vision)",
        subtype='FILE_PATH',
        default="",
    )
    bpy.types.Scene.fp3d_status = bpy.props.StringProperty(
        name="Status",
        default="Ready",
    )
    bpy.types.Scene.fp3d_correction_active = bpy.props.BoolProperty(
        name="Correction Active",
        description="Whether correction mode is currently active",
        default=False,
    )
    bpy.types.Scene.fp3d_correction_enabled = bpy.props.BoolProperty(
        name="Review Before 3D",
        description="Show 2D correction preview before generating 3D model",
        default=True,
    )


def unregister():
    # Stop the Qwen inference daemon (if one was started) so a 14 GB model
    # process doesn't outlive the add-on / Blender session.
    qwen_client.shutdown()

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.fp3d_image_path
    del bpy.types.Scene.fp3d_wall_height
    del bpy.types.Scene.fp3d_model_type
    del bpy.types.Scene.fp3d_sensitivity
    del bpy.types.Scene.fp3d_refine_with_claude
    del bpy.types.Scene.fp3d_scale_factor
    del bpy.types.Scene.fp3d_auto_scale
    del bpy.types.Scene.fp3d_ai_backend
    del bpy.types.Scene.fp3d_local_model
    del bpy.types.Scene.fp3d_local_url
    del bpy.types.Scene.fp3d_claude_api_key
    del bpy.types.Scene.fp3d_claude_model
    del bpy.types.Scene.fp3d_claude_prompt
    del bpy.types.Scene.fp3d_json_path
    del bpy.types.Scene.fp3d_generate_ceiling
    del bpy.types.Scene.fp3d_num_stories
    del bpy.types.Scene.fp3d_story_items
    del bpy.types.Scene.fp3d_active_story
    del bpy.types.Scene.fp3d_stair_auto
    del bpy.types.Scene.fp3d_stair_width
    del bpy.types.Scene.fp3d_stair_position_x
    del bpy.types.Scene.fp3d_stair_position_y
    del bpy.types.Scene.fp3d_stair_direction
    del bpy.types.Scene.fp3d_exterior_style
    del bpy.types.Scene.fp3d_reference_image
    del bpy.types.Scene.fp3d_status
    del bpy.types.Scene.fp3d_correction_active
    del bpy.types.Scene.fp3d_correction_enabled


if __name__ == "__main__":
    register()
