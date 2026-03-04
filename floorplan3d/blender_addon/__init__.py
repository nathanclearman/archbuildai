bl_info = {
    "name": "FloorPlan3D",
    "author": "FloorPlan3D Team",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > FloorPlan3D",
    "description": "Generate 3D architectural models from 2D floor plan images",
    "category": "3D View",
}

import bpy
from . import operators, panels


classes = [
    operators.FP3D_OT_GenerateSample,
    operators.FP3D_OT_GenerateModel,
    operators.FP3D_OT_AdjustWallHeight,
    operators.FP3D_OT_ExportModel,
    operators.FP3D_OT_LoadFloorPlan,
    panels.FP3D_PT_MainPanel,
    panels.FP3D_PT_AdjustPanel,
    panels.FP3D_PT_ExportPanel,
    panels.FP3D_PT_ClaudePanel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

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
    bpy.types.Scene.fp3d_scale_factor = bpy.props.FloatProperty(
        name="Scale Factor",
        description="Pixels per meter in the floor plan image",
        default=50.0,
        min=1.0,
        max=500.0,
    )
    bpy.types.Scene.fp3d_claude_api_key = bpy.props.StringProperty(
        name="Claude API Key",
        description="Optional: Anthropic API key for smart features",
        subtype='PASSWORD',
        default="",
    )
    bpy.types.Scene.fp3d_claude_prompt = bpy.props.StringProperty(
        name="Prompt",
        description="Natural language instruction for Claude",
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
    bpy.types.Scene.fp3d_status = bpy.props.StringProperty(
        name="Status",
        default="Ready",
    )


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.fp3d_image_path
    del bpy.types.Scene.fp3d_wall_height
    del bpy.types.Scene.fp3d_scale_factor
    del bpy.types.Scene.fp3d_claude_api_key
    del bpy.types.Scene.fp3d_claude_prompt
    del bpy.types.Scene.fp3d_json_path
    del bpy.types.Scene.fp3d_generate_ceiling
    del bpy.types.Scene.fp3d_status


if __name__ == "__main__":
    register()
