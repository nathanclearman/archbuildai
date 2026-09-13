import bpy

from . import preferences


def _draw_readiness(layout, model_type):
    """One-line setup hint when the chosen backend can't run yet."""
    env = preferences.ENV
    if not env["checked"]:
        return
    st = env["status"] or {}
    problems = []
    if model_type in ('HYBRID', 'YOLO') and env["cv_missing"]:
        problems.append("geometry packages not installed")
    if model_type in ('HYBRID', 'QWEN'):
        if not st.get("python"):
            problems.append("no Python with the VLM packages")
        elif not st.get("base_model_cached"):
            problems.append("base model not downloaded")
        if model_type == 'QWEN' and not st.get("adapter"):
            problems.append("fine-tuned adapter not found")
    if not problems:
        return
    box = layout.box()
    col = box.column()
    col.alert = True
    col.label(text="Setup needed: " + "; ".join(problems), icon='ERROR')
    box.label(text="Edit > Preferences > Add-ons > ArchbuildAI", icon='PREFERENCES')
    if model_type == 'HYBRID' and st.get("python") is None and not env["cv_missing"]:
        box.label(text="(Hybrid will still build geometry; labels fall back to YOLO)", icon='INFO')


class FP3D_PT_MainPanel(bpy.types.Panel):
    bl_label = "ArchbuildAI"
    bl_idname = "FP3D_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ArchbuildAI"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.label(text="Input", icon='IMAGE_DATA')
        num_stories = scene.fp3d_num_stories
        if num_stories <= 1:
            layout.prop(scene, "fp3d_image_path", text="Floor Plan")
        else:
            layout.prop(scene, "fp3d_image_path", text="Ground Floor")
        layout.prop(scene, "fp3d_json_path", text="JSON Override")

        layout.separator()

        # -- Correction mode UI --
        if scene.fp3d_correction_active:
            box = layout.box()
            box.label(text="Correction Mode", icon='GREASEPENCIL')

            row = box.row(align=True)
            row.scale_y = 1.5
            row.operator("fp3d.confirm_correction", icon='CHECKMARK')

            row = box.row(align=True)
            row.operator("fp3d.cancel_correction", icon='CANCEL')

            box.separator()
            box.label(text="Add Elements:", icon='ADD')
            row = box.row(align=True)
            row.operator("fp3d.correction_add_wall", icon='MOD_SOLIDIFY', text="Wall")
            row.operator("fp3d.correction_add_door", icon='OBJECT_DATA', text="Door")
            row.operator("fp3d.correction_add_window", icon='WINDOW', text="Window")
            row = box.row(align=True)
            row.operator("fp3d.correction_add_full_window", icon='MOD_LATTICE', text="Full-Wall Window")

            row = box.row(align=True)
            row.operator("fp3d.correction_delete", icon='TRASH', text="Delete Selected")

            box.separator()
            col = box.column()
            col.scale_y = 0.7
            col.label(text="G = Move selected point", icon='BLANK1')
            col.label(text="X = Delete selected", icon='BLANK1')
            col.label(text="A = Select all", icon='BLANK1')

            if scene.fp3d_status and scene.fp3d_status != "Ready":
                box = layout.box()
                box.label(text=scene.fp3d_status, icon='INFO')
            return

        # -- Normal mode UI --
        layout.label(text="Settings", icon='PREFERENCES')
        layout.prop(scene, "fp3d_model_type")
        _draw_readiness(layout, scene.fp3d_model_type)
        if scene.fp3d_model_type == 'CLAUDE_VISION':
            box = layout.box()
            box.label(text="Premium feature (requires API key below)", icon='URL')
            api_key = getattr(scene, "fp3d_claude_api_key", "")
            if not api_key or not api_key.strip():
                col = box.column()
                col.alert = True
                col.label(text="Set API key in Premium AI panel", icon='ERROR')
        else:
            layout.prop(scene, "fp3d_sensitivity")
            # Show refine option when local model is selected and API key exists
            api_key = getattr(scene, "fp3d_claude_api_key", "")
            if api_key and api_key.strip():
                layout.prop(scene, "fp3d_refine_with_claude", text="Refine with Premium AI")
        row = layout.row(align=True)
        row.prop(scene, "fp3d_scale_factor")
        if scene.fp3d_model_type == 'HYBRID':
            row.prop(scene, "fp3d_auto_scale", text="Auto", toggle=True)
        layout.prop(scene, "fp3d_wall_height")
        layout.prop(scene, "fp3d_generate_ceiling")
        layout.prop(scene, "fp3d_num_stories")

        # Per-story floor plan inputs (only when multi-story)
        if num_stories > 1:
            box = layout.box()
            box.label(text="Upper Story Floor Plans", icon='RENDERLAYERS')
            items = scene.fp3d_story_items
            missing_count = 0
            # Story 0 (Ground Floor) uses the main "Floor Plan" input above,
            # so only show pickers for Story 1+.
            for i in range(1, min(len(items), num_stories)):
                item = items[i]
                row = box.row(align=True)
                label = item.label or f"Story {i}"
                has_image = bool(item.image_path and item.image_path.strip())
                has_json = bool(item.json_path and item.json_path.strip())
                if not has_image and not has_json:
                    missing_count += 1
                    row.alert = True
                row.label(text=f"{label}:")
                sub = row.row(align=True)
                sub.prop(item, "image_path", text="")
                op = sub.operator("fp3d.load_story_floor_plan",
                                  text="", icon='FILEBROWSER')
                op.story_index = i
            if missing_count > 0:
                col = box.column()
                col.alert = True
                col.scale_y = 0.8
                col.label(
                    text=f"⚠ {missing_count} upper floor(s) need an image!",
                    icon='ERROR')
            else:
                col = box.column()
                col.scale_y = 0.7
                col.label(text="✓ All floors have images", icon='CHECKMARK')

            # Staircase configuration
            box = layout.box()
            box.label(text="Staircases", icon='SORT_ASC')
            box.prop(scene, "fp3d_stair_auto")
            if not scene.fp3d_stair_auto:
                row = box.row(align=True)
                row.prop(scene, "fp3d_stair_position_x", text="X")
                row.prop(scene, "fp3d_stair_position_y", text="Y")
                box.prop(scene, "fp3d_stair_direction")
            box.prop(scene, "fp3d_stair_width")

        layout.prop(scene, "fp3d_correction_enabled")

        layout.separator()
        row = layout.row(align=True)
        row.scale_y = 1.5
        row.operator("fp3d.generate_model", icon='MOD_BUILD')

        row = layout.row(align=True)
        row.operator("fp3d.generate_sample", icon='MESH_CUBE', text="Generate Sample")

        if scene.fp3d_status and scene.fp3d_status != "Ready":
            box = layout.box()
            status = scene.fp3d_status
            icon = 'INFO'
            if status.startswith("Error"):
                icon = 'ERROR'
            elif status.startswith("Done") or status.startswith("Sample"):
                icon = 'CHECKMARK'
            box.label(text=status, icon=icon)


class FP3D_PT_AdjustPanel(bpy.types.Panel):
    bl_label = "Adjust"
    bl_idname = "FP3D_PT_adjust"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ArchbuildAI"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.prop(scene, "fp3d_wall_height")
        layout.operator("fp3d.adjust_wall_height", icon='ARROW_LEFTRIGHT')


class FP3D_PT_ExportPanel(bpy.types.Panel):
    bl_label = "Export"
    bl_idname = "FP3D_PT_export"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ArchbuildAI"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.operator("fp3d.export_model", icon='EXPORT')


class FP3D_PT_ClaudePanel(bpy.types.Panel):
    bl_label = "Premium AI"
    bl_idname = "FP3D_PT_claude"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ArchbuildAI"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        backend = getattr(scene, "fp3d_ai_backend", "CLAUDE")

        # Backend selector
        layout.label(text="Backend", icon='PREFERENCES')
        layout.prop(scene, "fp3d_ai_backend", text="")

        layout.separator()

        if backend == "LOCAL":
            # Local LLM configuration
            layout.label(text="Local LLM Settings", icon='SETTINGS')
            layout.prop(scene, "fp3d_local_model")
            layout.prop(scene, "fp3d_local_url")

            box = layout.box()
            col = box.column(align=True)
            col.scale_y = 0.75
            col.label(text="Start your model server first:", icon='INFO')
            col.label(text="  mlx_lm.server --model <name> --port 8080")
            col.label(text="  or: ollama serve")
        else:
            # Premium API configuration
            layout.label(text="Premium API Settings", icon='SETTINGS')
            layout.prop(scene, "fp3d_claude_api_key")
            layout.prop(scene, "fp3d_claude_model")

            if not scene.fp3d_claude_api_key:
                box = layout.box()
                box.label(text="Enter your API key to unlock", icon='INFO')
                box.label(text="Premium AI features below.")
                return

        layout.separator()

        # Furniture placement
        layout.label(text="Furniture", icon='MESH_CUBE')
        row = layout.row(align=True)
        row.scale_y = 1.3
        row.operator("fp3d.suggest_furniture", icon='OUTLINER_OB_MESH')
        row = layout.row(align=True)
        row.operator("fp3d.remove_furniture", icon='TRASH', text="Remove Furniture")

        layout.separator()

        # Exterior generation
        layout.label(text="Exterior", icon='HOME')
        layout.prop(scene, "fp3d_exterior_style", text="", icon='CONSOLE')
        layout.prop(scene, "fp3d_reference_image", text="Reference Photo")
        row = layout.row(align=True)
        row.scale_y = 1.3
        row.operator("fp3d.generate_exterior", icon='MESH_CONE')
        row = layout.row(align=True)
        row.operator("fp3d.remove_exterior", icon='TRASH', text="Remove Exterior")
        col = layout.column()
        col.scale_y = 0.7
        col.label(text="e.g. 'modern minimalist'  → flat w/ parapet", icon='BLANK1')
        col.label(text="e.g. 'Scandinavian cabin' → shed roof", icon='BLANK1')
        col.label(text="e.g. 'Mediterranean villa' → hip clay tile", icon='BLANK1')

        layout.separator()

        # Layout analysis
        layout.label(text="Layout Analysis", icon='VIEWZOOM')
        row = layout.row(align=True)
        row.scale_y = 1.3
        row.operator("fp3d.critique_layout", icon='GREASEPENCIL')

        layout.separator()

        # Natural language modification
        layout.label(text="Modify Layout", icon='OUTLINER_DATA_GP_LAYER')
        layout.prop(scene, "fp3d_claude_prompt", text="", icon='CONSOLE')
        row = layout.row(align=True)
        row.scale_y = 1.3
        row.operator("fp3d.modify_layout", icon='FILE_REFRESH')
        col = layout.column()
        col.scale_y = 0.7
        col.label(text="e.g. 'make the kitchen bigger'", icon='BLANK1')
        col.label(text="e.g. 'add a bathroom near bedroom'", icon='BLANK1')
