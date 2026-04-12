import bpy


class FP3D_PT_MainPanel(bpy.types.Panel):
    bl_label = "FloorPlan3D"
    bl_idname = "FP3D_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "FloorPlan3D"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.label(text="Input", icon='IMAGE_DATA')
        layout.prop(scene, "fp3d_image_path", text="Floor Plan")
        layout.prop(scene, "fp3d_json_path", text="JSON Override")

        layout.separator()
        layout.label(text="Settings", icon='PREFERENCES')
        layout.prop(scene, "fp3d_scale_factor")
        layout.prop(scene, "fp3d_wall_height")
        layout.prop(scene, "fp3d_generate_ceiling")

        layout.separator()
        layout.label(text="Pipeline", icon='NODETREE')
        layout.prop(scene, "fp3d_cv_only")
        layout.prop(scene, "fp3d_use_refiner")

        layout.separator()
        row = layout.row(align=True)
        row.scale_y = 1.5
        row.operator("fp3d.generate_model", icon='MOD_BUILD')

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
    bl_category = "FloorPlan3D"
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
    bl_category = "FloorPlan3D"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.operator("fp3d.export_model", icon='EXPORT')


class FP3D_PT_ClaudePanel(bpy.types.Panel):
    bl_label = "AI Assistant (Optional)"
    bl_idname = "FP3D_PT_claude"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "FloorPlan3D"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.prop(scene, "fp3d_claude_api_key")
        layout.prop(scene, "fp3d_claude_prompt", text="")
        layout.label(text="(Phase 3 feature)", icon='TIME')
