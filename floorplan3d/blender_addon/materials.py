"""Default material assignment for generated geometry."""

import bpy


def _get_or_create_material(name, color):
    """Get existing material by name or create a new one with the given RGBA color."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = color
    return mat


def assign_wall_material(obj):
    """Assign a default wall material (off-white)."""
    mat = _get_or_create_material("FP3D_Wall", (0.92, 0.90, 0.87, 1.0))
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def assign_floor_material(obj):
    """Assign a default floor material (light wood tone)."""
    mat = _get_or_create_material("FP3D_Floor", (0.76, 0.60, 0.42, 1.0))
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def assign_ceiling_material(obj):
    """Assign a default ceiling material (white)."""
    mat = _get_or_create_material("FP3D_Ceiling", (0.95, 0.95, 0.95, 1.0))
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
