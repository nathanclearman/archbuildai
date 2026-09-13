"""Default material assignment for generated geometry.

Includes procedural texture generation for exterior materials
(wood grain, brick patterns, stone variation, metal finishes, etc.)
"""

import bpy
from math import pi



# Per-room-type floor colors
FLOOR_COLORS = {
    "kitchen":      (0.82, 0.71, 0.55, 1.0),   # warm tile
    "living_room":  (0.76, 0.60, 0.42, 1.0),   # wood
    "bedroom":      (0.70, 0.65, 0.60, 1.0),   # carpet grey
    "bath":         (0.78, 0.82, 0.85, 1.0),   # light tile
    "bathroom":     (0.78, 0.82, 0.85, 1.0),
    "hallway":      (0.68, 0.62, 0.55, 1.0),   # darker wood
    "corridor":     (0.68, 0.62, 0.55, 1.0),
    "closet":       (0.72, 0.68, 0.64, 1.0),   # neutral
    "other":        (0.65, 0.60, 0.55, 1.0),   # muted neutral
}
DEFAULT_FLOOR_COLOR = (0.76, 0.60, 0.42, 1.0)  # fallback wood

# Furniture material colors
FURNITURE_COLORS = {
    "bed":           (0.55, 0.55, 0.65, 1.0),  # blue-grey
    "double_bed":    (0.55, 0.55, 0.65, 1.0),
    "single_bed":    (0.55, 0.55, 0.65, 1.0),
    "sofa":          (0.50, 0.38, 0.28, 1.0),  # leather brown
    "couch":         (0.50, 0.38, 0.28, 1.0),
    "armchair":      (0.50, 0.38, 0.28, 1.0),
    "table":         (0.55, 0.45, 0.35, 1.0),  # wood
    "dining_table":  (0.55, 0.45, 0.35, 1.0),
    "coffee_table":  (0.55, 0.45, 0.35, 1.0),
    "desk":          (0.55, 0.45, 0.35, 1.0),
    "nightstand":    (0.55, 0.45, 0.35, 1.0),
    "wardrobe":      (0.45, 0.38, 0.30, 1.0),  # dark wood
    "dresser":       (0.45, 0.38, 0.30, 1.0),
    "bookshelf":     (0.45, 0.38, 0.30, 1.0),
    "cabinet":       (0.45, 0.38, 0.30, 1.0),
    "chair":         (0.60, 0.55, 0.45, 1.0),  # light wood
    "dining_chair":  (0.60, 0.55, 0.45, 1.0),
    "stool":         (0.60, 0.55, 0.45, 1.0),
    "refrigerator":  (0.85, 0.85, 0.88, 1.0),  # stainless
    "fridge":        (0.85, 0.85, 0.88, 1.0),
    "oven":          (0.30, 0.30, 0.32, 1.0),  # dark appliance
    "stove":         (0.30, 0.30, 0.32, 1.0),
    "sink":          (0.80, 0.82, 0.85, 1.0),  # porcelain
    "toilet":        (0.92, 0.92, 0.92, 1.0),  # white
    "bathtub":       (0.90, 0.90, 0.92, 1.0),  # white
    "shower":        (0.75, 0.78, 0.82, 1.0),  # glass-ish
    "washing_machine": (0.85, 0.85, 0.88, 1.0),
    "tv":            (0.15, 0.15, 0.18, 1.0),  # black
    "television":    (0.15, 0.15, 0.18, 1.0),
}
DEFAULT_FURNITURE_COLOR = (0.55, 0.45, 0.35, 1.0)  # generic wood


# ---------------------------------------------------------------------------
# PBR presets — roughness / metallic / specular per material type
# ---------------------------------------------------------------------------
_PBR_PRESETS = {
    # Glass
    "glass":          {"roughness": 0.05, "metallic": 0.0,  "specular": 1.0,  "alpha": 0.3},
    # Metals
    "metal":          {"roughness": 0.50, "metallic": 0.4,  "specular": 0.5},
    "black_metal":    {"roughness": 0.25, "metallic": 0.9,  "specular": 0.9},
    "zinc":           {"roughness": 0.35, "metallic": 0.7,  "specular": 0.7},
    "copper":         {"roughness": 0.30, "metallic": 0.8,  "specular": 0.8},
    "corten":         {"roughness": 0.70, "metallic": 0.4,  "specular": 0.3},
    # Masonry / mineral
    "concrete":       {"roughness": 0.85, "metallic": 0.0,  "specular": 0.1},
    "white_concrete": {"roughness": 0.80, "metallic": 0.0,  "specular": 0.15},
    "stone":          {"roughness": 0.75, "metallic": 0.0,  "specular": 0.15},
    "brick":          {"roughness": 0.80, "metallic": 0.0,  "specular": 0.1},
    "white_stucco":   {"roughness": 0.70, "metallic": 0.0,  "specular": 0.1},
    "terracotta":     {"roughness": 0.65, "metallic": 0.0,  "specular": 0.15},
    # Wood
    "dark_wood":      {"roughness": 0.55, "metallic": 0.0,  "specular": 0.25},
    "light_wood":     {"roughness": 0.50, "metallic": 0.0,  "specular": 0.30},
    # Roof tiles
    "clay_tile":      {"roughness": 0.60, "metallic": 0.0,  "specular": 0.18},
    "slate":          {"roughness": 0.45, "metallic": 0.05, "specular": 0.35},
    "shingle":        {"roughness": 0.75, "metallic": 0.0,  "specular": 0.1},
    "thatch":         {"roughness": 0.90, "metallic": 0.0,  "specular": 0.05},
    # Other
    "charcoal":       {"roughness": 0.40, "metallic": 0.1,  "specular": 0.2},
    "greenery":       {"roughness": 0.90, "metallic": 0.0,  "specular": 0.05},
}


def _get_or_create_material(name, color, material_key=None):
    """Get or create a material and ALWAYS update its color and PBR settings.

    Args:
        name: Material name for Blender.
        color: (R, G, B, A) tuple.
        material_key: Optional key into _PBR_PRESETS for roughness/metallic/specular.
    """
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True

    # Always update color and PBR — ensures regeneration picks up new palettes
    bsdf = mat.node_tree.nodes.get("Principled BSDF") if mat.use_nodes else None
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color

        preset = _PBR_PRESETS.get(material_key, {}) if material_key else {}
        if "roughness" in preset:
            bsdf.inputs["Roughness"].default_value = preset["roughness"]
        if "metallic" in preset:
            bsdf.inputs["Metallic"].default_value = preset["metallic"]
        if "specular" in preset:
            spec_input = (bsdf.inputs.get("Specular IOR Level")
                          or bsdf.inputs.get("Specular"))
            if spec_input:
                spec_input.default_value = preset["specular"]
        if "alpha" in preset:
            bsdf.inputs["Alpha"].default_value = preset["alpha"]
            # Enable transparency — API varies by Blender version
            if hasattr(mat, "blend_method"):
                # Blender 3.x / 4.x EEVEE Legacy
                mat.blend_method = "BLEND"
            # Blender 4.2+ / 5.x: transparency is automatic when Alpha < 1
            mat.use_backface_culling = True

    mat.diffuse_color = color  # viewport Solid mode display
    return mat


def _apply_mat(obj, mat):
    """Apply a material to an object (replace slot 0 or append)."""
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def assign_wall_material(obj):
    """Assign a default wall material (off-white)."""
    _apply_mat(obj, _get_or_create_material("FP3D_Wall", (0.92, 0.90, 0.87, 1.0)))


def assign_floor_material(obj, room_label=None):
    """Assign a floor material based on room type."""
    color = DEFAULT_FLOOR_COLOR
    mat_name = "FP3D_Floor"
    if room_label:
        # Normalize label
        key = room_label.lower().replace(" ", "_")
        # Try exact match, then partial
        if key in FLOOR_COLORS:
            color = FLOOR_COLORS[key]
            mat_name = f"FP3D_Floor_{key}"
        else:
            # Try matching prefix
            for k, v in FLOOR_COLORS.items():
                if k in key or key in k:
                    color = v
                    mat_name = f"FP3D_Floor_{k}"
                    break

    _apply_mat(obj, _get_or_create_material(mat_name, color))


def assign_ceiling_material(obj):
    """Assign a default ceiling material (white)."""
    _apply_mat(obj, _get_or_create_material("FP3D_Ceiling", (0.95, 0.95, 0.95, 1.0)))


def assign_stair_material(obj):
    """Assign a concrete-grey material for staircases."""
    _apply_mat(obj, _get_or_create_material("FP3D_Staircase", (0.70, 0.68, 0.65, 1.0)))


def assign_window_glass_material(obj):
    """Assign a translucent glass material for window panes."""
    _apply_mat(obj, _get_or_create_material(
        "FP3D_WindowGlass", (0.70, 0.78, 0.85, 1.0), material_key="glass"))


# Door panel colors by type
DOOR_COLORS = {
    "hinged":   (0.35, 0.24, 0.16, 1.0),   # dark wood
    "sliding":  (0.40, 0.28, 0.18, 1.0),   # medium wood
    "pocket":   (0.40, 0.28, 0.18, 1.0),
    "double":   (0.35, 0.24, 0.16, 1.0),
    "french":   (0.85, 0.87, 0.90, 1.0),   # white-painted with glass look
    "glass":    (0.70, 0.78, 0.85, 1.0),   # tinted glass
    "barn":     (0.32, 0.22, 0.14, 1.0),   # rustic dark wood
    "folding":  (0.85, 0.83, 0.80, 1.0),   # painted white
}
DEFAULT_DOOR_COLOR = (0.35, 0.24, 0.16, 1.0)  # dark wood default


def assign_door_material(obj, door_type=None):
    """Assign a material to a door panel based on its type."""
    color = DEFAULT_DOOR_COLOR
    mat_name = "FP3D_Door"
    pbr_key = "dark_wood"

    if door_type:
        key = door_type.lower().replace(" ", "_")
        if key in DOOR_COLORS:
            color = DOOR_COLORS[key]
            mat_name = f"FP3D_Door_{key}"
            if key in ("glass",):
                pbr_key = "glass"
            elif key in ("french", "folding"):
                pbr_key = None
            else:
                pbr_key = "dark_wood"

    _apply_mat(obj, _get_or_create_material(mat_name, color, material_key=pbr_key))


def assign_furniture_material(obj, furniture_name=None):
    """Assign a material to a furniture object based on its type."""
    color = DEFAULT_FURNITURE_COLOR
    mat_name = "FP3D_Furniture"
    if furniture_name:
        key = furniture_name.lower().replace(" ", "_")
        if key in FURNITURE_COLORS:
            color = FURNITURE_COLORS[key]
            mat_name = f"FP3D_Furn_{key}"
        else:
            for k, v in FURNITURE_COLORS.items():
                if k in key or key in k:
                    color = v
                    mat_name = f"FP3D_Furn_{k}"
                    break

    _apply_mat(obj, _get_or_create_material(mat_name, color))


# ---------------------------------------------------------------------------
# Exterior material colors — calibrated to real-world architectural materials
# Colors are in linear sRGB (Blender's working space).
# ---------------------------------------------------------------------------
EXTERIOR_COLORS = {
    # Roofing
    "clay_tile":      (0.56, 0.22, 0.10, 1.0),   # natural terracotta clay
    "slate":          (0.28, 0.30, 0.35, 1.0),   # Welsh/Vermont slate grey-blue
    "metal":          (0.52, 0.54, 0.56, 1.0),   # standing-seam zinc/steel (medium grey)
    "shingle":        (0.22, 0.20, 0.17, 1.0),   # aged cedar shingle
    "thatch":         (0.60, 0.52, 0.34, 1.0),   # dried reed/straw
    # Masonry
    "concrete":       (0.58, 0.56, 0.53, 1.0),   # poured/board-formed concrete
    "white_stucco":   (0.92, 0.90, 0.86, 1.0),   # lime render / white stucco
    "warm_stucco":    (0.92, 0.82, 0.58, 1.0),   # Mediterranean warm yellow/ochre stucco
    "stone":          (0.55, 0.50, 0.42, 1.0),    # natural limestone/sandstone
    "brick":          (0.55, 0.20, 0.12, 1.0),    # traditional red facing brick
    "white_concrete": (0.85, 0.84, 0.82, 1.0),    # white-pigment architectural concrete
    "terracotta":     (0.65, 0.33, 0.17, 1.0),    # fired clay terracotta
    # Wood
    "dark_wood":      (0.35, 0.23, 0.13, 1.0),    # stained cedar/walnut cladding
    "light_wood":     (0.58, 0.45, 0.28, 1.0),    # natural pine/birch/larch
    # Glass
    "glass":          (0.65, 0.75, 0.82, 1.0),    # low-e tinted architectural glass
    # Metal finishes
    "copper":         (0.62, 0.38, 0.15, 1.0),    # new/polished copper
    "corten":         (0.45, 0.22, 0.10, 1.0),    # weathered Cor-Ten patina
    "charcoal":       (0.16, 0.16, 0.18, 1.0),    # anthracite powder-coat
    "black_metal":    (0.08, 0.08, 0.10, 1.0),    # matte black powder-coat steel
    "zinc":           (0.55, 0.58, 0.60, 1.0),    # pre-weathered zinc
    # Other
    "greenery":       (0.28, 0.48, 0.22, 1.0),    # living wall / planter foliage
}
DEFAULT_EXTERIOR_COLOR = (0.58, 0.56, 0.53, 1.0)

# Secondary color used for texture variation (grain highlights, mortar, etc.)
_EXTERIOR_SECONDARY = {
    "clay_tile":      (0.68, 0.30, 0.15, 1.0),
    "slate":          (0.35, 0.37, 0.42, 1.0),
    "metal":          (0.58, 0.60, 0.63, 1.0),
    "shingle":        (0.30, 0.26, 0.20, 1.0),
    "thatch":         (0.72, 0.62, 0.40, 1.0),
    "concrete":       (0.62, 0.60, 0.57, 1.0),
    "white_stucco":   (0.88, 0.86, 0.82, 1.0),
    "warm_stucco":    (0.88, 0.78, 0.52, 1.0),
    "stone":          (0.48, 0.44, 0.36, 1.0),
    "brick":          (0.70, 0.68, 0.62, 1.0),     # mortar
    "white_concrete": (0.80, 0.79, 0.77, 1.0),
    "terracotta":     (0.75, 0.42, 0.24, 1.0),
    "dark_wood":      (0.48, 0.34, 0.20, 1.0),     # grain highlight
    "light_wood":     (0.68, 0.55, 0.35, 1.0),
    "copper":         (0.50, 0.32, 0.12, 1.0),
    "corten":         (0.55, 0.28, 0.14, 1.0),
    "charcoal":       (0.20, 0.20, 0.22, 1.0),
    "black_metal":    (0.12, 0.12, 0.14, 1.0),
    "zinc":           (0.60, 0.62, 0.65, 1.0),
    "greenery":       (0.35, 0.55, 0.28, 1.0),
}


# ---------------------------------------------------------------------------
# Procedural texture node builders
# ---------------------------------------------------------------------------

def _add_node(tree, node_type, x, y, **settings):
    """Add a shader node at position (x, y) with optional settings."""
    node = tree.nodes.new(node_type)
    node.location = (x, y)
    for k, v in settings.items():
        setattr(node, k, v)
    return node


def _add_mix_rgb(tree, x, y, blend_type="MIX", fac=0.5):
    """Add a color mix node, compatible with Blender 3.x, 4.x, and 5.x.

    Returns (node, fac_input_idx, color1_input_idx, color2_input_idx, output_idx).
    All indices are integers for reliable access across Blender versions — the
    ShaderNodeMix node in Blender 4+/5.x has multiple inputs named "A"/"B"
    for different data types (Float/Vector/RGBA), and name-based access like
    node.inputs["A"] returns the FIRST match (Float), not the Color input.
    """
    try:
        # Blender 4.0+: ShaderNodeMix replaces ShaderNodeMixRGB
        node = tree.nodes.new("ShaderNodeMix")
        node.data_type = "RGBA"
        node.blend_type = blend_type
        node.location = (x, y)

        # Find inputs by socket type, not name — bullet-proof for all versions.
        # In RGBA mode we need: Factor (FLOAT), Color A (RGBA), Color B (RGBA).
        fac_idx = None
        color_indices = []
        for i, inp in enumerate(node.inputs):
            sock_type = getattr(inp, "type", "")
            if sock_type == "VALUE" and fac_idx is None:
                fac_idx = i
                inp.default_value = fac
            elif sock_type == "RGBA":
                color_indices.append(i)

        if fac_idx is not None and len(color_indices) >= 2:
            # Find output: first RGBA output
            out_idx = 0
            for i, out in enumerate(node.outputs):
                if getattr(out, "type", "") == "RGBA":
                    out_idx = i
                    break
            return node, fac_idx, color_indices[0], color_indices[1], out_idx

        # Fallback: inputs are dynamic (filtered by data_type), use indices 0/1/2
        # This handles Blender versions where only the active-type inputs appear.
        node.inputs[0].default_value = fac
        return node, 0, 1, 2, 0

    except Exception:
        # Blender 3.x fallback: ShaderNodeMixRGB
        node = tree.nodes.new("ShaderNodeMixRGB")
        node.blend_type = blend_type
        node.location = (x, y)
        node.inputs["Fac"].default_value = fac
        return node, "Fac", "Color1", "Color2", "Color"


def _clear_extra_nodes(tree):
    """Remove all nodes except the Principled BSDF and Material Output."""
    for node in list(tree.nodes):
        if node.bl_idname not in ("ShaderNodeBsdfPrincipled", "ShaderNodeOutputMaterial"):
            tree.nodes.remove(node)


def _setup_tex_coords(tree, x=-800, y=0, scale=(1, 1, 1)):
    """Add TexCoord → Mapping chain, return the mapping node."""
    tc = _add_node(tree, "ShaderNodeTexCoord", x, y)
    mapping = _add_node(tree, "ShaderNodeMapping", x + 180, y)
    mapping.inputs["Scale"].default_value = scale
    tree.links.new(tc.outputs["Object"], mapping.inputs["Vector"])
    return mapping


def _setup_wood_texture(tree, bsdf, base_color, secondary, scale=8.0):
    """Procedural wood grain via Wave + Noise distortion."""
    mapping = _setup_tex_coords(tree, scale=(scale, scale, scale))

    # Noise for grain distortion
    noise = _add_node(tree, "ShaderNodeTexNoise", -400, 100)
    noise.inputs["Scale"].default_value = 18.0
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Roughness"].default_value = 0.7
    tree.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

    # Wave texture for wood bands
    wave = _add_node(tree, "ShaderNodeTexWave", -400, -100)
    wave.wave_type = "BANDS"
    wave.bands_direction = "Y"
    wave.inputs["Scale"].default_value = 3.0
    wave.inputs["Distortion"].default_value = 4.5
    wave.inputs["Detail"].default_value = 3.0
    wave.inputs["Detail Scale"].default_value = 1.5
    tree.links.new(mapping.outputs["Vector"], wave.inputs["Vector"])

    # Mix noise into wave for natural variation
    mix_vec, fac_in, c1_in, c2_in, c_out = _add_mix_rgb(tree, -200, 0, "MIX", 0.3)
    tree.links.new(wave.outputs["Fac"], mix_vec.inputs[c1_in])
    tree.links.new(noise.outputs["Fac"], mix_vec.inputs[c2_in])

    # Color ramp: map factor to wood colors
    ramp = _add_node(tree, "ShaderNodeValToRGB", 0, 0)
    ramp.color_ramp.elements[0].position = 0.3
    ramp.color_ramp.elements[0].color = base_color
    ramp.color_ramp.elements[1].position = 0.7
    ramp.color_ramp.elements[1].color = secondary
    tree.links.new(mix_vec.outputs[c_out], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    # Roughness variation from grain
    rough_ramp = _add_node(tree, "ShaderNodeValToRGB", 0, -200)
    rough_ramp.color_ramp.elements[0].position = 0.3
    rough_ramp.color_ramp.elements[0].color = (0.45, 0.45, 0.45, 1.0)
    rough_ramp.color_ramp.elements[1].position = 0.7
    rough_ramp.color_ramp.elements[1].color = (0.65, 0.65, 0.65, 1.0)
    tree.links.new(mix_vec.outputs[c_out], rough_ramp.inputs["Fac"])
    tree.links.new(rough_ramp.outputs["Color"], bsdf.inputs["Roughness"])

    # Bump for grain depth
    bump = _add_node(tree, "ShaderNodeBump", 200, -300)
    bump.inputs["Strength"].default_value = 0.15
    tree.links.new(mix_vec.outputs[c_out], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def _setup_brick_texture(tree, bsdf, base_color, mortar_color, scale=5.0):
    """Procedural brick pattern using Blender's Brick Texture node."""
    mapping = _setup_tex_coords(tree, scale=(scale, scale, scale))

    brick = _add_node(tree, "ShaderNodeTexBrick", -400, 0)
    brick.inputs["Color1"].default_value = base_color
    brick.inputs["Color2"].default_value = (
        base_color[0] * 0.85, base_color[1] * 0.90,
        base_color[2] * 0.88, 1.0
    )
    brick.inputs["Mortar"].default_value = mortar_color
    brick.inputs["Scale"].default_value = scale
    brick.inputs["Mortar Size"].default_value = 0.02
    brick.inputs["Mortar Smooth"].default_value = 0.1
    brick.inputs["Bias"].default_value = 0.0
    brick.inputs["Brick Width"].default_value = 0.5
    brick.inputs["Row Height"].default_value = 0.25
    tree.links.new(mapping.outputs["Vector"], brick.inputs["Vector"])

    # Add subtle noise variation to brick color
    noise = _add_node(tree, "ShaderNodeTexNoise", -400, -200)
    noise.inputs["Scale"].default_value = 30.0
    noise.inputs["Detail"].default_value = 4.0
    tree.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

    mix, fac_in, c1_in, c2_in, c_out = _add_mix_rgb(tree, -100, 0, "OVERLAY", 0.15)
    tree.links.new(brick.outputs["Color"], mix.inputs[c1_in])
    tree.links.new(noise.outputs["Color"], mix.inputs[c2_in])
    tree.links.new(mix.outputs[c_out], bsdf.inputs["Base Color"])

    # Roughness: mortar is rougher than brick face
    rough_ramp = _add_node(tree, "ShaderNodeValToRGB", -100, -200)
    rough_ramp.color_ramp.elements[0].position = 0.0
    rough_ramp.color_ramp.elements[0].color = (0.70, 0.70, 0.70, 1.0)  # brick face
    rough_ramp.color_ramp.elements[1].position = 1.0
    rough_ramp.color_ramp.elements[1].color = (0.95, 0.95, 0.95, 1.0)  # mortar
    tree.links.new(brick.outputs["Fac"], rough_ramp.inputs["Fac"])
    tree.links.new(rough_ramp.outputs["Color"], bsdf.inputs["Roughness"])

    # Bump for mortar joints
    bump = _add_node(tree, "ShaderNodeBump", 200, -300)
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.02
    invert = _add_node(tree, "ShaderNodeInvert", 100, -250)
    tree.links.new(brick.outputs["Fac"], invert.inputs["Color"])
    tree.links.new(invert.outputs["Color"], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def _setup_stone_texture(tree, bsdf, base_color, secondary):
    """Procedural stone via Voronoi cells + noise variation."""
    mapping = _setup_tex_coords(tree, scale=(3, 3, 3))

    # Voronoi for stone block pattern
    voronoi = _add_node(tree, "ShaderNodeTexVoronoi", -400, 100)
    voronoi.feature = "F1"
    voronoi.inputs["Scale"].default_value = 4.0
    voronoi.inputs["Randomness"].default_value = 0.8
    tree.links.new(mapping.outputs["Vector"], voronoi.inputs["Vector"])

    # Noise for per-stone color variation
    noise = _add_node(tree, "ShaderNodeTexNoise", -400, -100)
    noise.inputs["Scale"].default_value = 8.0
    noise.inputs["Detail"].default_value = 5.0
    noise.inputs["Roughness"].default_value = 0.6
    tree.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

    # Color ramp from base to secondary
    ramp = _add_node(tree, "ShaderNodeValToRGB", -100, 100)
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[0].color = base_color
    ramp.color_ramp.elements[1].position = 0.65
    ramp.color_ramp.elements[1].color = secondary
    tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])

    # Mix with Voronoi edges for joint lines
    mix, fac_in, c1_in, c2_in, c_out = _add_mix_rgb(tree, 100, 0, "DARKEN", 0.2)
    tree.links.new(ramp.outputs["Color"], mix.inputs[c1_in])
    tree.links.new(voronoi.outputs["Distance"], mix.inputs[c2_in])
    tree.links.new(mix.outputs[c_out], bsdf.inputs["Base Color"])

    # Bump from Voronoi distance
    bump = _add_node(tree, "ShaderNodeBump", 200, -200)
    bump.inputs["Strength"].default_value = 0.4
    tree.links.new(voronoi.outputs["Distance"], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def _setup_concrete_texture(tree, bsdf, base_color, secondary):
    """Subtle concrete surface variation — noise-based imperfections."""
    mapping = _setup_tex_coords(tree, scale=(2, 2, 2))

    # Large-scale tone variation
    noise1 = _add_node(tree, "ShaderNodeTexNoise", -400, 100)
    noise1.inputs["Scale"].default_value = 3.0
    noise1.inputs["Detail"].default_value = 2.0
    noise1.inputs["Roughness"].default_value = 0.4
    tree.links.new(mapping.outputs["Vector"], noise1.inputs["Vector"])

    # Fine surface texture
    noise2 = _add_node(tree, "ShaderNodeTexNoise", -400, -100)
    noise2.inputs["Scale"].default_value = 40.0
    noise2.inputs["Detail"].default_value = 8.0
    noise2.inputs["Roughness"].default_value = 0.7
    tree.links.new(mapping.outputs["Vector"], noise2.inputs["Vector"])

    # Color blend
    ramp = _add_node(tree, "ShaderNodeValToRGB", -100, 100)
    ramp.color_ramp.elements[0].position = 0.40
    ramp.color_ramp.elements[0].color = base_color
    ramp.color_ramp.elements[1].position = 0.60
    ramp.color_ramp.elements[1].color = secondary
    tree.links.new(noise1.outputs["Fac"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    # Fine roughness variation
    rough_ramp = _add_node(tree, "ShaderNodeValToRGB", -100, -200)
    rough_ramp.color_ramp.elements[0].position = 0.3
    rough_ramp.color_ramp.elements[0].color = (0.75, 0.75, 0.75, 1.0)
    rough_ramp.color_ramp.elements[1].position = 0.7
    rough_ramp.color_ramp.elements[1].color = (0.92, 0.92, 0.92, 1.0)
    tree.links.new(noise2.outputs["Fac"], rough_ramp.inputs["Fac"])
    tree.links.new(rough_ramp.outputs["Color"], bsdf.inputs["Roughness"])

    # Micro bump
    bump = _add_node(tree, "ShaderNodeBump", 200, -300)
    bump.inputs["Strength"].default_value = 0.1
    tree.links.new(noise2.outputs["Fac"], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def _setup_stucco_texture(tree, bsdf, base_color, secondary):
    """Stucco/render surface — slight trowel texture and imperfections."""
    mapping = _setup_tex_coords(tree, scale=(3, 3, 3))

    # Medium trowel marks
    noise = _add_node(tree, "ShaderNodeTexNoise", -400, 0)
    noise.inputs["Scale"].default_value = 25.0
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Roughness"].default_value = 0.5
    tree.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

    # Subtle discoloration
    ramp = _add_node(tree, "ShaderNodeValToRGB", -100, 100)
    ramp.color_ramp.elements[0].position = 0.45
    ramp.color_ramp.elements[0].color = base_color
    ramp.color_ramp.elements[1].position = 0.55
    ramp.color_ramp.elements[1].color = secondary
    tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    # Roughness variation
    rough_ramp = _add_node(tree, "ShaderNodeValToRGB", -100, -200)
    rough_ramp.color_ramp.elements[0].position = 0.3
    rough_ramp.color_ramp.elements[0].color = (0.60, 0.60, 0.60, 1.0)
    rough_ramp.color_ramp.elements[1].position = 0.7
    rough_ramp.color_ramp.elements[1].color = (0.80, 0.80, 0.80, 1.0)
    tree.links.new(noise.outputs["Fac"], rough_ramp.inputs["Fac"])
    tree.links.new(rough_ramp.outputs["Color"], bsdf.inputs["Roughness"])

    # Bump for trowel texture
    bump = _add_node(tree, "ShaderNodeBump", 200, -300)
    bump.inputs["Strength"].default_value = 0.08
    tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def _setup_metal_texture(tree, bsdf, base_color, secondary):
    """Brushed/matte metal surface with subtle directional streaks."""
    mapping = _setup_tex_coords(tree, scale=(5, 5, 5))

    # Anisotropic-like streaks via stretched noise
    noise = _add_node(tree, "ShaderNodeTexNoise", -400, 0)
    noise.inputs["Scale"].default_value = 60.0
    noise.inputs["Detail"].default_value = 3.0
    noise.inputs["Roughness"].default_value = 0.3
    tree.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

    # Very subtle color variation
    ramp = _add_node(tree, "ShaderNodeValToRGB", -100, 0)
    ramp.color_ramp.elements[0].position = 0.45
    ramp.color_ramp.elements[0].color = base_color
    ramp.color_ramp.elements[1].position = 0.55
    ramp.color_ramp.elements[1].color = secondary
    tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    # Roughness micro-variation — matte finish for architectural metal roofing
    rough_ramp = _add_node(tree, "ShaderNodeValToRGB", -100, -200)
    rough_ramp.color_ramp.elements[0].position = 0.40
    rough_ramp.color_ramp.elements[0].color = (0.60, 0.60, 0.60, 1.0)
    rough_ramp.color_ramp.elements[1].position = 0.60
    rough_ramp.color_ramp.elements[1].color = (0.75, 0.75, 0.75, 1.0)
    tree.links.new(noise.outputs["Fac"], rough_ramp.inputs["Fac"])
    tree.links.new(rough_ramp.outputs["Color"], bsdf.inputs["Roughness"])


def _setup_corten_texture(tree, bsdf, base_color, secondary):
    """Weathered Cor-Ten steel with rust patina variation."""
    mapping = _setup_tex_coords(tree, scale=(3, 3, 3))

    # Large rust patches
    noise1 = _add_node(tree, "ShaderNodeTexNoise", -400, 150)
    noise1.inputs["Scale"].default_value = 5.0
    noise1.inputs["Detail"].default_value = 4.0
    noise1.inputs["Roughness"].default_value = 0.6
    tree.links.new(mapping.outputs["Vector"], noise1.inputs["Vector"])

    # Fine rust texture
    noise2 = _add_node(tree, "ShaderNodeTexNoise", -400, -50)
    noise2.inputs["Scale"].default_value = 30.0
    noise2.inputs["Detail"].default_value = 8.0
    noise2.inputs["Roughness"].default_value = 0.8
    tree.links.new(mapping.outputs["Vector"], noise2.inputs["Vector"])

    # Rust color palette
    ramp = _add_node(tree, "ShaderNodeValToRGB", -100, 100)
    ramp.color_ramp.elements[0].position = 0.25
    ramp.color_ramp.elements[0].color = base_color
    ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[1].color = secondary
    # Add midpoint for deeper rust
    mid = ramp.color_ramp.elements.new(0.50)
    mid.color = (0.35, 0.15, 0.06, 1.0)
    tree.links.new(noise1.outputs["Fac"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    # Roughness: heavy variation for patina
    rough_ramp = _add_node(tree, "ShaderNodeValToRGB", -100, -200)
    rough_ramp.color_ramp.elements[0].position = 0.2
    rough_ramp.color_ramp.elements[0].color = (0.55, 0.55, 0.55, 1.0)
    rough_ramp.color_ramp.elements[1].position = 0.8
    rough_ramp.color_ramp.elements[1].color = (0.90, 0.90, 0.90, 1.0)
    tree.links.new(noise2.outputs["Fac"], rough_ramp.inputs["Fac"])
    tree.links.new(rough_ramp.outputs["Color"], bsdf.inputs["Roughness"])

    # Bump for pitted surface
    bump = _add_node(tree, "ShaderNodeBump", 200, -300)
    bump.inputs["Strength"].default_value = 0.3
    tree.links.new(noise2.outputs["Fac"], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def _setup_clay_tile_texture(tree, bsdf, base_color, secondary, scale=6.0):
    """Clay/terracotta roof tile pattern with wave ridges."""
    mapping = _setup_tex_coords(tree, scale=(scale, scale, scale))

    # Wave for tile ridges
    wave = _add_node(tree, "ShaderNodeTexWave", -400, 100)
    wave.wave_type = "BANDS"
    wave.bands_direction = "X"
    wave.inputs["Scale"].default_value = 12.0
    wave.inputs["Distortion"].default_value = 1.0
    wave.inputs["Detail"].default_value = 2.0
    tree.links.new(mapping.outputs["Vector"], wave.inputs["Vector"])

    # Per-tile color noise
    noise = _add_node(tree, "ShaderNodeTexNoise", -400, -100)
    noise.inputs["Scale"].default_value = 15.0
    noise.inputs["Detail"].default_value = 3.0
    tree.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

    ramp = _add_node(tree, "ShaderNodeValToRGB", -100, 0)
    ramp.color_ramp.elements[0].position = 0.3
    ramp.color_ramp.elements[0].color = base_color
    ramp.color_ramp.elements[1].position = 0.7
    ramp.color_ramp.elements[1].color = secondary
    tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])

    mix, fac_in, c1_in, c2_in, c_out = _add_mix_rgb(tree, 100, 0, "MULTIPLY", 0.15)
    tree.links.new(ramp.outputs["Color"], mix.inputs[c1_in])
    tree.links.new(wave.outputs["Fac"], mix.inputs[c2_in])
    tree.links.new(mix.outputs[c_out], bsdf.inputs["Base Color"])

    # Bump for tile ridges
    bump = _add_node(tree, "ShaderNodeBump", 200, -200)
    bump.inputs["Strength"].default_value = 0.5
    tree.links.new(wave.outputs["Fac"], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def _setup_greenery_texture(tree, bsdf, base_color, secondary):
    """Foliage/planter box texture with leaf-like variation."""
    mapping = _setup_tex_coords(tree, scale=(4, 4, 4))

    # Voronoi for leaf-like cells
    voronoi = _add_node(tree, "ShaderNodeTexVoronoi", -400, 100)
    voronoi.feature = "F1"
    voronoi.inputs["Scale"].default_value = 15.0
    voronoi.inputs["Randomness"].default_value = 1.0
    tree.links.new(mapping.outputs["Vector"], voronoi.inputs["Vector"])

    noise = _add_node(tree, "ShaderNodeTexNoise", -400, -100)
    noise.inputs["Scale"].default_value = 8.0
    noise.inputs["Detail"].default_value = 5.0
    tree.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

    ramp = _add_node(tree, "ShaderNodeValToRGB", -100, 0)
    ramp.color_ramp.elements[0].position = 0.3
    ramp.color_ramp.elements[0].color = base_color
    ramp.color_ramp.elements[1].position = 0.7
    ramp.color_ramp.elements[1].color = secondary
    mid = ramp.color_ramp.elements.new(0.5)
    mid.color = (0.22, 0.40, 0.18, 1.0)  # darker green
    tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    # Bump for leaf depth
    bump = _add_node(tree, "ShaderNodeBump", 200, -200)
    bump.inputs["Strength"].default_value = 0.3
    tree.links.new(voronoi.outputs["Distance"], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


# Dispatch table: material_key → texture setup function
_TEXTURE_BUILDERS = {
    "dark_wood":      _setup_wood_texture,
    "light_wood":     _setup_wood_texture,
    "brick":          _setup_brick_texture,
    "stone":          _setup_stone_texture,
    "concrete":       _setup_concrete_texture,
    "white_concrete": _setup_concrete_texture,
    "white_stucco":   _setup_stucco_texture,
    "warm_stucco":    _setup_stucco_texture,
    "metal":          _setup_metal_texture,
    "black_metal":    _setup_metal_texture,
    "zinc":           _setup_metal_texture,
    "copper":         _setup_metal_texture,
    "charcoal":       _setup_metal_texture,
    "corten":         _setup_corten_texture,
    "clay_tile":      _setup_clay_tile_texture,
    "terracotta":     _setup_clay_tile_texture,
    "greenery":       _setup_greenery_texture,
}


def _get_or_create_textured_material(name, color, material_key=None):
    """Create a material with procedural textures for exterior surfaces.

    Falls back to a plain PBR material for material types that don't
    have a texture builder defined.
    """
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True

    tree = mat.node_tree
    bsdf = tree.nodes.get("Principled BSDF")
    if not bsdf:
        # Shouldn't happen, but recreate if missing
        bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
        output = tree.nodes.get("Material Output")
        if output:
            tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Apply PBR base settings first
    preset = _PBR_PRESETS.get(material_key, {}) if material_key else {}
    if "metallic" in preset:
        bsdf.inputs["Metallic"].default_value = preset["metallic"]
    if "specular" in preset:
        spec_input = (bsdf.inputs.get("Specular IOR Level")
                      or bsdf.inputs.get("Specular"))
        if spec_input:
            spec_input.default_value = preset["specular"]
    if "alpha" in preset:
        bsdf.inputs["Alpha"].default_value = preset["alpha"]
        if hasattr(mat, "blend_method"):
            mat.blend_method = "BLEND"
        mat.use_backface_culling = True

    # Clear old texture nodes (keep BSDF + Output)
    _clear_extra_nodes(tree)

    # Build procedural texture if available
    builder = _TEXTURE_BUILDERS.get(material_key)
    if builder:
        # Derive secondary from the base color (slightly darker variant)
        # instead of using the hardcoded _EXTERIOR_SECONDARY table, so that
        # harmonized / palette-overridden colors are preserved in the texture.
        secondary = (
            color[0] * 0.85,
            color[1] * 0.85,
            color[2] * 0.85,
            color[3] if len(color) > 3 else 1.0,
        )
        builder(tree, bsdf, color, secondary)
    else:
        # Plain color fallback
        bsdf.inputs["Base Color"].default_value = color
        if "roughness" in preset:
            bsdf.inputs["Roughness"].default_value = preset["roughness"]

    mat.diffuse_color = color  # viewport Solid mode display
    return mat


# ---------------------------------------------------------------------------
# Roof ↔ cladding color harmony
# ---------------------------------------------------------------------------

# Explicit roof color overrides keyed by (cladding_material, roof_material).
# These replace the default EXTERIOR_COLORS entry entirely — no blending.
# Designed to produce good-looking, architecturally appropriate pairings.
_ROOF_PALETTE = {
    # Dark wood cladding
    ("dark_wood", "metal"):          (0.15, 0.14, 0.13, 1.0),   # warm charcoal (standing-seam)
    ("dark_wood", "slate"):          (0.35, 0.28, 0.22, 1.0),   # warm charcoal-brown
    ("dark_wood", "shingle"):        (0.30, 0.22, 0.15, 1.0),   # dark cedar
    ("dark_wood", "clay_tile"):      (0.60, 0.30, 0.15, 1.0),   # terracotta
    # Light wood cladding
    ("light_wood", "metal"):         (0.90, 0.88, 0.82, 1.0),   # warm white
    ("light_wood", "slate"):         (0.32, 0.30, 0.28, 1.0),   # warm dark grey
    ("light_wood", "clay_tile"):     (0.55, 0.28, 0.14, 1.0),   # warm tile
    # Brick cladding
    ("brick", "metal"):              (0.28, 0.26, 0.24, 1.0),   # dark bronze
    ("brick", "slate"):              (0.30, 0.28, 0.30, 1.0),   # cool dark slate
    ("brick", "shingle"):            (0.25, 0.20, 0.16, 1.0),   # dark brown
    # White stucco / concrete cladding
    ("white_stucco", "metal"):       (0.30, 0.28, 0.26, 1.0),   # dark warm grey
    ("white_stucco", "clay_tile"):   (0.58, 0.25, 0.12, 1.0),   # rich terracotta
    ("white_concrete", "metal"):     (0.25, 0.25, 0.27, 1.0),   # anthracite
    # Stone cladding
    ("stone", "metal"):              (0.38, 0.32, 0.26, 1.0),   # bronze
    ("stone", "slate"):              (0.30, 0.30, 0.35, 1.0),   # blue-grey slate
    ("stone", "clay_tile"):          (0.55, 0.25, 0.12, 1.0),   # warm tile
    # Concrete cladding
    ("concrete", "metal"):           (0.35, 0.33, 0.30, 1.0),   # warm dark grey
    # Corten cladding
    ("corten", "metal"):             (0.22, 0.20, 0.18, 1.0),   # near-black warm
    # Black metal cladding
    ("black_metal", "metal"):        (0.88, 0.86, 0.82, 1.0),   # light cream contrast
    # Glass cladding
    ("glass", "metal"):              (0.25, 0.25, 0.27, 1.0),   # anthracite
    ("glass", "slate"):              (0.30, 0.30, 0.35, 1.0),   # blue-grey slate
    # Charcoal cladding
    ("charcoal", "metal"):           (0.85, 0.82, 0.76, 1.0),   # warm cream contrast
    ("charcoal", "slate"):           (0.35, 0.32, 0.28, 1.0),   # warm charcoal
    ("charcoal", "shingle"):         (0.28, 0.22, 0.16, 1.0),   # dark cedar
    # Zinc cladding
    ("zinc", "metal"):               (0.30, 0.28, 0.24, 1.0),   # dark warm bronze
    ("zinc", "slate"):               (0.32, 0.30, 0.28, 1.0),   # warm dark grey
}

# Fallback tint when no explicit pair exists
_CLADDING_TINTS = {
    "dark_wood":      ((0.45, 0.35, 0.22), 0.45),
    "light_wood":     ((0.50, 0.42, 0.30), 0.35),
    "brick":          ((0.35, 0.22, 0.16), 0.30),
    "terracotta":     ((0.40, 0.28, 0.18), 0.30),
    "corten":         ((0.30, 0.20, 0.14), 0.30),
    "copper":         ((0.35, 0.28, 0.18), 0.25),
    "stone":          ((0.38, 0.32, 0.24), 0.25),
    "white_stucco":   ((0.50, 0.48, 0.44), 0.15),
    "concrete":       ((0.45, 0.42, 0.38), 0.15),
    "white_concrete": ((0.48, 0.46, 0.44), 0.10),
    "black_metal":    ((0.80, 0.78, 0.74), 0.20),
    "charcoal":       ((0.25, 0.22, 0.20), 0.15),
    "zinc":           ((0.50, 0.48, 0.45), 0.10),
    "glass":          ((0.50, 0.50, 0.48), 0.10),
}

# Track the active cladding so roof assignment can harmonize
_active_cladding = {"key": None}


def set_active_cladding(cladding_key):
    """Call before assigning roof materials to enable color harmony."""
    _active_cladding["key"] = cladding_key


def _harmonize_roof_color(roof_color, roof_key, cladding_key):
    """Return a harmonized roof color based on the cladding material.

    Uses explicit palette pairs first, falls back to strong tinting.
    """
    if not cladding_key:
        return roof_color
    # Check explicit good pair
    pair_color = _ROOF_PALETTE.get((cladding_key, roof_key))
    if pair_color:
        return pair_color
    # Fallback: strong tint blend
    tint_entry = _CLADDING_TINTS.get(cladding_key)
    if not tint_entry:
        return roof_color
    tint, strength = tint_entry
    return (
        roof_color[0] * (1 - strength) + tint[0] * strength,
        roof_color[1] * (1 - strength) + tint[1] * strength,
        roof_color[2] * (1 - strength) + tint[2] * strength,
        roof_color[3],
    )


# ---------------------------------------------------------------------------
# Public exterior material assignment (now with textures + harmony)
# ---------------------------------------------------------------------------

def assign_roof_material(obj, material_name=None, cladding_override=None,
                         color_override=None):
    """Assign a roof material with style-appropriate color.

    Roofs are NEVER metallic/reflective — they are matte painted/coated
    surfaces.  The color is chosen to complement the active cladding.

    Args:
        obj: Blender object to assign the material to.
        material_name: Roof material type (e.g. "metal", "slate").
        cladding_override: Explicit cladding key — bypasses _active_cladding.
        color_override: Optional RGBA tuple for user-specified roof color.
    """
    key = material_name.lower().replace(" ", "_") if material_name else ""
    # Prefer explicit cladding parameter over module-level state
    cladding = cladding_override or _active_cladding.get("key") or ""

    # ── Pick roof color directly ──────────────────────────────────
    # Inline lookup — no external function dependencies.
    # Keyed by (cladding, roof_material).  Fallback: warm-tinted default.
    _ROOF_COLORS = {
        # dark_wood cladding  →  warm / earthy roofs
        ("dark_wood", "metal"):      (0.15, 0.14, 0.13, 1.0),  # warm charcoal (standing-seam)
        ("dark_wood", "slate"):      (0.35, 0.28, 0.22, 1.0),  # charcoal-brown
        ("dark_wood", "shingle"):    (0.30, 0.22, 0.15, 1.0),  # dark cedar
        ("dark_wood", "clay_tile"):  (0.60, 0.30, 0.15, 1.0),  # terracotta
        ("dark_wood", "concrete"):   (0.78, 0.74, 0.68, 1.0),  # warm light concrete
        ("dark_wood", "zinc"):       (0.75, 0.72, 0.66, 1.0),  # warm zinc
        ("dark_wood", "copper"):     (0.62, 0.38, 0.15, 1.0),  # copper
        # light_wood cladding
        ("light_wood", "metal"):     (0.90, 0.88, 0.82, 1.0),  # warm white
        ("light_wood", "slate"):     (0.32, 0.30, 0.28, 1.0),  # warm dark grey
        ("light_wood", "clay_tile"): (0.55, 0.28, 0.14, 1.0),  # warm tile
        # brick cladding
        ("brick", "metal"):          (0.28, 0.26, 0.24, 1.0),  # dark bronze
        ("brick", "slate"):          (0.30, 0.28, 0.30, 1.0),  # cool dark slate
        ("brick", "shingle"):        (0.25, 0.20, 0.16, 1.0),  # dark brown
        # white / stucco cladding
        ("white_stucco", "metal"):   (0.30, 0.28, 0.26, 1.0),  # dark warm grey
        ("white_stucco", "clay_tile"): (0.58, 0.25, 0.12, 1.0),  # rich terracotta
        # warm_stucco cladding (Mediterranean)
        ("warm_stucco", "clay_tile"): (0.55, 0.22, 0.10, 1.0),  # terracotta
        ("warm_stucco", "metal"):     (0.30, 0.28, 0.24, 1.0),  # warm bronze
        ("warm_stucco", "slate"):     (0.32, 0.28, 0.22, 1.0),  # warm charcoal
        ("white_concrete", "metal"): (0.25, 0.25, 0.27, 1.0),  # anthracite
        # stone cladding
        ("stone", "metal"):          (0.38, 0.32, 0.26, 1.0),  # bronze
        ("stone", "slate"):          (0.30, 0.30, 0.35, 1.0),  # blue-grey slate
        ("stone", "clay_tile"):      (0.55, 0.25, 0.12, 1.0),  # warm tile
        # concrete cladding
        ("concrete", "metal"):       (0.35, 0.33, 0.30, 1.0),  # warm dark grey
        # corten cladding
        ("corten", "metal"):         (0.22, 0.20, 0.18, 1.0),  # near-black warm
        # black_metal cladding
        ("black_metal", "metal"):    (0.88, 0.86, 0.82, 1.0),  # light cream contrast
        # glass cladding
        ("glass", "metal"):          (0.25, 0.25, 0.27, 1.0),  # anthracite
        ("glass", "slate"):          (0.30, 0.30, 0.35, 1.0),  # blue-grey slate
        # charcoal cladding
        ("charcoal", "metal"):       (0.85, 0.82, 0.76, 1.0),  # warm cream contrast
        ("charcoal", "slate"):       (0.35, 0.32, 0.28, 1.0),  # warm charcoal
        ("charcoal", "shingle"):     (0.28, 0.22, 0.16, 1.0),  # dark cedar
        # zinc cladding
        ("zinc", "metal"):           (0.30, 0.28, 0.24, 1.0),  # dark warm bronze
        ("zinc", "slate"):           (0.32, 0.30, 0.28, 1.0),  # warm dark grey
    }

    # User-specified color takes absolute priority
    if color_override:
        color = color_override
    else:
        color = _ROOF_COLORS.get((cladding, key))
    if color is None:
        # Fallback: use _harmonize_roof_color for proper cladding-aware tinting.
        # Use warm architectural defaults for metal/zinc roofs instead of raw
        # steel grey — real metal roofs are painted or coated, not bare metal.
        _ROOF_DEFAULTS = {
            "metal": (0.32, 0.30, 0.27, 1.0),   # warm dark charcoal (painted standing-seam)
            "zinc":  (0.38, 0.36, 0.32, 1.0),    # warm weathered zinc
        }
        base = _ROOF_DEFAULTS.get(key, EXTERIOR_COLORS.get(key, DEFAULT_EXTERIOR_COLOR))
        color = _harmonize_roof_color(base, key, cladding)

    if color_override:
        r, g, b = color_override[:3]
        mat_name = f"FP3D_Roof_{key}_{r:.2f}_{g:.2f}_{b:.2f}"
    elif key:
        mat_name = f"FP3D_Roof_{key}_{cladding}"
    else:
        mat_name = "FP3D_Roof"
    print(f"[FP3D] roof: ({cladding}, {key}) → {color}")

    # --- Reuse existing material if available, otherwise create fresh ---
    # Multiple objects (main roof, raised_volume, ceiling) may share the same
    # material name.  Removing it would orphan earlier users' references.
    import bpy
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    tree = mat.node_tree
    # Clear any stale nodes from a previous generation
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Textured roofs (clay_tile, slate, shingle, thatch) get their
    # dedicated texture builders for realistic appearance.  Metal/zinc/copper
    # roofs use flat color — the metal texture builder creates noise patterns
    # designed for walls that wash out color on large flat roof planes.
    _TEXTURED_ROOFS = {"clay_tile", "slate", "shingle", "thatch", "terracotta"}
    secondary = _EXTERIOR_SECONDARY.get(key, color)

    if key in _TEXTURED_ROOFS:
        builder = _TEXTURE_BUILDERS.get(key)
        if builder:
            builder(tree, bsdf, color, secondary)
    else:
        bsdf.inputs["Base Color"].default_value = color

    # Non-metallic, matte PBR — roofs are not mirrors
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.65
    spec_input = (bsdf.inputs.get("Specular IOR Level")
                  or bsdf.inputs.get("Specular"))
    if spec_input:
        spec_input.default_value = 0.15

    mat.diffuse_color = color  # viewport Solid mode display
    _apply_mat(obj, mat)


def assign_foundation_material(obj, material_name=None):
    """Assign a textured foundation material."""
    key = material_name.lower().replace(" ", "_") if material_name else ""
    color = EXTERIOR_COLORS.get(key, DEFAULT_EXTERIOR_COLOR)
    mat_name = f"FP3D_Foundation_{key}" if key else "FP3D_Foundation"
    _apply_mat(obj, _get_or_create_textured_material(mat_name, color, material_key=key or None))


def assign_exterior_material(obj, material_name=None, color_override=None):
    """Assign a textured exterior material (trim, cladding, frames, etc.).

    Args:
        obj: Blender object.
        material_name: Material key (e.g. "white_stucco").
        color_override: Optional RGBA tuple to override the default color
                        (e.g. for user-specified colors like pink or yellow).
    """
    key = material_name.lower().replace(" ", "_") if material_name else ""
    color = color_override or EXTERIOR_COLORS.get(key, DEFAULT_EXTERIOR_COLOR)
    # Use a unique material name when color is overridden so it doesn't
    # conflict with the default-colored version of the same material.
    if color_override:
        r, g, b = color_override[:3]
        mat_name = f"FP3D_Ext_{key}_{r:.2f}_{g:.2f}_{b:.2f}"
    else:
        mat_name = f"FP3D_Ext_{key}" if key else "FP3D_Ext"
    _apply_mat(obj, _get_or_create_textured_material(mat_name, color, material_key=key or None))
