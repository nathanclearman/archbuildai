"""
Blender integration tests for geometry generation.

Run inside Blender:
    blender --background --python tests/test_geometry_blender.py

This tests the full bpy pipeline by loading sample JSON and generating geometry.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import bpy


def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Remove FloorPlan3D collection if it exists
    col = bpy.data.collections.get("FloorPlan3D")
    if col:
        bpy.data.collections.remove(col)


def load_sample_data(name):
    sample_path = Path(__file__).parent / "sample_plans" / name
    with open(sample_path, 'r') as f:
        return json.load(f)


def test_generate_simple_apartment():
    """Test generating geometry from the simple apartment sample."""
    clear_scene()

    from blender_addon import geometry

    data = load_sample_data("simple_apartment.json")
    context = bpy.context

    collection = geometry.create_floorplan_collection(context)
    geometry.generate_walls(data, collection, wall_height=2.7)
    geometry.generate_floors(data, collection)
    geometry.generate_room_labels(data, collection)

    # Verify objects were created
    walls = [o for o in collection.objects if o.get("fp3d_type") == "wall"]
    floors = [o for o in collection.objects if o.get("fp3d_type") == "floor"]
    labels = [o for o in collection.objects if o.get("fp3d_type") == "label"]

    assert len(walls) == 5, f"Expected 5 walls, got {len(walls)}"
    assert len(floors) == 2, f"Expected 2 floors, got {len(floors)}"
    assert len(labels) == 2, f"Expected 2 labels, got {len(labels)}"

    print("PASS: test_generate_simple_apartment")


def test_generate_with_openings():
    """Test door and window boolean operations."""
    clear_scene()

    from blender_addon import geometry

    data = load_sample_data("simple_apartment.json")
    context = bpy.context

    collection = geometry.create_floorplan_collection(context)
    geometry.generate_walls(data, collection, wall_height=2.7)
    geometry.generate_door_openings(data, collection, wall_height=2.7)
    geometry.generate_window_openings(data, collection, wall_height=2.7)

    walls = [o for o in collection.objects if o.get("fp3d_type") == "wall"]
    assert len(walls) == 5, f"Expected 5 walls after openings, got {len(walls)}"

    # Cutter objects should be removed
    cutters = [o for o in collection.objects if "Cutter" in o.name]
    assert len(cutters) == 0, f"Found {len(cutters)} leftover cutter objects"

    print("PASS: test_generate_with_openings")


def test_generate_ceilings():
    """Test ceiling generation."""
    clear_scene()

    from blender_addon import geometry

    data = load_sample_data("simple_apartment.json")
    context = bpy.context

    collection = geometry.create_floorplan_collection(context)
    geometry.generate_ceilings(data, collection, wall_height=2.7)

    ceilings = [o for o in collection.objects if o.get("fp3d_type") == "ceiling"]
    assert len(ceilings) == 2, f"Expected 2 ceilings, got {len(ceilings)}"

    # Verify ceilings are at wall height
    for ceil_obj in ceilings:
        # Check that vertices are at the expected height
        mesh = ceil_obj.data
        for vert in mesh.vertices:
            assert abs(vert.co.z - 2.7) < 0.01, f"Ceiling vertex not at wall height: {vert.co.z}"

    print("PASS: test_generate_ceilings")


def test_collection_clearing():
    """Test that re-generating clears the previous model."""
    clear_scene()

    from blender_addon import geometry

    data = load_sample_data("simple_apartment.json")
    context = bpy.context

    # Generate once
    collection = geometry.create_floorplan_collection(context)
    geometry.generate_walls(data, collection, wall_height=2.7)
    count_1 = len(collection.objects)

    # Generate again (should clear first)
    collection = geometry.create_floorplan_collection(context)
    geometry.generate_walls(data, collection, wall_height=2.7)
    count_2 = len(collection.objects)

    assert count_1 == count_2, f"Second generation has different object count: {count_1} vs {count_2}"

    print("PASS: test_collection_clearing")


if __name__ == "__main__":
    tests = [
        test_generate_simple_apartment,
        test_generate_with_openings,
        test_generate_ceilings,
        test_collection_clearing,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")

    if failed > 0:
        sys.exit(1)
