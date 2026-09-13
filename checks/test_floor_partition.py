"""楼层分区规则链检查。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml
from shapely.ops import unary_union

from core.floor_partition import build_floor_partition_problem, export_unit_configs, run_residential_floor_partition


def fixture():
    return {
        "ProjectType": "adaptive_reuse",
        "ExistingBuilding": {
            "boundary": [[0.0, 0.0], [20.0, 0.0], [20.0, 16.0], [0.0, 16.0]],
            "door_positions": [[9.0, 0.0], [11.0, 0.0]],
            "fixed_objects": [
                {"id": "c1", "type": "column", "center": [4.0, 4.0], "size": [0.4, 0.4], "rect": [3.8, 3.8, 4.2, 4.2], "parametric": True},
                {"id": "c2", "type": "column", "center": [16.0, 12.0], "size": [0.4, 0.4], "rect": [15.8, 11.8, 16.2, 12.2], "parametric": True},
                {"id": "tc", "type": "traffic_core", "rect": [8.0, 5.0, 12.0, 9.0], "parametric": True},
            ],
            "original_spaces": [],
        },
        "AdaptiveReuseEnvironment": {"grid_size": 0.5},
        "Training": {"ckpt_path": ""},
        "FloorPartition": {
            "enabled": True,
            "program_type": "residential",
            "grid_size": 0.5,
            "residential": {
                "unit_count": 3,
                "target_area_mode": "custom",
                "target_areas": [60.0, 55.0, 45.0],
                "corridor_width": 1.5,
                "min_door_spacing": 2.4,
                "door_width": 0.9,
                "opening_width": 1.8,
                "opening_side": "south",
                "export_config_prefix": "unit",
            },
        },
    }


class FloorPartitionTest(unittest.TestCase):
    def test_problem_contract_and_rule_pipeline(self):
        problem = build_floor_partition_problem(fixture())
        self.assertEqual(problem.profile.unit_count, 3)
        self.assertEqual(problem.program_type, "residential")

        _, result = run_residential_floor_partition(fixture())
        self.assertEqual(len(result.doors), 3)
        self.assertEqual(len(result.unit_polygons), 3)
        self.assertTrue(all(len(door.points) == 2 for door in result.doors))

        combined = unary_union(list(result.unit_polygons.values()))
        self.assertAlmostEqual(combined.area, result.allocatable_space.area, places=4)
        self.assertLess(result.corridor.intersection(combined).area, 1e-9)

    def test_export_writes_unit_configs(self):
        config = fixture()
        _, result = run_residential_floor_partition(config)
        with tempfile.TemporaryDirectory() as folder:
            paths = export_unit_configs(config, result, folder)
            self.assertEqual(len(paths), 3)
            for path in paths:
                self.assertTrue(Path(path).is_file())
                exported = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
                self.assertEqual(len(exported["ExistingBuilding"]["door_positions"]), 2)
                self.assertIn("door_positions", exported["FloorPartitionResult"])
                self.assertNotIn("door", exported["FloorPartitionResult"])
            self.assertTrue((Path(folder) / "summary.yaml").is_file())


if __name__ == "__main__":
    unittest.main()
