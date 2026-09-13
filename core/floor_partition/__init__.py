"""楼层功能分区模块。"""

from __future__ import annotations

from .circulation import CirculationLayout, generate_residential_circulation
from .contracts import FloorPartitionProblem, build_floor_partition_problem, target_areas_for_area
from .doors import Door, enumerate_residential_door_candidates, solve_residential_doors
from .export import export_unit_configs
from .growth import PartitionResult, grow_residential_units


def run_residential_floor_partition(config: dict):
    """执行一轮住宅楼层分区规则链。"""
    problem = build_floor_partition_problem(config)
    circulation = generate_residential_circulation(problem)
    doors = solve_residential_doors(problem, circulation)
    result = grow_residential_units(problem, circulation, doors)
    return problem, result


__all__ = [
    "CirculationLayout",
    "Door",
    "FloorPartitionProblem",
    "PartitionResult",
    "build_floor_partition_problem",
    "enumerate_residential_door_candidates",
    "export_unit_configs",
    "generate_residential_circulation",
    "grow_residential_units",
    "run_residential_floor_partition",
    "solve_residential_doors",
    "target_areas_for_area",
]
