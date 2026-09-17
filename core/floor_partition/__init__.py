"""楼层功能分区模块。"""

from __future__ import annotations
from functools import lru_cache
from copy import deepcopy
import json

from .circulation import CirculationLayout, generate_residential_circulation
from .contracts import FloorPartitionProblem, build_floor_partition_problem, target_areas_for_area
from .doors import Door, enumerate_residential_door_candidates, solve_residential_doors
from .export import export_unit_configs
from .growth import PartitionResult, grow_residential_units


def run_residential_floor_partition(config: dict):
    """Budgeted preview; reuse identical geometry across GUI panels."""
    relevant={key:config.get(key,{}) for key in ('ExistingBuilding','FloorPartition','AdaptiveReuseEnvironment')}
    return deepcopy(_preview_cached(json.dumps(relevant,sort_keys=True)))


@lru_cache(maxsize=8)
def _preview_cached(serialized):
    config=json.loads(serialized)
    problem = build_floor_partition_problem(config)
    from .structured import candidate_partitions
    result, _ = candidate_partitions(problem)[0]
    if problem.search_settings.get('enabled', True):
        from .joint import search_joint
        result, _, _ = search_joint(problem, result, steps=int(problem.search_settings.get('preview_steps', 4)))
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
