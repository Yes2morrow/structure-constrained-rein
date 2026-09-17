"""楼层功能分区模块。"""

from __future__ import annotations
from functools import lru_cache
from copy import deepcopy
import json
from threading import RLock

from .circulation import CirculationLayout, generate_residential_circulation
from .contracts import FloorPartitionProblem, build_floor_partition_problem, target_areas_for_area
from .contracts import Door
from .export import export_unit_configs
from .contracts import PartitionResult

_preview_lock = RLock()


def run_residential_floor_partition(config: dict):
    """Budgeted preview; reuse identical geometry across GUI panels."""
    relevant={key:config.get(key,{}) for key in ('ExistingBuilding','FloorPartition','AdaptiveReuseEnvironment')}
    # Streamlit can start a rerun while the previous script is still solving.
    # lru_cache alone permits duplicate concurrent misses and stalls both runs.
    with _preview_lock:
        preview = _preview_cached(json.dumps(relevant,sort_keys=True))
    return deepcopy(preview)


@lru_cache(maxsize=8)
def _preview_cached(serialized):
    config=json.loads(serialized)
    problem = build_floor_partition_problem(config)
    from .structured import candidate_partitions
    result, _ = candidate_partitions(problem)[0]
    if problem.search_settings.get('enabled', True) and int(problem.search_settings.get('preview_steps', 4)) > 0:
        from .joint import search_joint
        result, _, _ = search_joint(problem, result, steps=int(problem.search_settings.get('preview_steps', 4)))
    return problem, result


__all__ = [
    "CirculationLayout",
    "Door",
    "FloorPartitionProblem",
    "PartitionResult",
    "build_floor_partition_problem",
    "export_unit_configs",
    "generate_residential_circulation",
    "run_residential_floor_partition",
    "target_areas_for_area",
]
