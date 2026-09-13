"""住宅楼层分区的走道生成。"""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from .contracts import FloorPartitionProblem, TRAFFIC_CORE_TYPES
from core.envs.structure_geometry import fixed_polygon


@dataclass(frozen=True)
class CirculationLayout:
    corridor: object
    opening: object
    opening_side: str


def _choose_opening_side(problem: FloorPartitionProblem) -> str:
    if problem.profile.opening_side != "auto":
        return problem.profile.opening_side
    min_x, min_y, max_x, max_y = problem.boundary.bounds
    core_min_x, core_min_y, core_max_x, core_max_y = problem.traffic_core.bounds
    if problem.entrance is not None:
        midpoint = problem.entrance.interpolate(0.5, normalized=True)
        distances = {
            "west": abs(midpoint.x - core_min_x),
            "east": abs(midpoint.x - core_max_x),
            "south": abs(midpoint.y - core_min_y),
            "north": abs(midpoint.y - core_max_y),
        }
        return min(distances, key=distances.get)
    clearances = {
        "west": core_min_x - min_x,
        "east": max_x - core_max_x,
        "south": core_min_y - min_y,
        "north": max_y - core_max_y,
    }
    return max(clearances, key=clearances.get)


def _opening_cut(problem: FloorPartitionProblem, side: str) -> Polygon:
    width = problem.profile.corridor_width
    opening_width = problem.profile.opening_width
    min_x, min_y, max_x, max_y = problem.traffic_core.bounds
    epsilon = max(problem.profile.grid_size * 0.5, 0.05)
    if side == "north":
        center = (min_x + max_x) / 2.0
        return box(center - opening_width / 2.0, max_y - epsilon, center + opening_width / 2.0, max_y + width + epsilon)
    if side == "south":
        center = (min_x + max_x) / 2.0
        return box(center - opening_width / 2.0, min_y - width - epsilon, center + opening_width / 2.0, min_y + epsilon)
    if side == "east":
        center = (min_y + max_y) / 2.0
        return box(max_x - epsilon, center - opening_width / 2.0, max_x + width + epsilon, center + opening_width / 2.0)
    center = (min_y + max_y) / 2.0
    return box(min_x - width - epsilon, center - opening_width / 2.0, min_x + epsilon, center + opening_width / 2.0)


def generate_residential_circulation(problem: FloorPartitionProblem) -> CirculationLayout:
    """沿交通核外侧生成贴核走道，并预留一个开口。"""
    shell = problem.traffic_core.buffer(problem.profile.corridor_width, cap_style=2, join_style=2).intersection(problem.boundary)
    blocked = unary_union(
        [
            fixed_polygon(item)
            for item in problem.fixed_objects
            if str(item.get("type")) not in TRAFFIC_CORE_TYPES
        ]
    )
    corridor = shell.difference(problem.traffic_core)
    if not blocked.is_empty:
        corridor = corridor.difference(blocked)
    side = _choose_opening_side(problem)
    opening = _opening_cut(problem, side).intersection(corridor)
    if not opening.is_empty:
        candidate = corridor.difference(opening)
        if not candidate.is_empty and candidate.area > 1e-9:
            corridor = candidate
    if corridor.is_empty or corridor.area <= 1e-9:
        raise ValueError("无法根据当前交通核与结构约束生成有效走道")
    return CirculationLayout(corridor=corridor, opening=opening, opening_side=side)
