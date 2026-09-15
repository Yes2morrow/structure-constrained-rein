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
    epsilon = max(problem.profile.grid_size * 0.5, 0.05)
    clearances = {
        "west": core_min_x - min_x,
        "east": max_x - core_max_x,
        "south": core_min_y - min_y,
        "north": max_y - core_max_y,
    }
    # 只在真正放得下走道的一侧里挑：优先整宽贴核，其次任何有余量的侧边。
    usable = [side for side, value in clearances.items() if value >= problem.profile.corridor_width - 1e-9]
    if not usable:
        usable = [side for side, value in clearances.items() if value > epsilon]
    if not usable:
        return max(clearances, key=clearances.get)
    if problem.entrance is not None:
        midpoint = problem.entrance.interpolate(0.5, normalized=True)
        distances = {
            "west": abs(midpoint.x - core_min_x),
            "east": abs(midpoint.x - core_max_x),
            "south": abs(midpoint.y - core_min_y),
            "north": abs(midpoint.y - core_max_y),
        }
        return min(usable, key=lambda side: (distances[side], -clearances[side]))
    return max(usable, key=clearances.get)


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


def _required_length(problem: FloorPartitionProblem) -> float:
    """走道条带需要覆盖的核边长度：核开口宽度与全部户门面宽取较大值。

    户门面宽按“门宽 × 户数 + 隐私门间距 × (户数 - 1)”估算，保证各户门
    在走道外侧能按最小间距排开；一层一户时收敛到核开口宽度的最小门厅。
    """
    profile = problem.profile
    frontage = profile.door_width * profile.unit_count + profile.min_door_spacing * (profile.unit_count - 1)
    return max(profile.opening_width, frontage)


def _side_band(problem: FloorPartitionProblem, side: str, length: float) -> Polygon:
    """沿开口一侧生成贴核条带：深度为走道宽，长度按需收敛并居中于核门。"""
    width = problem.profile.corridor_width
    min_x, min_y, max_x, max_y = problem.traffic_core.bounds
    low, high = (min_y, max_y) if side in ("east", "west") else (min_x, max_x)
    length = min(length, high - low)
    center = (low + high) / 2.0
    start = min(max(center - length / 2.0, low), high - length)
    end = start + length
    if side == "west":
        return box(min_x - width, start, min_x, end)
    if side == "east":
        return box(max_x, start, max_x + width, end)
    if side == "south":
        return box(start, min_y - width, end, min_y)
    return box(start, max_y, end, max_y)


def generate_residential_circulation(problem: FloorPartitionProblem) -> CirculationLayout:
    """在开口一侧生成贴核的最小走道条带，不再整圈包裹交通核。

    条带只覆盖核门开口一侧：长度取“核开口宽度”与“按门宽与隐私门间距排布
    全部户门所需面宽”的较大值，再收敛到该侧核边的实际长度；核门开口落在
    条带内部保持走道连通，满足入户通勤与各户门前的隐私间距即可。
    """
    side = _choose_opening_side(problem)
    corridor = _side_band(problem, side, _required_length(problem)).intersection(problem.boundary)
    blocked = unary_union(
        [
            fixed_polygon(item)
            for item in problem.fixed_objects
            if str(item.get("type")) not in TRAFFIC_CORE_TYPES
        ]
    )
    corridor = corridor.difference(problem.traffic_core)
    if not blocked.is_empty:
        corridor = corridor.difference(blocked)
    opening = _opening_cut(problem, side).intersection(corridor)
    if corridor.is_empty or corridor.area <= 1e-9:
        raise ValueError("无法根据当前交通核与结构约束生成有效走道")
    return CirculationLayout(corridor=corridor, opening=opening, opening_side=side)
