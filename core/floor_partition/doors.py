"""住宅楼层分区的门位求解。"""

from __future__ import annotations

from dataclasses import dataclass
import math

from shapely.geometry import Point
from shapely.ops import unary_union

from .circulation import CirculationLayout
from .contracts import FloorPartitionProblem


@dataclass(frozen=True)
class Door:
    unit_id: str
    points: tuple[tuple[float, float], tuple[float, float]]
    edge: str
    corridor_distance: float

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.points[0][0] + self.points[1][0]) / 2.0,
            (self.points[0][1] + self.points[1][1]) / 2.0,
        )


@dataclass(frozen=True)
class DoorCandidate:
    points: tuple[tuple[float, float], tuple[float, float]]
    edge: str
    corridor_distance: float
    free_probe_area: float


def _polygon_parts(geometry):
    if geometry.geom_type == "Polygon":
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [part for part in geometry.geoms if part.geom_type == "Polygon"]
    return []


def _edge_label(problem: FloorPartitionProblem, start: tuple[float, float], end: tuple[float, float]) -> str:
    center = problem.traffic_core.centroid
    if abs(end[0] - start[0]) >= abs(end[1] - start[1]):
        return "north" if (start[1] + end[1]) / 2.0 >= center.y else "south"
    return "east" if (start[0] + end[0]) / 2.0 >= center.x else "west"


def _allocatable_space(problem: FloorPartitionProblem, circulation: CirculationLayout):
    return problem.boundary.difference(unary_union([problem.fixed_union, circulation.corridor]))


def _candidate_positions(length: float, door_width: float, spacing: float) -> list[float]:
    if length <= door_width + 1e-9:
        return [length / 2.0]
    positions = []
    current = door_width / 2.0
    limit = length - door_width / 2.0
    while current <= limit + 1e-9:
        positions.append(current)
        current += spacing
    if not positions or positions[-1] < limit - spacing * 0.35:
        positions.append(limit)
    return positions


def _wrapped_distance(value: float, other: float, perimeter: float) -> float:
    direct = abs(value - other)
    return min(direct, perimeter - direct)


def _door_segment(
    center: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    width: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    if length <= 1e-9:
        return (center, center)
    tangent = ((end[0] - start[0]) / length, (end[1] - start[1]) / length)
    half = width / 2.0
    return (
        (center[0] - tangent[0] * half, center[1] - tangent[1] * half),
        (center[0] + tangent[0] * half, center[1] + tangent[1] * half),
    )


def enumerate_residential_door_candidates(problem: FloorPartitionProblem, circulation: CirculationLayout) -> list[DoorCandidate]:
    """在走道外侧枚举可进入剩余可分配空间的门位候选。"""
    allocatable = _allocatable_space(problem, circulation)
    if allocatable.is_empty or allocatable.area <= 1e-9:
        raise ValueError("走道之外没有可供户型分配的剩余空间")
    candidates: list[DoorCandidate] = []
    perimeter_offset = 0.0
    probe_radius = max(problem.profile.grid_size * 0.45, 0.2)
    for polygon in _polygon_parts(circulation.corridor):
        coords = list(polygon.exterior.coords)
        for start, end in zip(coords, coords[1:]):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = math.hypot(dx, dy)
            if length <= 1e-9:
                continue
            edge = _edge_label(problem, start, end)
            for distance in _candidate_positions(length, problem.profile.door_width, problem.profile.min_door_spacing):
                ratio = distance / length
                x = start[0] + dx * ratio
                y = start[1] + dy * ratio
                segment = _door_segment((x, y), start, end, problem.profile.door_width)
                probe = Point(x, y).buffer(probe_radius, cap_style=1).intersection(allocatable)
                if probe.is_empty or probe.area <= 1e-9:
                    continue
                candidates.append(
                    DoorCandidate(
                        points=segment,
                        edge=edge,
                        corridor_distance=perimeter_offset + distance,
                        free_probe_area=float(probe.area),
                    )
                )
            perimeter_offset += length
    if not candidates:
        raise ValueError("当前走道外侧没有可用门位候选")
    return candidates


def solve_residential_doors(problem: FloorPartitionProblem, circulation: CirculationLayout) -> list[Door]:
    """优先拉开门间距，并尽量分散到不同朝向。"""
    candidates = enumerate_residential_door_candidates(problem, circulation)
    unit_count = problem.profile.unit_count
    if len(candidates) < unit_count:
        raise ValueError(f"门位候选不足：需要 {unit_count} 个，实际只有 {len(candidates)} 个")

    perimeter = max(candidate.corridor_distance for candidate in candidates)
    selected: list[DoorCandidate] = []
    seen_edges: set[str] = set()
    first = max(candidates, key=lambda item: (item.free_probe_area, item.corridor_distance))
    selected.append(first)
    seen_edges.add(first.edge)
    remaining = [item for item in candidates if item is not first]

    while len(selected) < unit_count:
        def score(item: DoorCandidate):
            min_distance = min(
                _wrapped_distance(item.corridor_distance, other.corridor_distance, perimeter)
                for other in selected
            )
            edge_bonus = 1.0 if item.edge not in seen_edges else 0.0
            return (min_distance, edge_bonus, item.free_probe_area, item.corridor_distance)

        best = max(remaining, key=score)
        selected.append(best)
        seen_edges.add(best.edge)
        remaining = [item for item in remaining if item is not best]

    ordered = sorted(selected, key=lambda item: item.corridor_distance)
    return [
        Door(
            unit_id=problem.targets[index].unit_id,
            points=door.points,
            edge=door.edge,
            corridor_distance=door.corridor_distance,
        )
        for index, door in enumerate(ordered)
    ]
