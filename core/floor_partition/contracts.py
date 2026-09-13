"""楼层功能分区的输入契约与基础编译。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from core.envs.structure_geometry import fixed_polygon, normalize_structures


TRAFFIC_CORE_TYPES = {"traffic_core", "core"}


def _finite_pair(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} 必须是两个数值")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} 必须是有限数值")
    return result


def _normalize_vertices(points: list[list[float]] | list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [_finite_pair(point, "boundary") for point in points]


@dataclass(frozen=True)
class PartitionTarget:
    unit_id: str
    name: str
    requested_area: float


@dataclass(frozen=True)
class ResidentialProfile:
    unit_count: int
    target_area_mode: str
    target_areas: tuple[float, ...]
    corridor_width: float
    min_door_spacing: float
    door_width: float
    opening_width: float
    opening_side: str
    grid_size: float
    export_config_prefix: str


@dataclass(frozen=True)
class FloorPartitionProblem:
    boundary: Polygon
    fixed_objects: tuple[dict[str, Any], ...]
    fixed_union: object
    traffic_core: object
    entrance: LineString | None
    program_type: str
    profile: ResidentialProfile
    targets: tuple[PartitionTarget, ...]


def target_areas_for_area(problem: FloorPartitionProblem, allocatable_area: float) -> tuple[list[float], float]:
    """把配置面积解析成当前可分配面积下的实际目标面积。"""
    if allocatable_area <= 1e-9:
        raise ValueError("走道与结构扣除后没有可分配面积")
    unit_count = problem.profile.unit_count
    if problem.profile.target_area_mode == "custom":
        requested = [float(value) for value in problem.profile.target_areas]
        total_requested = sum(requested)
        if len(requested) != unit_count or total_requested <= 1e-9:
            raise ValueError("自定义户型面积数量必须与户数一致，且总面积大于 0")
        scale = allocatable_area / total_requested
        return [value * scale for value in requested], scale
    equal_area = allocatable_area / unit_count
    return [equal_area] * unit_count, 1.0


def build_floor_partition_problem(config: dict[str, Any]) -> FloorPartitionProblem:
    """从现有配置编译楼层分区问题。当前先支持住宅分区。"""
    building = config.get("ExistingBuilding", {})
    boundary = Polygon(_normalize_vertices(building.get("boundary", [])))
    if not boundary.is_valid or boundary.area <= 0:
        raise ValueError("ExistingBuilding.boundary 必须是有效且面积大于 0 的多边形")

    fixed_objects = tuple(normalize_structures(building.get("fixed_objects", [])))
    fixed_polygons = [fixed_polygon(item) for item in fixed_objects]
    fixed_union = unary_union(fixed_polygons) if fixed_polygons else Polygon()

    traffic_parts = [
        fixed_polygon(item)
        for item in fixed_objects
        if str(item.get("type")) in TRAFFIC_CORE_TYPES
    ]
    if not traffic_parts:
        raise ValueError("楼层分区需要至少一个 traffic_core 或 core 结构区域")
    traffic_core = unary_union(traffic_parts).intersection(boundary)
    if traffic_core.is_empty or traffic_core.area <= 1e-9:
        raise ValueError("交通核区域无效或未落在建筑边界内")

    entrance = None
    raw_entrance = building.get("door_positions")
    if isinstance(raw_entrance, list) and len(raw_entrance) == 2:
        entrance = LineString([_finite_pair(point, "door_positions") for point in raw_entrance])

    floor_partition = config.setdefault("FloorPartition", {})
    program_type = str(floor_partition.get("program_type", "residential")).strip().lower() or "residential"
    if program_type != "residential":
        raise ValueError(f"当前版本仅支持 residential，收到 {program_type}")

    residential = floor_partition.setdefault("residential", {})
    raw_target_areas = [float(value) for value in residential.get("target_areas", [])]
    unit_count = int(residential.get("unit_count", len(raw_target_areas) or 1))
    if unit_count <= 0:
        raise ValueError("FloorPartition.residential.unit_count 必须大于 0")

    target_area_mode = str(residential.get("target_area_mode", "equal")).strip().lower() or "equal"
    if target_area_mode not in {"equal", "custom"}:
        raise ValueError("FloorPartition.residential.target_area_mode 只能是 equal 或 custom")
    if target_area_mode == "custom" and len(raw_target_areas) != unit_count:
        raise ValueError("自定义面积模式下，target_areas 数量必须与 unit_count 一致")

    grid_size = float(
        floor_partition.get(
            "grid_size",
            config.get("SeedGrowth", {}).get(
                "grid_size",
                config.get("AdaptiveReuseEnvironment", {}).get("grid_size", 0.5),
            ),
        )
    )
    if grid_size <= 0 or not math.isfinite(grid_size):
        raise ValueError("FloorPartition.grid_size 必须为正的有限数值")

    profile = ResidentialProfile(
        unit_count=unit_count,
        target_area_mode=target_area_mode,
        target_areas=tuple(raw_target_areas),
        corridor_width=float(residential.get("corridor_width", 1.5)),
        min_door_spacing=float(residential.get("min_door_spacing", 2.4)),
        door_width=float(residential.get("door_width", 0.9)),
        opening_width=float(residential.get("opening_width", 1.5)),
        opening_side=str(residential.get("opening_side", "auto")).strip().lower() or "auto",
        grid_size=grid_size,
        export_config_prefix=str(residential.get("export_config_prefix", "unit")).strip() or "unit",
    )
    if min(profile.corridor_width, profile.min_door_spacing, profile.door_width, profile.opening_width) <= 0:
        raise ValueError("走道、门间距、门宽和开口宽度都必须大于 0")
    if profile.opening_side not in {"auto", "north", "south", "east", "west"}:
        raise ValueError("opening_side 只能是 auto/north/south/east/west")

    if target_area_mode == "custom":
        requested = list(profile.target_areas)
    else:
        requested = [0.0] * unit_count
    targets = tuple(
        PartitionTarget(
            unit_id=f"{profile.export_config_prefix}_{index + 1:02d}",
            name=f"{profile.export_config_prefix}_{index + 1:02d}",
            requested_area=requested[index],
        )
        for index in range(unit_count)
    )
    return FloorPartitionProblem(
        boundary=boundary,
        fixed_objects=fixed_objects,
        fixed_union=fixed_union,
        traffic_core=traffic_core,
        entrance=entrance,
        program_type=program_type,
        profile=profile,
        targets=targets,
    )
