"""住宅楼层分区的规则生长。"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

from shapely.geometry import Point, box
from shapely.ops import unary_union

from .circulation import CirculationLayout
from .contracts import FloorPartitionProblem, target_areas_for_area
from .doors import Door


@dataclass(frozen=True)
class PartitionCell:
    row: int
    column: int
    center: tuple[float, float]
    geometry: object
    area: float


@dataclass(frozen=True)
class PartitionResult:
    corridor: object
    opening: object
    opening_side: str
    allocatable_space: object
    doors: tuple[Door, ...]
    unit_polygons: dict[str, object]
    target_areas: dict[str, float]
    requested_areas: dict[str, float]
    area_scale: float


def _allocatable_space(problem: FloorPartitionProblem, circulation: CirculationLayout):
    geometry = problem.boundary.difference(problem.fixed_union)
    geometry = geometry.difference(circulation.corridor)
    if geometry.is_empty or geometry.area <= 1e-9:
        raise ValueError("走道与结构扣除后没有可分配空间")
    return geometry


def _grid_cells(problem: FloorPartitionProblem, allocatable_space) -> tuple[list[PartitionCell], dict[tuple[int, int], int]]:
    min_x, min_y, max_x, max_y = allocatable_space.bounds
    step = problem.profile.grid_size
    rows = int(math.ceil((max_y - min_y) / step))
    columns = int(math.ceil((max_x - min_x) / step))
    cells: list[PartitionCell] = []
    index_by_coord: dict[tuple[int, int], int] = {}
    for row in range(rows):
        for column in range(columns):
            cell_box = box(
                min_x + column * step,
                min_y + row * step,
                min_x + (column + 1) * step,
                min_y + (row + 1) * step,
            )
            clipped = allocatable_space.intersection(cell_box)
            if clipped.is_empty or clipped.area <= cell_box.area * 0.15:
                continue
            cell = PartitionCell(
                row=row,
                column=column,
                center=((cell_box.bounds[0] + cell_box.bounds[2]) / 2.0, (cell_box.bounds[1] + cell_box.bounds[3]) / 2.0),
                geometry=clipped,
                area=float(clipped.area),
            )
            index_by_coord[(row, column)] = len(cells)
            cells.append(cell)
    if not cells:
        raise ValueError("当前可分配空间无法按网格离散")
    return cells, index_by_coord


def _neighbors(cell: PartitionCell, index_by_coord: dict[tuple[int, int], int]) -> list[int]:
    result = []
    for delta_row, delta_column in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        key = (cell.row + delta_row, cell.column + delta_column)
        if key in index_by_coord:
            result.append(index_by_coord[key])
    return result


def _assign_sources(cells: list[PartitionCell], doors: list[Door]) -> dict[str, int]:
    taken: set[int] = set()
    result: dict[str, int] = {}
    for door in doors:
        order = sorted(
            range(len(cells)),
            key=lambda index: Point(cells[index].center).distance(Point(door.center)),
        )
        source = next((index for index in order if index not in taken), None)
        if source is None:
            raise ValueError("无法为每个门位分配唯一的生长起始网格")
        taken.add(source)
        result[door.unit_id] = source
    return result


def grow_residential_units(problem: FloorPartitionProblem, circulation: CirculationLayout, doors: list[Door]) -> PartitionResult:
    """从各户门位相邻网格出发，按面积配比生长并覆盖全部剩余空间。"""
    allocatable = _allocatable_space(problem, circulation)
    target_values, scale = target_areas_for_area(problem, allocatable.area)
    target_areas = {
        target.unit_id: target_values[index]
        for index, target in enumerate(problem.targets)
    }
    requested_areas = {
        target.unit_id: float(target.requested_area)
        for target in problem.targets
    }
    cells, index_by_coord = _grid_cells(problem, allocatable)
    neighbors = {
        index: _neighbors(cell, index_by_coord)
        for index, cell in enumerate(cells)
    }
    sources = _assign_sources(cells, doors)

    claimed: dict[int, str] = {}
    claimed_cells = {door.unit_id: set() for door in doors}
    claimed_area = {door.unit_id: 0.0 for door in doors}
    queue: list[tuple[float, float, int, str, int]] = []

    def push_neighbors(unit_id: str, source_index: int) -> None:
        for neighbor in neighbors[source_index]:
            if neighbor in claimed:
                continue
            fill_ratio = claimed_area[unit_id] / max(target_areas[unit_id], 1e-9)
            distance = Point(cells[neighbor].center).distance(Point(next(door.center for door in doors if door.unit_id == unit_id)))
            heapq.heappush(queue, (fill_ratio, distance, neighbor, unit_id, source_index))

    for door in doors:
        source = sources[door.unit_id]
        claimed[source] = door.unit_id
        claimed_cells[door.unit_id].add(source)
        claimed_area[door.unit_id] += cells[source].area

    for door in doors:
        push_neighbors(door.unit_id, sources[door.unit_id])

    while len(claimed) < len(cells):
        if not queue:
            remaining = [index for index in range(len(cells)) if index not in claimed]
            for index in remaining:
                unit_id = min(
                    claimed_area,
                    key=lambda key: (
                        claimed_area[key] / max(target_areas[key], 1e-9),
                        Point(cells[index].center).distance(Point(next(door.center for door in doors if door.unit_id == key))),
                    ),
                )
                claimed[index] = unit_id
                claimed_cells[unit_id].add(index)
                claimed_area[unit_id] += cells[index].area
            break
        _, _, index, unit_id, parent = heapq.heappop(queue)
        if index in claimed or parent not in claimed_cells[unit_id]:
            continue
        claimed[index] = unit_id
        claimed_cells[unit_id].add(index)
        claimed_area[unit_id] += cells[index].area
        push_neighbors(unit_id, index)

    unit_polygons = {}
    for unit_id, indices in claimed_cells.items():
        geometry = unary_union([cells[index].geometry for index in sorted(indices)])
        if geometry.is_empty or geometry.area <= 1e-9:
            raise ValueError(f"{unit_id} 没有生成有效户型边界")
        unit_polygons[unit_id] = geometry.buffer(0)

    return PartitionResult(
        corridor=circulation.corridor,
        opening=circulation.opening,
        opening_side=circulation.opening_side,
        allocatable_space=allocatable,
        doors=tuple(doors),
        unit_polygons=unit_polygons,
        target_areas=target_areas,
        requested_areas=requested_areas,
        area_scale=float(scale),
    )
