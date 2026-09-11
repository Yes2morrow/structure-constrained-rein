"""住区楼栋的几何数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, replace

from shapely.geometry import Polygon, box


@dataclass
class Building:
    """第一阶段使用的轴对齐矩形住宅楼栋。"""

    building_id: int
    building_type: str
    x1: float
    y1: float
    x2: float
    y2: float
    floors: int
    floor_height: float = 3.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def depth(self) -> float:
        return self.y2 - self.y1

    @property
    def footprint_area(self) -> float:
        return self.width * self.depth

    @property
    def gross_floor_area(self) -> float:
        return self.footprint_area * self.floors

    @property
    def height(self) -> float:
        return self.floors * self.floor_height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def get_polygon(self) -> Polygon:
        return box(self.x1, self.y1, self.x2, self.y2)

    def copy(self, **changes) -> "Building":
        return replace(self, **changes)

