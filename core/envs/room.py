""" 定义房间 """
from typing import Tuple, List, Optional
from dataclasses import dataclass, field
from shapely.geometry import Polygon, LineString
from enum import Enum
import math

__all__ = [
    "Enum", "RoomType", "Room", "name_transform",
    "RoomColor", "DoorColor", "MinAreas", "MaxAreasRatio",
    "ASPECT_RATIO_RANGES", "NESTING_PRIORITY", "NESTING_WEIGHTS",
    "NOISE_WEIGHTS", "MaxAreasValue"
]


## 房间类型
class RoomType(Enum):
    """ 房间类型 """
    LIVING_ROOM = 1 # 客厅
    BEDROOM = 2     # 卧室
    KITCHEN = 3     # 厨房
    BATHROOM = 4    # 浴室
    BALCONY = 5     # 阳台
    

def name_transform(name:str) -> str:
    """ 房间名称转换，英转中 """
    name = name.lower()
    return {
        "living_room": "客厅",
        "bedroom": "卧室",
        "kitchen": "厨房",
        "bathroom": "浴室",
        "balcony": "阳台"
    }.get(name, name)
    
    
## 房间颜色
RoomColor = {
    RoomType.LIVING_ROOM: "#f4f2e5",
    RoomType.BEDROOM: "#fdf4ab",
    RoomType.KITCHEN: "#ead8d6",
    RoomType.BATHROOM: "#cde9fc",
    RoomType.BALCONY: "#d0d887"
}


## 门颜色
DoorColor = "#ffdd00"


## 房间面积最小值
MinAreas = {
    RoomType.LIVING_ROOM: 10,
    RoomType.BEDROOM: 9,
    RoomType.KITCHEN: 4, 
    RoomType.BATHROOM: 4,
    RoomType.BALCONY: 4,
}


## 房间面积最大比例, 按整体区域的百分比计算
MaxAreasRatio = {
    RoomType.LIVING_ROOM: 0.4,
    RoomType.BEDROOM: 0.40,
    RoomType.KITCHEN: 0.25,
    RoomType.BATHROOM: 0.25,
    RoomType.BALCONY: 0.25,
}

MaxAreasValue = {
    RoomType.LIVING_ROOM: 40,
    RoomType.BEDROOM: 40,
    RoomType.KITCHEN: 25,
    RoomType.BATHROOM: 25,
    RoomType.BALCONY: 25,
}


## 房间长宽比范围
ASPECT_RATIO_RANGES = {
    RoomType.LIVING_ROOM: (1, 1.6),
    RoomType.BEDROOM: (1, 1.8),
    RoomType.KITCHEN: (1, 1.85),
    RoomType.BATHROOM: (1, 1.6), 
    RoomType.BALCONY: (1, 2.8),
}


## 嵌套优先级, 1最弱，5最强， 5级可以嵌套在1级中
NESTING_PRIORITY = {
    "LIVING_ROOM": 1,   # 客厅
    "KITCHEN": 2,       # 厨房
    "BALCONY": 3,       # 阳台
    "BATHROOM": 4,      # 浴室
    "BEDROOM": 5,       # 卧室
}


## 嵌套关系最大被允许侵占
NESTING_WEIGHTS = {
    "LIVING_ROOM": 0.25,   # 客厅
    "KITCHEN": 0.15,       # 厨房
    "BATHROOM": 0.10,      # 浴室
    "BALCONY": 0.30,       # 阳台
    "BEDROOM": 0,          # 卧室
}


## 噪声干扰度
NOISE_WEIGHTS = {
    "LIVING_ROOM": 5.5,   # 客厅
    "KITCHEN": 5.5,       # 厨房
    "BATHROOM": 2.5,      # 浴室
    "BALCONY": 0.0,       # 阳台
    "BEDROOM": 1.5,       # 卧室
}


@dataclass
class Room:
    """ 房间类 """
    room_type: RoomType  # 房间类型
    x1: float            # 房间左上角x坐标
    y1: float            # 房间左上角y坐标
    x2: float            # 房间右下角x坐标
    y2: float            # 房间右下角y坐标
    door_position: Optional[Tuple[float, float]] = None  # 门位置[x1, y1, x2, y2]
    _cached_geometry_signature: Optional[Tuple[float, float, float, float]] = field(default=None, init=False, repr=False)
    _cached_corners: Optional[List[Tuple[float, float]]] = field(default=None, init=False, repr=False)
    _cached_polygon: Optional[Polygon] = field(default=None, init=False, repr=False)
    _cached_area: Optional[float] = field(default=None, init=False, repr=False)
    _cached_center: Optional[Tuple[float, float]] = field(default=None, init=False, repr=False)
    
    @property
    def width(self) -> float:
        """ 房间宽度 """
        return round(self.x2 - self.x1, 1)
    
    @property
    def height(self) -> float:
        """ 房间高度 """
        return round(self.y2 - self.y1, 1)
    
    @property
    def area(self) -> float:
        """ 房间面积 """
        self._ensure_geometry_cache()
        return self._cached_area
    
    @property
    def aspect_ratio(self) -> float:
        """ 房间长宽比 """
        return max(self.width, self.height) / min(self.width, self.height)

    @property
    def center(self) -> Tuple[float, float]:
        """ 房间中心点 """
        self._ensure_geometry_cache()
        return self._cached_center
    
    def get_corners(self) -> List[Tuple[float, float]]:
        """ 获取四个角点坐标 """
        self._ensure_geometry_cache()
        return list(self._cached_corners)
        
    def get_polygon(self) -> Polygon:
        """ 获取多边形 """
        self._ensure_geometry_cache()
        return self._cached_polygon

    def _geometry_signature(self) -> Tuple[float, float, float, float]:
        """ 返回用于判断几何缓存是否失效的签名 """
        return (
            round(self.x1, 1),
            round(self.y1, 1),
            round(self.x2, 1),
            round(self.y2, 1),
        )

    def _ensure_geometry_cache(self) -> None:
        """ 在房间坐标未变化时复用几何结果，减少重复的 Shapely 构造 """
        signature = self._geometry_signature()
        if signature == self._cached_geometry_signature:
            return

        corners = [
            (signature[0], signature[1]),  # 左上角
            (signature[2], signature[1]),  # 右上角
            (signature[2], signature[3]),  # 右下角
            (signature[0], signature[3]),  # 左下角
        ]
        polygon = Polygon(corners)

        self._cached_geometry_signature = signature
        self._cached_corners = corners
        self._cached_polygon = polygon
        self._cached_area = round(polygon.area, 2)
        self._cached_center = (
            round((signature[0] + signature[2]) / 2, 1),
            round((signature[1] + signature[3]) / 2, 1),
        )
    
    def copy(self) -> 'Room':
        """创建房间的副本"""
        return Room(
            room_type=self.room_type,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2
        )
        
    def set_door_position(self, living_room_poly: Polygon, other_rooms: List['Room'] = None, boundary_polygon: Polygon = None) -> None:
        """ 计算门位置 - 优化版本 """
        corners = self.get_corners()
        edges = []
        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]
            edges.append((start, end, i))
        
        # 第一步：快速筛选候选边
        candidate_edges = self._quick_filter_edges(edges, living_room_poly, boundary_polygon, other_rooms)
        
        # 第二步：在候选边上寻找最佳门位置
        best_door_position = self._find_best_door_position(candidate_edges, living_room_poly, other_rooms, boundary_polygon)
        
        # 设置门位置
        if best_door_position:
            self.door_position = best_door_position
        else:
            # 退路：使用房间中心
            cx, cy = self.center
            self.door_position = (cx, cy, cx, cy)
        
        return self

    def _quick_filter_edges(self, edges, living_room_poly, boundary_polygon, other_rooms):
        """快速筛选候选边 - 修复版本"""
        candidate_edges = []
        
        for start, end, edge_idx in edges:
            edge_line = LineString([start, end])
            edge_length = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
            
            # 快速得分计算
            score = 0
            
            # 1. 与客厅的距离 - 最重要因素
            living_room_distance = edge_line.distance(living_room_poly)
            if living_room_distance < 0.1:
                score += 100  # 与客厅相邻，最高优先级
                # 即使贴墙，只要与客厅相邻就保留
                candidate_edges.append((edge_idx, start, end, edge_length, score, living_room_distance))
                continue
            
            # 2. 检查是否贴墙 - 不再直接排除，而是扣分
            wall_penalty = 0
            if boundary_polygon:
                wall_distance = edge_line.distance(boundary_polygon.exterior)
                if wall_distance < 0.1:
                    wall_penalty = -50  # 贴墙扣分，但不排除
            score += wall_penalty
            
            # 3. 检查是否贴其他房间 - 同样改为扣分而非排除
            other_room_penalty = 0
            if other_rooms:
                for other_room in other_rooms:
                    if other_room == self:
                        continue
                    other_distance = edge_line.distance(other_room.get_polygon())
                    if other_distance < 0.1:
                        other_room_penalty = -30  # 贴其他房间扣分
                        break
            score += other_room_penalty
            
            # 4. 边长度和距离得分
            score += edge_length * 5
            score += max(0, 50 - living_room_distance * 10)
            
            # 放宽阈值，允许更多边成为候选
            if score > -60:  # 降低阈值
                candidate_edges.append((edge_idx, start, end, edge_length, score, living_room_distance))
        
        # 按得分排序，优先选择得分高的边
        candidate_edges.sort(key=lambda x: x[4], reverse=True)
        
        # 确保至少有一条候选边
        if not candidate_edges and edges:
            # 如果没有候选边，选择与客厅距离最近的边
            min_distance = float('inf')
            fallback_edge = None
            for edge_idx, start, end, edge_length in [(e[2], e[0], e[1], math.sqrt((e[1][0]-e[0][0])**2 + (e[1][1]-e[0][1])**2)) for e in edges]:
                edge_line = LineString([start, end])
                distance = edge_line.distance(living_room_poly)
                if distance < min_distance:
                    min_distance = distance
                    fallback_edge = (edge_idx, start, end, edge_length, 0, distance)
            if fallback_edge:
                candidate_edges.append(fallback_edge)
        
        # 最多保留3条候选边进行详细测试
        return candidate_edges[:3]

    def _find_best_door_position(self, candidate_edges, living_room_poly, other_rooms, boundary_polygon):
        """在候选边上寻找最佳门位置"""
        best_door_position = None
        best_score = -float('inf')
        
        # 预计算其他房间的多边形，避免重复计算
        other_polygons = []
        if other_rooms:
            for other_room in other_rooms:
                if other_room != self:
                    other_polygons.append(other_room.get_polygon())
        
        # 预计算其他房间的门位置
        other_doors = []
        if other_rooms:
            for other_room in other_rooms:
                if other_room != self and other_room.door_position:
                    other_doors.append(LineString([
                        (other_room.door_position[0], other_room.door_position[1]),
                        (other_room.door_position[2], other_room.door_position[3])
                    ]))
        
        for edge_idx, start, end, edge_length, edge_score, living_room_distance in candidate_edges:
            # 计算边的方向向量
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            if edge_length > 0:
                dx, dy = dx/edge_length, dy/edge_length
            else:
                continue
            
            # 减少测试位置数量
            if living_room_distance < 0.1:
                # 与客厅相邻的边，测试更多位置
                test_positions = [0.2, 0.4, 0.5, 0.6, 0.8]
            else:
                # 其他边，只测试中间位置
                test_positions = [0.3, 0.5, 0.7]
            
            for pos_ratio in test_positions:
                # 计算门的位置
                door_len = min(0.8, edge_length * 0.8)
                door_center_x = start[0] + dx * edge_length * pos_ratio
                door_center_y = start[1] + dy * edge_length * pos_ratio
                
                door_x1 = door_center_x - dx * door_len/2
                door_y1 = door_center_y - dy * door_len/2
                door_x2 = door_center_x + dx * door_len/2
                door_y2 = door_center_y + dy * door_len/2
                
                # 创建门线段
                door_line = LineString([(door_x1, door_y1), (door_x2, door_y2)])
                
                # 快速计算位置得分
                position_score = self._calculate_door_position_score(
                    door_line, edge_score, living_room_poly, other_polygons, 
                    other_doors, boundary_polygon, pos_ratio
                )
                
                # 更新最佳位置
                if position_score > best_score:
                    best_score = position_score
                    best_door_position = (door_x1, door_y1, door_x2, door_y2)
        
        return best_door_position

    def _calculate_door_position_score(self, door_line, base_score, living_room_poly, 
                                    other_polygons, other_doors, boundary_polygon, pos_ratio):
        """快速计算门位置得分"""
        position_score = base_score
        
        # 1. 检查门是否被其他房间遮挡
        min_door_to_room_distance = float('inf')
        for other_poly in other_polygons:
            distance = door_line.distance(other_poly)
            if distance < min_door_to_room_distance:
                min_door_to_room_distance = distance
        
        # 距离越近，扣分越多
        if min_door_to_room_distance < 0.5:
            position_score -= (0.5 - min_door_to_room_distance) * 100
        
        # 2. 检查门是否被墙壁遮挡
        if boundary_polygon:
            wall_distance = door_line.distance(boundary_polygon.exterior)
            if wall_distance < 0.1:
                position_score -= 20
        
        # 3. 检查门是否与客厅有良好的连接
        living_room_distance = door_line.distance(living_room_poly)
        if living_room_distance < 0.1:
            position_score += 50  # 与客厅直接连接
        
        # 4. 检查门是否在边的中间位置
        distance_from_center = abs(pos_ratio - 0.5)
        position_score += 10 * (1 - distance_from_center)
        
        # 5. 检查门是否与其他房间的门太近
        for other_door in other_doors:
            door_to_door_distance = door_line.distance(other_door)
            if door_to_door_distance < 1.0:
                position_score -= (1.0 - door_to_door_distance) * 20
        
        return position_score
