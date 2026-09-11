from typing import Tuple, List, Optional, Dict
import os
import random
import math
import time
import glob
import numpy as np
import shapely.geometry as sg
from shapely.geometry import LineString
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import gym
from gym import spaces

from .room import *

import warnings
warnings.filterwarnings('ignore')


class HouseEnv(gym.Env):
    """ 房屋环境类 """
    def __init__(
        self, 
        grid_size:int=15,
        grid_resolution:float=0.2,
        shift_value:float=0.1,
        num_agents:int=4,
        method:str="base",
        language:str="zh",
        show_prior:bool=False,
        init_width:int=1,
        init_height:int=1,
        use_area_ratio:bool=False,
        prior_interval:float=3.0,
        prior_layout_scale:float=2.0,
        **kwargs
    ) -> None:
        super(HouseEnv, self).__init__()
        """ 
        Args:
            grid_size (int, optional): 网格大小. Defaults to 15.
            grid_resolution (float, optional): 网格分辨率. Defaults to 0.2.
            shift_value (float, optional): 布局偏移量. Defaults to 0.1.
            num_agents (int, optional): 智能体数量. Defaults to 4.
            method (str, optional): 布局方法. Defaults to "base".
        """
        # 基础参数
        self.grid_size = grid_size
        self.grid_resolution = grid_resolution
        self.shift_value = shift_value
        self.num_agents = num_agents
        self.method = method
        self.language = language
        self.show_prior = show_prior
        self.init_width = init_width
        self.init_height = init_height
        self.use_area_ratio = use_area_ratio
        self.prior_layout_scale = float(prior_layout_scale)
        self.prior_interval = float(prior_interval)
        self.environment_advanced = dict(kwargs.get("environment_advanced", {}) or {})
        self.reward_advanced = dict(kwargs.get("reward_advanced", {}) or {})
        
        if self.language.lower() == "zh":
            # windows
            plt.rcParams['font.sans-serif'] = ['SimHei']  
            plt.rcParams['axes.unicode_minus'] = False
        else:
            # windows
            plt.rcParams['font.sans-serif'] = ['Times New Roman']  
            plt.rcParams['axes.unicode_minus'] = False

        # 定义边界
        self.boundary = None
        self.boundary_polygon = None
        self.entrance_door = None
        
        # 先验布局
        self.prior_positions = None  # 先验位置 [(cx1, cy1), (cx2, cy2), (cx3, cy3), (cx4, cy4)]
        self.prior_thresholds = float(self.environment_advanced.get("prior_match_threshold", 1.0))
        self.fixed_position = True
        self.prior_limit_area = None  # 先验限制区域
        
        # 房间
        self.rooms:List[Room] = []
        self.room_type:List[Enum] = [
            RoomType.BEDROOM,   # 卧室
            RoomType.KITCHEN,   # 厨房
            RoomType.BATHROOM,  # 浴室
            RoomType.BALCONY,   # 阳台
        ]
        self.living_room:Optional[Room] = None  # 客厅
        
        # 运动方向(5): 上，下，左，右，不动
        # 伸缩方向(8): 上宽增加, 上宽减少, 下宽增加, 下宽减少, 左长增加, 左长减少, 右长增加, 右长减少
        action_num = 5 + 8
        # 动作空间
        self.action_space = spaces.MultiDiscrete(self.num_agents * [action_num])
        # 状态空间
        self.observation_space = spaces.Box(
            low=0.0, high=grid_size,
            shape=(20, ), dtype=np.float32)
        
        self.set_extra_reward_weight()

    def _env_param(self, key: str, default):
        """读取环境高级参数。"""
        return self.environment_advanced.get(key, default)

    def _reward_param(self, key: str, default):
        """读取奖励高级参数。"""
        return self.reward_advanced.get(key, default)
    
    def set_boundary(self, points:List[Tuple]=None, left_points:Tuple=None, right_points:Tuple=None) -> None:
        """ 设置边界 """
        self.boundary = points
        self.boundary_polygon = sg.Polygon(points)
        door_center_offset = float(self._env_param("door_center_offset", 0.6))
        vertical_door_default_length = float(self._env_param("vertical_door_default_length", 1.2))
        horizontal_door_default_length = float(self._env_param("horizontal_door_default_length", 1.0))
        
        if right_points is None and left_points is None:
            # 设置外门为右侧边界的中间位置
            right_side = max([p[0] for p in points])
            right_points = [p for p in points if abs(p[0] - right_side) < 0.1]
            if len(right_points) < 2:
                raise ValueError("无法找到合适的门位置")
            right_points.sort(key=lambda p: p[1])
            mid_y = (right_points[0][1] + right_points[-1][1]) / 2
            door_y = mid_y - door_center_offset
            self.entrance_door = (right_side, door_y, right_side, door_y + vertical_door_default_length)
            self.entrance_door_center = (right_side, door_y + door_center_offset)
        else:
            if left_points is not None:
                mid_y = (left_points[0][1] + left_points[-1][1]) / 2
                door_y = mid_y - door_center_offset
                self.entrance_door = (left_points[0][0], door_y, left_points[-1][0], door_y + vertical_door_default_length)
                self.entrance_door_center = (left_points[0][0] + door_center_offset, door_y + door_center_offset)
            else:
                mid_x = (right_points[0][0] + right_points[-1][0]) / 2
                door_x = mid_x - door_center_offset
                self.entrance_door = (door_x, right_points[0][1], door_x + horizontal_door_default_length, right_points[-1][1])
                self.entrance_door_center = (door_x + door_center_offset, right_points[0][1])

    def set_prior_positions(
        self, prior_positions: List[Tuple[float, float]], 
        randvalue:float=0.01,
        randomize:bool=False
    ):
        """ 设置先验知识布局位置 """
        if not randomize:
            prior = []
            for pos in prior_positions:
                x, y = pos
                new_x, new_y = round(random.uniform(x - randvalue, x + randvalue), 1), \
                            round(random.uniform(y - randvalue, y + randvalue), 1)
                # 判断位置是否越界
                if self.boundary_polygon.contains(sg.Point(new_x, new_y)):
                    prior.append((new_x, new_y))
                else:
                    prior.append(pos)
            self.prior_positions = prior
        else:
            # 随机生成先验位置
            self.prior_positions = self._generate_random_prior_positions()
    
    def set_prior_limit_area(self, arae:List[Tuple[float, float]] = []):
        """ 设置先验位置限制区域 """
        self.prior_limit_area = arae
    
    def set_fixed_position(self, fixed_position:bool) -> None:
        """ 设置是否固定先验位置 """
        self.fixed_position = fixed_position
    
    def set_room_type(self, room_type:List[RoomType]):
        """ 设置房间类型 """
        self.room_type = room_type
    
    def set_extra_reward_weight(self, value:Dict = None) -> None:
        """ 设置额外奖励权重 """
        if value is not None:
            self.extra_reward_weight = value
        else:
            self.extra_reward_weight = {
                "邻接关系奖励": 1.0,  # 房间面积
                "噪声干扰度奖励": 1.0,  # 房间距离
                "隐私保护奖励": 1.0,  # 房间重叠
                "空间开放性奖励": 0,  # 房间边界
                "南向采光奖励": 1.0,  # 房间入口
                "最短路径奖励": 1.0,  # 房间限制区域
            }
    
    def set_base_reward_weight(self, value:Dict = None) -> None:
        """ 设置基础奖励权重 """
        if value is not None:
            self.base_reward_weight = value
        else:
            self.base_reward_weight = {
                "面积奖励": 1.0,  
                "长宽比奖励": 1.0,  
                "贴边奖励": 1.0,  
                "房间与房间之间的边界贴合奖励": 1.0, 
                "角落占领奖励": 1.0,  
                "外门遮挡惩罚": 1.0,  
                "先验知识布局奖励": 1.0,  
                "房间嵌套惩罚": 1.0,
            }

    def _get_boundary_area(self) -> float:
        """获取当前轮廓面积。"""
        if self.boundary_polygon is None:
            return 0.0
        return float(self.boundary_polygon.area)

    def _get_room_max_area(self, room: Room) -> float:
        """根据当前配置计算房间最大面积。"""
        if self.use_area_ratio:
            ratio = float(MaxAreasRatio.get(room.room_type, 0.4))
            return max(0.1, self._get_boundary_area() * ratio)
        return max(0.1, float(MaxAreasValue.get(room.room_type, 10.0)))
        
    def _generate_random_prior_positions(self) -> List[Tuple[float, float]]:
        """随机生成先验位置，满足所有约束条件，且位置间至少间隔3米"""
        positions = []
        max_attempts = int(self._env_param("prior_random_max_attempts", 3000))
        wall_clearance = float(self._env_param("prior_random_wall_clearance", 0.8))
        door_avoidance = float(self._env_param("prior_random_door_avoidance", 1.2))
        corner_region_size = float(self._env_param("prior_corner_region_size", 1.5))
        corner_attempt_limit = int(self._env_param("prior_corner_attempt_limit", 50))
        distance_relax_attempts = int(self._env_param("prior_distance_relax_attempts", 100))
        distance_requirements = list(self._env_param("prior_distance_relaxation", [2.5, 2.0, 1.5, 1.0, 0.8]))
        
        # 获取边界范围
        min_x, min_y, max_x, max_y = self.boundary_polygon.bounds
        
        # 计算有效区域
        valid_min_x = min_x + wall_clearance
        valid_min_y = min_y + wall_clearance
        valid_max_x = max_x - wall_clearance
        valid_max_y = max_y - wall_clearance
        
        # 计算外门附近需要避开的区域
        door_avoid_zone = None
        if self.entrance_door:
            door_x1, door_y1, door_x2, door_y2 = self.entrance_door
            door_center_x = (door_x1 + door_x2) / 2
            door_center_y = (door_y1 + door_y2) / 2
            
            # 创建外门附近避让区域
            door_avoid_zone = sg.Polygon([
                (door_center_x - door_avoidance, door_center_y - door_avoidance),
                (door_center_x + door_avoidance, door_center_y - door_avoidance),
                (door_center_x + door_avoidance, door_center_y + door_avoidance),
                (door_center_x - door_avoidance, door_center_y + door_avoidance)
            ])
        
        # 创建先验限制区域多边形（如果存在）
        limit_polygon = None
        if self.prior_limit_area and len(self.prior_limit_area) >= 3:
            limit_polygon = sg.Polygon(self.prior_limit_area)
        
        # 定义角落区域（优先在这些区域生成位置）
        corner_regions = [
            # 四个角落区域
            (valid_min_x, valid_min_y, valid_min_x + corner_region_size, valid_min_y + corner_region_size),
            (valid_max_x - corner_region_size, valid_min_y, valid_max_x, valid_min_y + corner_region_size),
            (valid_min_x, valid_max_y - corner_region_size, valid_min_x + corner_region_size, valid_max_y),
            (valid_max_x - corner_region_size, valid_max_y - corner_region_size, valid_max_x, valid_max_y),
        ]
        
        # 尝试为每个房间生成位置
        for room_idx in range(self.num_agents):
            attempts = 0
            position_found = False
            
            while not position_found and attempts < max_attempts:
                attempts += 1
                
                # 优先在角落区域生成位置
                if attempts <= corner_attempt_limit and room_idx < len(corner_regions):
                    # 前50次尝试优先使用角落区域
                    corner_idx = room_idx % len(corner_regions)
                    region = corner_regions[corner_idx]
                    x = round(random.uniform(region[0], region[2]), 1)
                    y = round(random.uniform(region[1], region[3]), 1)
                else:
                    # 随机在有效区域内生成
                    x = round(random.uniform(valid_min_x, valid_max_x), 1)
                    y = round(random.uniform(valid_min_y, valid_max_y), 1)
                
                candidate_point = sg.Point(x, y)
                
                # 检查约束条件
                valid_position = True
                
                # 1. 检查是否在边界内
                if not self.boundary_polygon.contains(candidate_point):
                    valid_position = False
                    continue
                
                # 2. 检查是否距离墙壁足够远
                distance_to_boundary = self.boundary_polygon.exterior.distance(candidate_point)
                if distance_to_boundary < wall_clearance:
                    valid_position = False
                    continue
                
                # 3. 检查是否在避开门区域
                if door_avoid_zone and door_avoid_zone.contains(candidate_point):
                    valid_position = False
                    continue
                
                # 4. 检查是否在先验限制区域内（如果设置了限制区域）
                if limit_polygon and limit_polygon.contains(candidate_point):
                    valid_position = False
                    continue
                
                # 5. 检查与已有位置的最小距离 - 修改为3米
                min_distance_to_others = float('inf')
                for existing_pos in positions:
                    dist = math.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2)
                    min_distance_to_others = min(min_distance_to_others, dist)
                
                if positions and min_distance_to_others < self.prior_interval:
                    valid_position = False
                    continue
                
                # 如果所有条件都满足，添加位置
                if valid_position:
                    positions.append((x, y))
                    position_found = True
                    break
            
            # 如果无法找到满足所有条件的位置，放宽距离条件
            if not position_found:
                for required_distance in distance_requirements:
                    for attempt in range(distance_relax_attempts):
                        x = round(random.uniform(valid_min_x, valid_max_x), 1)
                        y = round(random.uniform(valid_min_y, valid_max_y), 1)
                        candidate_point = sg.Point(x, y)
                        
                        # 基本检查
                        if (not self.boundary_polygon.contains(candidate_point) or
                            (door_avoid_zone and door_avoid_zone.contains(candidate_point)) or
                            (limit_polygon and limit_polygon.contains(candidate_point))):
                            continue
                        
                        # 检查与已有位置的距离（使用当前要求的最小距离）
                        min_distance_to_others = float('inf')
                        for existing_pos in positions:
                            dist = math.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2)
                            min_distance_to_others = min(min_distance_to_others, dist)
                        
                        if not positions or min_distance_to_others >= required_distance:
                            positions.append((x, y))
                            position_found = True
                            break
                    
                    if position_found:
                        break
        
        return positions
    
    def _is_room_in_boundary(self, room: 'Room') -> bool:
        """检查房间是否完全在边界多边形内"""
        room_poly = room.get_polygon()
        return self.boundary_polygon.contains(room_poly)
    
    def _get_room_move_direction(self, room: 'Room', dx: float, dy: float) -> Tuple[float, float]:
        """获取房间在边界内的有效移动方向"""
        # 尝试移动房间
        test_room = room.copy()
        test_room.x1 += dx
        test_room.x2 += dx
        test_room.y1 += dy
        test_room.y2 += dy
        
        # 如果移动后在边界内，返回原始移动向量
        if self._is_room_in_boundary(test_room):
            return dx, dy
        
        # 否则尝试减小移动步长
        scale_factors = list(self._env_param("boundary_retry_scale_factors", [0.8, 0.6, 0.4, 0.2, 0.1, 0.05]))
        for scale in scale_factors:
            test_room = room.copy()
            test_room.x1 += dx * scale
            test_room.x2 += dx * scale
            test_room.y1 += dy * scale
            test_room.y2 += dy * scale
            
            if self._is_room_in_boundary(test_room):
                return dx * scale, dy * scale
        
        # 如果所有尝试都失败，返回零移动
        return 0.0, 0.0
    
    def _get_room_resize_direction(self, room: 'Room', dx1: float, dy1: float, dx2: float, dy2: float) -> Tuple[float, float, float, float]:
        """获取房间在边界内的有效调整大小方向"""
        # 尝试调整房间大小
        test_room = room.copy()
        test_room.x1 += dx1
        test_room.x2 += dx2
        test_room.y1 += dy1
        test_room.y2 += dy2
        
        # 确保房间坐标正确排序
        test_room.x1, test_room.x2 = sorted([test_room.x1, test_room.x2])
        test_room.y1, test_room.y2 = sorted([test_room.y1, test_room.y2])
        
        # 如果调整后在边界内，返回原始调整向量
        if self._is_room_in_boundary(test_room):
            return dx1, dy1, dx2, dy2
        
        # 否则尝试减小调整步长
        scale_factors = list(self._env_param("boundary_retry_scale_factors", [0.8, 0.6, 0.4, 0.2, 0.1, 0.05]))
        for scale in scale_factors:
            test_room = room.copy()
            test_room.x1 += dx1 * scale
            test_room.x2 += dx2 * scale
            test_room.y1 += dy1 * scale
            test_room.y2 += dy2 * scale
            
            # 确保房间坐标正确排序
            test_room.x1, test_room.x2 = sorted([test_room.x1, test_room.x2])
            test_room.y1, test_room.y2 = sorted([test_room.y1, test_room.y2])
            
            if self._is_room_in_boundary(test_room):
                return dx1 * scale, dy1 * scale, dx2 * scale, dy2 * scale
        
        # 如果所有尝试都失败，返回零调整
        return 0.0, 0.0, 0.0, 0.0
        
    def _create_living_room(self, rnd:bool=False) -> None:
        """ 创建客厅 """
        # 获取边界范围
        min_x, min_y, max_x, max_y = self.boundary_polygon.bounds
        # 以门中心的位置的y, 为客厅中心位置y
        door_y = self.entrance_door_center[1]
        # 宽为整个区域的宽度
        width = max_x - min_x
        # 计算场地总面积
        boundary_area = self.boundary_polygon.area
        # 客厅面积范围
        min_area = boundary_area * float(self._env_param("living_room_min_ratio", 0.3))
        max_area = boundary_area * float(self._env_param("living_room_max_ratio", 0.5))
        # 根据面积确定客厅的高度
        min_height = min_area / width
        max_height = max_area / width
        # 随机取高度
        if rnd:
            height = round(np.random.uniform(min_height, max_height), 1)
        else:
            height = round((min_height + max_height) / 2, 1)
        # 以外门的长度为界限, 随机调整客厅中心位置y
        if rnd:
            door_y = round(np.random.uniform(door_y - height / 4, door_y + height / 4), 1)
        
        # 设置客厅初始化
        self.living_room = Room(
            room_type=RoomType.LIVING_ROOM,
            x1=min_x,               # 左上x
            y1=door_y + height / 2, # 左上y
            x2=max_x,               # 右下x
            y2=door_y - height / 2, # 右下y
        )
        return self
        
    def _cal_inner_doors_position(self) -> None:
        """计算内门位置 - 确保内门开向客厅"""
        # 第一遍：初步计算所有门位置
        for room in self.rooms:
            other_rooms = [r for r in self.rooms if r != room]
            room.set_door_position(
                self.living_room.get_polygon(),
                other_rooms=other_rooms,
                boundary_polygon=self.boundary_polygon
            )
        
        # 第二遍：验证和修复门位置
        for room in self.rooms:
            self._validate_and_fix_door_direction(room)
        
        # 第三遍：确保所有房间都有有效的门位置
        for room in self.rooms:
            self._ensure_valid_door_position(room)
    
    def _force_set_door_position(self, room: Room) -> None:
        """强制设置门位置"""
        corners = room.get_corners()
        
        # 找出所有可能的边
        candidate_edges = []
        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]
            edge_length = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
            candidate_edges.append((start, end, i, edge_length))
        
        # 选择最长的边
        candidate_edges.sort(key=lambda x: x[3], reverse=True)
        
        if candidate_edges:
            start, end, edge_idx, edge_length = candidate_edges[0]
            
            # 计算边的方向向量
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            if edge_length > 0:
                dx, dy = dx/edge_length, dy/edge_length
            
            # 在边的中间设置门
            door_len = min(0.8, edge_length * 0.8)
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            
            door_x1 = mid_x - dx * door_len/2
            door_y1 = mid_y - dy * door_len/2
            door_x2 = mid_x + dx * door_len/2
            door_y2 = mid_y + dy * door_len/2
            
            room.door_position = (door_x1, door_y1, door_x2, door_y2)

    def _ensure_valid_door_position(self, room: Room) -> None:
        """确保房间有有效的门位置"""
        if not room.door_position:
            # 如果没有门位置，强制设置一个
            self._force_set_door_position(room)
            return
        
        # 检查门位置是否有效（不是单点）
        dx1, dy1, dx2, dy2 = room.door_position
        if dx1 == dx2 and dy1 == dy2:
            # 门位置是单点，需要修复
            self._force_set_door_position(room)
    
    def _get_allow_actions(self) -> List[List[int]]:
        """ 为了减轻模型训练步数加速收敛, 获取每个智能体允许的动作列表
        规则:
            1. 房间先移动到边界，没有碰到边界前，不允许调整宽高
            2. 房间移动到了边界, 允许沿着边界移动, 并且没有贴着边界的地方可以伸缩边长宽
            3. 如果房间智能体贴边，只能按边的方向移动：
                a. 智能体贴住上边界/下边界，则只能左右移动；
                b. 智能体贴住左边界/右边界，则只能上下移动；
                c. 如果贴住了角，则不能移动了；
        """
        allowed_actions = []
        tolerance = 0.05  # 容差，用于判断是否接触边界

        # 获取边界范围
        min_x, min_y, max_x, max_y = self.boundary_polygon.bounds
        for room in self.rooms:
            # 初始化允许的动作列表，全部不允许
            room_allowed = [0] * 13  # 0-12共13个动作
            
            # 检查房间是否接触边界
            touches_left = abs(room.x1 - min_x) < tolerance
            touches_right = abs(room.x2 - max_x) < tolerance
            touches_bottom = abs(room.y1 - min_y) < tolerance
            touches_top = abs(room.y2 - max_y) < tolerance
            
            # 检查是否贴角（同时接触多个边界）
            touches_corner = (touches_left and touches_top) or \
                            (touches_left and touches_bottom) or \
                            (touches_right and touches_top) or \
                            (touches_right and touches_bottom) or \
                            (touches_left and touches_top and touches_bottom) or \
                            (touches_left and touches_right and touches_top) or \
                            (touches_left and touches_right and touches_bottom) or \
                            (touches_right and touches_top and touches_bottom) or \
                            (touches_left and touches_top and touches_bottom and touches_right)
                            
            if not touches_corner:
                # 房间没有贴角，允许移动
                if touches_left or touches_right:  # 智能体贴住左/右边界， 上下移动
                    room_allowed[0] = 1  # 上
                    room_allowed[1] = 1  # 下
                    room_allowed[5]  = 1  # 上边增加
                    room_allowed[6]  = 1  # 上边减少
                    room_allowed[7]  = 1  # 下边增加
                    room_allowed[8]  = 1  # 下边减少
                    room_allowed[9]  = 1  # 左边增加
                    room_allowed[10] = 1  # 左边减少
                    room_allowed[11] = 1  # 右边增加
                    room_allowed[12] = 1  # 右边减少
                elif touches_top or touches_bottom:  # 智能体贴住上/下边界， 左右移动
                    room_allowed[2] = 1  # 左
                    room_allowed[3] = 1  # 右
                    room_allowed[5]  = 1  # 上边增加
                    room_allowed[6]  = 1  # 上边减少
                    room_allowed[7]  = 1  # 下边增加
                    room_allowed[8]  = 1  # 下边减少
                    room_allowed[9]  = 1  # 左边增加
                    room_allowed[10] = 1  # 左边减少
                    room_allowed[11] = 1  # 右边增加
                    room_allowed[12] = 1  # 右边减少
                else:  # 智能体没有贴住边界，可以上下左右移动
                    for i in range(4):
                        room_allowed[i] = 1
            else:
                # 房间贴角，不允许移动
                if touches_left and touches_top:  # 智能体贴住左上角
                    room_allowed[1] = 1  # 下
                    room_allowed[3] = 1  # 右
                    room_allowed[11] = 1  # 右边增加
                    room_allowed[12] = 1  # 右边减少
                    room_allowed[7]  = 1  # 下边增加
                    room_allowed[8]  = 1  # 下边减少
                elif touches_left and touches_bottom:  # 智能体贴住左下角
                    room_allowed[0] = 1  # 上
                    room_allowed[3] = 1  # 右
                    room_allowed[11] = 1  # 右边增加
                    room_allowed[12] = 1  # 右边减少
                    room_allowed[5]  = 1  # 上边增加
                    room_allowed[6]  = 1  # 上边减少
                elif touches_right and touches_top:  # 智能体贴住右上角
                    room_allowed[1] = 1  # 下
                    room_allowed[2] = 1  # 左
                    room_allowed[9]  = 1  # 左边增加
                    room_allowed[10] = 1  # 左边减少
                    room_allowed[7]  = 1  # 下边增加
                    room_allowed[8]  = 1  # 下边减少
                elif touches_right and touches_bottom:  # 智能体贴住右下角
                    room_allowed[0] = 1  # 上
                    room_allowed[2] = 1  # 左
                    room_allowed[9]  = 1  # 左边增加
                    room_allowed[10] = 1  # 左边减少
                    room_allowed[5]  = 1  # 上边增加
                    room_allowed[6]  = 1  # 上边减少
                elif touches_left and touches_right and touches_top: # 贴左右上
                    room_allowed[1] = 1  # 下
                    room_allowed[7]  = 1  # 下边增加
                    room_allowed[8]  = 1  # 下边减少
                elif touches_left and touches_right and touches_bottom: # 贴左右下
                    room_allowed[0] = 1  # 上
                    room_allowed[5]  = 1  # 上边增加
                    room_allowed[6]  = 1  # 上边减少
                elif touches_top and touches_bottom and touches_left: # 贴上下左
                    room_allowed[3] = 1  # 右
                    room_allowed[11] = 1  # 右边增加
                    room_allowed[12] = 1  # 右边减少
                elif touches_top and touches_bottom and touches_right: # 贴上下右
                    room_allowed[2] = 1  # 左
                    room_allowed[9]  = 1  # 左边增加
                    room_allowed[10] = 1  # 左边减少
                    
            # 将允许的动作转换为列表格式
            act = []
            for i in range(len(room_allowed)):
                if room_allowed[i] == 1:
                    act.append(i)
            allowed_actions.append(act)

        return allowed_actions
    
    def _get_door_edge(self, room: Room, door_center: Tuple[float, float]) -> str:
        """确定门靠近房间的哪一边"""
        tolerance = 0.1  # 容差
        
        if abs(door_center[0] - room.x1) < tolerance:
            return "left"
        elif abs(door_center[0] - room.x2) < tolerance:
            return "right"
        elif abs(door_center[1] - room.y1) < tolerance:
            return "bottom"
        elif abs(door_center[1] - room.y2) < tolerance:
            return "top"
        else:
            # 如果无法确定，返回中心点距离最近的一边
            dist_left = abs(door_center[0] - room.x1)
            dist_right = abs(door_center[0] - room.x2)
            dist_bottom = abs(door_center[1] - room.y1)
            dist_top = abs(door_center[1] - room.y2)
            
            min_dist = min(dist_left, dist_right, dist_bottom, dist_top)
            if min_dist == dist_left:
                return "left"
            elif min_dist == dist_right:
                return "right"
            elif min_dist == dist_bottom:
                return "bottom"
            else:
                return "top"
            
    def _cal_prior_layout_reward(self) -> List[float]:
        """计算先验知识布局奖励
        规则：
        1. 每个房间到达预定先验位置给予正向奖励
        2. 同一个先验位置被多个房间占据时给予惩罚
        3. 距离越近奖励越高，距离越远惩罚越大
        4. 新增：如果南面只有一个先验位置，卧室朝南移动得到正向奖励，其他房间朝南去逼近其他的先验位置
        """
        if not self.prior_positions:
            return [0.0] * len(self.rooms)
        
        rewards = [0.0] * len(self.rooms)
        position_occupancy = [0] * len(self.prior_positions)  # 记录每个先验位置的占据情况

         # 动态调整阈值，适应不同布局尺度
        boundary_bounds = self.boundary_polygon.bounds
        diagonal = math.sqrt((boundary_bounds[2]-boundary_bounds[0])**2 + 
                            (boundary_bounds[3]-boundary_bounds[1])**2)
        
        # 根据边界大小动态调整匹配阈值
        dynamic_threshold = diagonal * 0.1  # 边界对角线的10%
        
        # 第一步：计算每个房间到各个先验位置的距离，并确定最近的位置
        room_to_prior_distances = []
        for i, room in enumerate(self.rooms):
            cx, cy = room.center
            distances = []
            for j, (prior_cx, prior_cy) in enumerate(self.prior_positions):
                distance = math.sqrt((cx - prior_cx)**2 + (cy - prior_cy)**2)
                distances.append((j, distance))
            # 按距离排序，找到最近的位置
            distances.sort(key=lambda x: x[1])
            room_to_prior_distances.append(distances)
            
            # 如果距离在阈值内，标记该位置被占据
            if distances[0][1] <= self.prior_thresholds:
                position_occupancy[distances[0][0]] += 1
        
        # 第二步：分析南向先验位置
        min_x, min_y, max_x, max_y = self.boundary_polygon.bounds
        south_threshold = min_y + (max_y - min_y) * 0.3  # 定义南向区域（下方30%的区域）
        
        # 找出所有南向的先验位置
        south_prior_positions = []
        for j, (prior_cx, prior_cy) in enumerate(self.prior_positions):
            if prior_cy <= south_threshold:
                south_prior_positions.append(j)
        
        # 南向奖励逻辑：如果只有一个南向先验位置
        south_prior_rewards = [0.0] * len(self.rooms)
        if len(south_prior_positions) == 1:
            south_prior_idx = south_prior_positions[0]
            
            for i, room in enumerate(self.rooms):
                closest_prior_idx, closest_distance = room_to_prior_distances[i][0]
                cx, cy = room.center
                
                if room.room_type == RoomType.BEDROOM:
                    # 卧室：如果最近的是南向先验位置，给予奖励
                    if closest_prior_idx == south_prior_idx:
                        # 距离越近奖励越高
                        max_possible_distance = math.sqrt((self.grid_size)**2 + (self.grid_size)**2)
                        normalized_distance = closest_distance / max_possible_distance
                        south_reward = dynamic_threshold * math.exp(-5 * normalized_distance)  # 南向奖励
                        south_prior_rewards[i] = south_reward
                    else:
                        # 卧室应该尽量朝南，如果不在南向位置，给予轻微惩罚
                        # 距离越近奖励越高
                        max_possible_distance = math.sqrt((self.grid_size)**2 + (self.grid_size)**2)
                        normalized_distance = closest_distance / max_possible_distance
                        south_penalty = -dynamic_threshold * (1 - math.exp(-2 * normalized_distance))
                        south_prior_rewards[i] = south_penalty
                else:
                    # 其他房间：如果不是卧室且占据了南向唯一位置，给予惩罚
                    if closest_prior_idx == south_prior_idx:
                        # 距离越近奖励越高
                        max_possible_distance = math.sqrt((self.grid_size)**2 + (self.grid_size)**2)
                        normalized_distance = closest_distance / max_possible_distance
                        south_penalty = -dynamic_threshold * (1 - math.exp(-2 * normalized_distance))
                        south_prior_rewards[i] = south_penalty
                    else:
                        # 其他房间朝自己的先验位置移动，不额外奖励也不惩罚
                        south_prior_rewards[i] = 0.0
        
        # 第三步：计算每个房间的基础奖励
        for i, room in enumerate(self.rooms):
            closest_prior_idx, closest_distance = room_to_prior_distances[i][0]
            
            # 基础距离奖励：距离越近奖励越高
            max_possible_distance = math.sqrt((self.grid_size)**2 + (self.grid_size)**2)
            normalized_distance = closest_distance / max_possible_distance
            
            # 使用指数衰减函数计算距离奖励
            distance_reward = dynamic_threshold * math.exp(-5 * normalized_distance)
            
            # 位置独占性惩罚：如果多个房间占据同一个位置
            occupancy_penalty = 0.0
            if position_occupancy[closest_prior_idx] > 1:
                # 惩罚与占据同一位置的其他房间数量成正比
                occupancy_penalty = -dynamic_threshold * (position_occupancy[closest_prior_idx] - 1)
            
            # 总奖励 = 基础距离奖励 + 独占性惩罚 + 南向奖励
            rewards[i] = distance_reward * 2 + 2 * occupancy_penalty + south_prior_rewards[i]
        
        return rewards
    
    def _cal_shortest_path_reward(self) -> float:
        """
        计算最短路径奖励，衡量空间开放性
        公式: u = L(实际客厅周长) / sum(外门中点到各个房间门口的距离)
        """
        if not self.living_room or not self.entrance_door:
            return 0.0
        
        # 1. 计算实际客厅周长L
        # 实际客厅周长 = 客厅矩形周长 - 被房间遮挡的边长
        living_room_poly = self.living_room.get_polygon()
        actual_perimeter = self.living_room.width * 2 + self.living_room.height * 2
        
        # 减去被其他房间遮挡的边长（近似计算）
        for room in self.rooms:
            room_poly = room.get_polygon()
            if living_room_poly.touches(room_poly):
                # 计算接触的边界长度
                intersection = living_room_poly.intersection(room_poly)
                if intersection.geom_type == 'LineString':
                    actual_perimeter -= intersection.length
                elif intersection.geom_type == 'MultiLineString':
                    for line in intersection.geoms:
                        actual_perimeter -= line.length
        
        # 确保周长不为负
        actual_perimeter = max(actual_perimeter, self.living_room.width + self.living_room.height)
        
        # 2. 计算外门中点到各个房间门口的距离之和
        door_center = ((self.entrance_door[0] + self.entrance_door[2]) / 2, 
                    (self.entrance_door[1] + self.entrance_door[3]) / 2)
        
        total_distance = 0.0
        valid_rooms = 0
        
        for room in self.rooms:
            if room.door_position:
                # 计算房间门中心点
                door_x1, door_y1, door_x2, door_y2 = room.door_position
                room_door_center = ((door_x1 + door_x2) / 2, (door_y1 + door_y2) / 2)
                
                # 计算直线距离（简化版，实际应该考虑路径）
                distance = math.sqrt((door_center[0] - room_door_center[0])**2 + 
                                (door_center[1] - room_door_center[1])**2)
                total_distance += distance
                valid_rooms += 1
        
        # 如果没有任何有效的房间门，使用房间中心点作为替代
        if valid_rooms == 0:
            for room in self.rooms:
                room_center = room.center
                distance = math.sqrt((door_center[0] - room_center[0])**2 + 
                                (door_center[1] - room_center[1])**2)
                total_distance += distance
            valid_rooms = len(self.rooms)
        
        if valid_rooms == 0 or total_distance == 0:
            return 0.0
        
        # 3. 计算u值
        u_value = actual_perimeter / total_distance
        
        # 4. 根据u值计算奖励（使用sigmoid函数进行平滑）
        # 理想情况下，u值应该在某个范围内（如0.1-0.5）时奖励最高
        # u值过大表示路径过长，开放性差；u值过小表示客厅太小或路径太短，空间局促
        
        # 使用sigmoid函数，让u=0.3时奖励最高
        optimal_u = 0.3
        # 调整系数，使奖励在合理范围内
        scale_factor = 10.0  # 调整这个值来控制奖励的大小
        
        # sigmoid函数的变体，让最优值在u=optimal_u时取得最大值
        # reward = scale_factor * exp(-((u - optimal_u) / 0.1)^2)  # 高斯函数形式
        deviation = (u_value - optimal_u) / 0.15  # 标准化偏差
        reward = scale_factor * math.exp(-deviation * deviation)
        
        # 添加额外奖励：如果u值在理想范围内
        if 0.2 <= u_value <= 0.4:
            reward += 5.0  # 额外奖励
        
        return reward

    def _fix_door_direction(self, room: Room) -> None:
        """ 修复门的方向，使其通向客厅 """
        corners = room.get_corners()
        
        # 找出所有真正通向客厅的边
        good_edges = []
        
        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]
            edge_line = LineString([start, end])
            
            # 计算边的中点
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            
            # 计算边的法向量（指向房间外部）
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            edge_length = math.sqrt(dx*dx + dy*dy)
            if edge_length > 0:
                normal_x = -dy / edge_length
                normal_y = dx / edge_length
                
                # 创建从边中点向外延伸的线段
                test_point_x = mid_x + normal_x * 0.5
                test_point_y = mid_y + normal_y * 0.5
                test_line = LineString([(mid_x, mid_y), (test_point_x, test_point_y)])
                
                # 检查这个延伸是否与客厅相交
                if test_line.intersects(self.living_room.get_polygon()):
                    good_edges.append((start, end, i, edge_length))
        
        # 如果没有真正通向客厅的边，使用与客厅相邻的边
        if not good_edges:
            for i in range(4):
                start = corners[i]
                end = corners[(i + 1) % 4]
                edge_line = LineString([start, end])
                
                # 检查是否与客厅相邻
                if edge_line.distance(self.living_room.get_polygon()) < 0.1:
                    edge_length = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
                    good_edges.append((start, end, i, edge_length))
        
        # 在选中的边上设置门
        if good_edges:
            # 选择最长的边
            good_edges.sort(key=lambda x: x[3], reverse=True)
            start, end, edge_idx, edge_length = good_edges[0]
            
            # 计算边的方向向量
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            if edge_length > 0:
                dx, dy = dx/edge_length, dy/edge_length
            else:
                return
            
            # 在边的中间设置门
            door_len = min(0.8, edge_length * 0.8)
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            
            door_x1 = mid_x - dx * door_len/2
            door_y1 = mid_y - dy * door_len/2
            door_x2 = mid_x + dx * door_len/2
            door_y2 = mid_y + dy * door_len/2
            
            room.door_position = (door_x1, door_y1, door_x2, door_y2)
    
    def _fix_wall_door(self, room: Room) -> None:
        """ 修复贴在墙上的门 """
        corners = room.get_corners()
        
        # 找出所有不贴墙的边
        non_wall_edges = []
        
        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]
            edge_line = LineString([start, end])
            
            # 检查是否贴墙
            wall_distance = edge_line.distance(self.boundary_polygon.exterior)
            if wall_distance >= 0.1:  # 不贴墙
                non_wall_edges.append((start, end, i))
        
        # 如果没有不贴墙的边，使用离客厅最近的边
        if not non_wall_edges:
            # 找出所有边
            all_edges = []
            for i in range(4):
                start = corners[i]
                end = corners[(i + 1) % 4]
                all_edges.append((start, end, i))
            
            # 按到客厅的距离排序
            all_edges.sort(key=lambda edge: LineString([edge[0], edge[1]]).distance(self.living_room.get_polygon()))
            non_wall_edges = [all_edges[0]]  # 使用离客厅最近的边
        
        # 在选中的边上设置门
        if non_wall_edges:
            start, end, edge_idx = non_wall_edges[0]
            edge_length = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
            
            # 计算边的方向向量
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            if edge_length > 0:
                dx, dy = dx/edge_length, dy/edge_length
            else:
                return
            
            # 在边的中间设置门
            door_len = min(0.8, edge_length * 0.8)
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            
            door_x1 = mid_x - dx * door_len/2
            door_y1 = mid_y - dy * door_len/2
            door_x2 = mid_x + dx * door_len/2
            door_y2 = mid_y + dy * door_len/2
            
            room.door_position = (door_x1, door_y1, door_x2, door_y2)
    
    def _validate_and_fix_door_direction(self, room: Room) -> None:
        """ 验证和修复门的方向，确保门开向客厅 """
        if not room.door_position:
            return
        
        door_x1, door_y1, door_x2, door_y2 = room.door_position
        door_center = ((door_x1 + door_x2) / 2, (door_y1 + door_y2) / 2)
        
        # 创建门线段
        door_line = LineString([(door_x1, door_y1), (door_x2, door_y2)])
        
        # 检查门是否贴墙
        if self.boundary_polygon:
            wall_distance = door_line.distance(self.boundary_polygon.exterior)
            if wall_distance < 0.1:
                # 门贴在墙上，需要修复
                self._fix_wall_door(room)
                return
        
        # 检查门是否真正通向客厅
        # 计算门的法向量（指向房间外部）
        dx = door_x2 - door_x1
        dy = door_y2 - door_y1
        door_length = math.sqrt(dx*dx + dy*dy)
        if door_length > 0:
            normal_x = -dy / door_length
            normal_y = dx / door_length
            
            # 创建从门中心向外延伸的线段
            test_point_x = door_center[0] + normal_x * 0.5
            test_point_y = door_center[1] + normal_y * 0.5
            test_line = LineString([door_center, (test_point_x, test_point_y)])
            
            # 检查这个延伸是否与客厅相交
            if not test_line.intersects(self.living_room.get_polygon()):
                # 门不通向客厅，需要修复
                self._fix_door_direction(room)

    def render(
        self, 
        mode="human", 
        render:bool=True,
        show_init_livingroom:bool=False,
        show_room_metrics:bool|None=None,
        show_total_area:bool|None=None,
        show_summary_panel:bool|None=None,
    ):
        """ 渲染环境 """
        if show_room_metrics is None:
            show_room_metrics = bool(getattr(self, "live_render_show_metrics", False))
        if show_total_area is None:
            show_total_area = bool(getattr(self, "live_render_show_total_area", show_room_metrics))
        if show_summary_panel is None:
            show_summary_panel = bool(getattr(self, "live_render_show_summary_panel", False))
        if not hasattr(self, "ax"):
            fig, ax = plt.subplots(figsize=(9, 9))
            self.ax = ax
        else:
            self.ax.clear()
            ax = self.ax
        
        # 绘制边界
        if self.boundary:
            xs, ys = zip(*self.boundary)
            ax.plot(xs + (xs[0], ), ys + (ys[0], ), 'k-', linewidth=3)
            
        if self.living_room:
            if show_init_livingroom:
                room = self.living_room
                # 房间矩形
                room_rect = patches.Rectangle(
                    (room.x1, room.y1), room.width, room.height,
                    linewidth=2, edgecolor='red', 
                    facecolor=RoomColor[room.room_type],
                    alpha=0.7
                )
                ax.add_patch(room_rect)

        # 绘制门之前添加检查
        for room in self.rooms:
            # 确保门位置有效
            if room.door_position and len(room.door_position) == 4:
                dx1, dy1, dx2, dy2 = room.door_position
                # 检查门是否是有效的线段（不是单点）
                if not (dx1 == dx2 and dy1 == dy2):
                    ax.plot([dx1, dx2], [dy1, dy2], 
                        color=DoorColor, linewidth=3)
                else:
                    # 如果门是单点，画一个小叉表示问题
                    ax.scatter(dx1, dy1, color='red', s=50, marker='x')
        
        # 绘制入口门
        if self.entrance_door:
            ed_x1, ed_y1, ed_x2, ed_y2 = self.entrance_door
            if self.language.lower() == "zh":
                ax.plot([ed_x1, ed_x2], [ed_y1, ed_y2], 
                    color=DoorColor, linewidth=3, label='入口门')
            elif self.language.lower() == "en":
                ax.plot([ed_x1, ed_x2], [ed_y1, ed_y2], 
                    color=DoorColor, linewidth=3, label='Entrance Door')
            else:
                ax.plot([ed_x1, ed_x2], [ed_y1, ed_y2], 
                    color=DoorColor, linewidth=3)
        
        # 首先绘制实际客厅区域（作为背景）
        if self.boundary_polygon and self.rooms:
            # 创建所有房间的多边形集合（只包含其他房间，不包含客厅）
            all_room_polygons = []
            for room in self.rooms:
                all_room_polygons.append(room.get_polygon())
            
            # 计算实际客厅区域 = 边界 - 所有其他房间区域
            actual_living_area = self.boundary_polygon
            for room_poly in all_room_polygons:
                actual_living_area = actual_living_area.difference(room_poly)
            
            # 绘制实际客厅区域（浅灰色背景）
            if hasattr(actual_living_area, 'geoms'):  # MultiPolygon
                for polygon in actual_living_area.geoms:
                    if polygon.geom_type == 'Polygon':
                        x, y = polygon.exterior.xy
                        ax.fill(x, y, alpha=0.3, color='lightgray', 
                            linewidth=1, edgecolor='gray')
            elif actual_living_area.geom_type == 'Polygon':
                x, y = actual_living_area.exterior.xy
                ax.fill(x, y, alpha=0.3, color='lightgray', 
                    linewidth=1, edgecolor='gray')
            
            # 添加客厅标签
            if actual_living_area.centroid:
                centroid = actual_living_area.centroid
                if self.language.lower() == "zh":
                    ax.text(centroid.x, centroid.y, '客厅', 
                        ha='center', va='center', fontweight='bold', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
                elif self.language.lower() == "en":
                    ax.text(centroid.x, centroid.y, 'Living Room', 
                        ha='center', va='center', fontweight='bold', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        # 绘制其他房间（在客厅背景之上）
        for room in self.rooms:
            # 房间矩形
            room_rect = patches.Rectangle(
                (room.x1, room.y1), room.width, room.height,
                linewidth=2, edgecolor='black', 
                facecolor=RoomColor[room.room_type],
                alpha=0.8
            )
            ax.add_patch(room_rect)
            
            # 房间标签
            room_center_x = room.x1 + room.width / 2
            room_center_y = room.y1 + room.height / 2
            room_label_lines = []
            if self.language.lower() == "zh":
                room_label_lines.append(name_transform(room.room_type.name))
                if show_room_metrics:
                    room_label_lines.append(f"比例:{room.aspect_ratio:.2f}")
                    room_label_lines.append(f"面积:{room.area:.2f}㎡")
            elif self.language.lower() == "en":
                room_label_lines.append(room.room_type.name)
                if show_room_metrics:
                    room_label_lines.append(f"Ratio:{room.aspect_ratio:.2f}")
                    room_label_lines.append(f"Area:{room.area:.2f}m^2")
            room_fontsize = 10
            if show_room_metrics:
                room_fontsize = 7 if min(room.width, room.height) < 1.5 else 8
            ax.text(
                room_center_x,
                room_center_y,
                "\n".join(room_label_lines),
                ha='center',
                va='center',
                fontweight='bold',
                fontsize=room_fontsize,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6),
            )
            
            # 门
            if room.door_position:
                dx1, dy1, dx2, dy2 = room.door_position
                ax.plot([dx1, dx2], [dy1, dy2], 
                    color=DoorColor, linewidth=3)
        
        # 绘制先验位置（如果存在）
        if self.prior_positions and self.show_prior:
            for i, (cx, cy) in enumerate(self.prior_positions):
                if self.language.lower() == "zh":
                    ax.scatter(cx, cy, color='red', s=20, marker='x', linewidth=2, 
                            label=f'先验位置{i+1}')
                elif self.language.lower() == "en":
                    ax.scatter(cx, cy, color='red', s=20, marker='x', linewidth=2, 
                            label=f'Prior Position {i+1}')
        
        total_area = self.boundary_polygon.area if self.boundary_polygon else 0
        if show_total_area:
            total_area_text = ""
            if self.language.lower() == "zh":
                total_area_text = f"总轮廓面积: {total_area:.2f}㎡"
            elif self.language.lower() == "en":
                total_area_text = f"Total Boundary Area: {total_area:.2f}m^2"
            ax.text(
                0.02,
                0.98,
                total_area_text,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            )

        if show_summary_panel:
            info_text = ""
            if self.language.lower() == "zh":
                info_text = "房间信息:\n"
                if show_total_area:
                    info_text += f"总轮廓面积: {total_area:.2f}㎡\n"
            elif self.language.lower() == "en":
                info_text = "Room Information:\n"
                if show_total_area:
                    info_text += f"Total Boundary Area: {total_area:.2f}m^2\n"
            for room in self.rooms:
                if self.language.lower() == "zh":
                    info_text += f"{name_transform(room.room_type.name)}: {room.area:.2f}㎡, 长宽比: {room.aspect_ratio:.2f}\n"
                elif self.language.lower() == "en":
                    info_text += f"{room.room_type.name}: {room.area:.2f}㎡, Aspect Ratio: {room.aspect_ratio:.2f}\n"

            ax.text(0.02, 0.90 if show_total_area else 0.98, info_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 设置图形属性
        # Determine plot limits
        max_coord = self.grid_size
        if self.boundary:
            xs, ys = zip(*self.boundary)
            max_x = max(xs)
            max_y = max(ys)
            max_coord = max(max_coord, max_x + 2, max_y + 2) # Add some margin
            
        ax.set_xlim(0, max_coord)
        ax.set_ylim(0, max_coord)
        ax.set_aspect('equal')

        if self.language.lower() == "zh":
            ax.set_title('房屋布局可视化', fontsize=14, fontweight='bold')
            ax.set_xlabel('宽度 (m)')
            ax.set_ylabel('高度 (m)')
        elif self.language.lower() == "en":
            ax.set_title('House Layout Visualization', fontsize=14, fontweight='bold')
            ax.set_xlabel('Width (m)')
        
        # 添加图例
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        live_render_path = getattr(self, "live_render_path", "")
        live_render_interval = getattr(self, "live_render_interval", 0.5)
        last_render_ts = getattr(self, "_last_live_render_ts", 0.0)
        now = time.time()
        if live_render_path and (not last_render_ts or now - last_render_ts >= live_render_interval):
            live_render_dir = os.path.dirname(live_render_path)
            live_render_basename = os.path.splitext(os.path.basename(live_render_path))[0]
            frame_path = os.path.join(live_render_dir, f"{live_render_basename}_{int(now * 1000)}.png")
            self.ax.figure.savefig(frame_path, dpi=160, bbox_inches="tight")
            old_frames = sorted(
                glob.glob(os.path.join(live_render_dir, f"{live_render_basename}_*.png")),
                key=os.path.getmtime,
            )
            for stale_frame in old_frames[:-3]:
                try:
                    os.remove(stale_frame)
                except OSError:
                    pass
            self._last_live_render_ts = now

        if render:
            plt.draw()
            plt.pause(0.001)
            
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """ 重置环境 - 修改为支持多边形边界 """
        # 初始化房间
        self.rooms.clear()
        self.living_room = None
        
        # 重置步数
        self.total_steps = 0
        
        # seed
        if kwargs.get("seed", None):
            random.seed(kwargs["seed"])
            np.random.seed(kwargs["seed"])
        
        # 初始化客厅
        self._create_living_room(kwargs.get("rnd", False))
        
        # 在多边形边界内生成初始房间位置
        for i, rtype in enumerate(self.room_type):
            room_placed = False
            attempts = 0
            max_attempts = int(self._env_param("room_init_max_attempts", 100))
            
            while not room_placed and attempts < max_attempts:
                attempts += 1
                
                # 在多边形边界内随机生成房间位置
                min_x, min_y, max_x, max_y = self.boundary_polygon.bounds
                width = self.init_width
                height = self.init_height
                
                # 在多边形内随机选择起始点
                while True:
                    x1 = random.uniform(min_x, max_x - width)
                    y1 = random.uniform(min_y, max_y - height)
                    room = Room(rtype, x1, y1, x1 + width, y1 + height)
                    
                    if self._is_room_in_boundary(room):
                        self.rooms.append(room)
                        room_placed = True
                        break
                    
                    if attempts >= max_attempts:
                        # 如果尝试次数过多，使用边界中心附近的保守位置
                        centroid = self.boundary_polygon.centroid
                        x1 = centroid.x - width / 2
                        y1 = centroid.y - height / 2
                        room = Room(rtype, x1, y1, x1 + width, y1 + height)
                        self.rooms.append(room)
                        room_placed = True
                        break
        
        # 计算内门位置
        self._cal_inner_doors_position()

        # 初始化允许动作列表
        allow_actions = self._get_allow_actions()
        info = {
            "allow_actions": allow_actions
        }
        return self.get_state(), info
    
    def get_state(self) -> np.ndarray:
        """ 获取当前环境状态 """
        states = []
        
        boundary_points = self.boundary
        min_x, min_y = np.min(boundary_points, axis=0)
        max_x, max_y = np.max(boundary_points, axis=0)
        x_range = max_x - min_x
        y_range = max_y - min_y
        diagonal_length = np.sqrt(x_range**2 + y_range**2)
        area_range = x_range * y_range
        
        # 获取所有房间的多边形（包括客厅）
        all_rooms = [self.living_room] + self.rooms if self.living_room else self.rooms
        room_polygons = [room.get_polygon() for room in all_rooms]
        
        for i, room in enumerate(self.rooms):
            # 1. 房间类型 (编码数字)
            states.extend([room.room_type.value])
            
            # 2. 房间中心坐标和宽高（归一化）
            cx, cy = room.center
            states.extend([
                cx / x_range,           # 房间中心x坐标
                cy / y_range,           # 房间中心y坐标
                room.width / x_range,   # 房间宽度
                room.height / y_range   # 房间高度
            ])
            
            # 3. 房间各边到对应边的距离，如上边与上界的距离
            min_x_b, min_y_b, max_x_b, max_y_b = self.boundary_polygon.bounds
            states.extend([
                abs(room.x1 - min_x_b) / x_range,  # 左边到左界的距离
                abs(room.y1 - min_y_b) / y_range,  # 上边到上界的距离
                abs(max_x_b - room.x2) / x_range,  # 右边到右界的距离
                abs(max_y_b - room.y2) / y_range   # 下边到下界的距离
            ])
            
            # 4. 房间到客厅的距离（归一化）
            lv_cx, lv_cy = self.living_room.center
            states.extend([
                abs(cx - lv_cx) / x_range,  # x方向距离
                abs(cy - lv_cy) / y_range   # y方向距离
            ])
            
            # 5. 房间面积和长宽比（面积归一化，长宽比保持不变）
            states.extend([
                room.area / area_range,  # 归一化面积
                room.aspect_ratio        # 长宽比保持不变
            ])
            
            # 6. 当前房间到其他所有房间（包括客厅）的最短边的距离（归一化）
            current_room_poly = room.get_polygon()
            for j, other_poly in enumerate(room_polygons):
                if (j != 0) and (i+1 != j):
                    distance = current_room_poly.distance(other_poly)
                    states.extend([distance / diagonal_length])  # 归一化最短距离
                    
            # 7. 房间中心点到各个先验位置的距离（归一化）
            if self.prior_positions:
                for prior_cx, prior_cy in self.prior_positions:
                    distance = math.sqrt((cx - prior_cx)**2 + (cy - prior_cy)**2)
                    states.extend([distance / diagonal_length])  # 归一化距离
            else:
                # 如果没有设置先验位置，用0填充
                states.extend([0.0] * self.num_agents)
                
        return np.array(states, np.float32).reshape(len(self.rooms), -1)
    
    def is_done(self) -> bool:
        """ 判断是否结束 """
        done_list = [False] * self.num_agents
        global_done = False
        return global_done, done_list
    
    def step(self, action) -> Tuple:
        """ 执行动作 - 修改为支持多边形边界 """
        self.total_steps += 1
        
        for i, room in enumerate(self.rooms):
            act = action[i]
            if self.use_area_ratio:
                ratio = MaxAreasRatio[room.room_type]
                max_area = self.boundary_polygon.area * ratio
            else:
                max_area = MaxAreasValue[room.room_type]
            # 保存原始位置用于回退
            original_x1, original_y1, original_x2, original_y2 = room.x1, room.y1, room.x2, room.y2
            
            if act == 0:  # 向上移动
                dx, dy = self._get_room_move_direction(room, 0, self.shift_value)
                room.y1 += dy
                room.y2 += dy
            elif act == 1:  # 向下移动
                dx, dy = self._get_room_move_direction(room, 0, -self.shift_value)
                room.y1 += dy
                room.y2 += dy
            elif act == 2:  # 向左移动
                dx, dy = self._get_room_move_direction(room, -self.shift_value, 0)
                room.x1 += dx
                room.x2 += dx
            elif act == 3:  # 向右移动
                dx, dy = self._get_room_move_direction(room, self.shift_value, 0)
                room.x1 += dx
                room.x2 += dx
            elif act == 4:  # 不做任何移动
                pass
            elif act == 5:  # 上边增加
                if room.area < max_area:
                    dx1, dy1, dx2, dy2 = self._get_room_resize_direction(room, 0, 0, 0, self.shift_value)
                    room.y2 += dy2
            elif act == 6:  # 上边减少
                if room.y2 - room.y1 > 1.0:
                    dx1, dy1, dx2, dy2 = self._get_room_resize_direction(room, 0, 0, 0, -self.shift_value)
                    room.y2 += dy2
            elif act == 7: # 底边增加
                if room.area < max_area:
                    dx1, dy1, dx2, dy2 = self._get_room_resize_direction(room, 0, -self.shift_value, 0, 0)
                    room.y1 += dy1
            elif act == 8: # 底边减少
                if room.y2 - room.y1 > 1.0:
                    dx1, dy1, dx2, dy2 = self._get_room_resize_direction(room, 0, self.shift_value, 0, 0)
                    room.y1 += dy1
            elif act == 9:  # 左边增加
                if room.area < max_area:
                    dx1, dy1, dx2, dy2 = self._get_room_resize_direction(room, -self.shift_value, 0, 0, 0)
                    room.x1 += dx1
            elif act == 10:  # 左边减少
                if room.x2 - room.x1 > 1.0 and room.area < max_area:
                    dx1, dy1, dx2, dy2 = self._get_room_resize_direction(room, self.shift_value, 0, 0, 0)
                    room.x1 += dx1
            elif act == 11:  # 右边增加
                if room.area < max_area:
                    dx1, dy1, dx2, dy2 = self._get_room_resize_direction(room, 0, 0, self.shift_value, 0)
                    room.x2 += dx2
            elif act == 12:  # 右边减少 
                if room.x2 - room.x1 > 1.0 and room.area < max_area:
                    dx1, dy1, dx2, dy2 = self._get_room_resize_direction(room, 0, 0, -self.shift_value, 0)
                    room.x2 += dx2
            else:
                raise ValueError(f"无效的动作: {act}")
            
            # 确保x1, x2, y1, y2的关系
            room.x1, room.x2 = sorted([room.x1, room.x2])
            room.y1, room.y2 = sorted([room.y1, room.y2])
            
            # 最终检查：如果房间不在边界内，回退到原始位置
            if not self._is_room_in_boundary(room):
                room.x1, room.y1, room.x2, room.y2 = original_x1, original_y1, original_x2, original_y2
        
        # 计算内门位置
        self._cal_inner_doors_position()
        
        # 获取下一个状态和奖励
        next_state = self.get_state()
        
        # 计算奖励
        reward_info = self.cal_reward()
        rewards = [r for r, _ in reward_info]
        reward_components = [c for _, c in reward_info]

        # 判断是否结束
        global_done, per_agent_done = self.is_done()
        # 其他信息
        info = {
            "reward_components": reward_components,
            "done_list": per_agent_done,
            "allow_actions": self._get_allow_actions()
        }
        return next_state, rewards, global_done, info
    
    def cal_reward(self) -> List[Tuple[float, Dict[str, float]]]:
        """ 计算奖励 """
        base_r = self._cal_base_reward()        # 基础奖励
        extra_r = self._cal_extra_reward()      # 额外奖励
        if self.method != "base":
            # 合并两个奖励
            new_r = []
            for i in range(len(self.rooms)):
                a = base_r[i][0] + extra_r[i][0]
                b = {**base_r[i][1], **extra_r[i][1]}
                new_r.append((a, b))
            return new_r
        else:
            return base_r
    
    def _cal_base_reward(self) -> List[Tuple[float, Dict[str, float]]]:
        """ 计算基础奖励 
        包含:
            1. 房间是否在边界内, 不在则惩罚
            2. 面积奖励/惩罚
            3. 长宽比奖励/惩罚
            4. 贴边奖励/惩罚
            5. 房间与房间贴边奖励/惩罚
            6. 角落占领奖励/惩罚
            7. 先验知识布局奖励/惩罚
        """
        rewards = []
        
        # 预处理：计算所有房间的门中心位置
        door_centers = []
        for room in self.rooms:
            if room.door_position:
                dx1, dy1, dx2, dy2 = room.door_position
                door_center = ((dx1 + dx2) / 2, (dy1 + dy2) / 2)
                # 确定门的方向（靠近哪一边）
                door_edge = self._get_door_edge(room, door_center)
                door_centers.append((door_center, door_edge, room))
            else:
                door_centers.append((None, None, room))
        
        # 先验知识布局奖励计算（新增）
        prior_rewards = self._cal_prior_layout_reward()
        
        room_polygons = [room.get_polygon() for room in self.rooms]

        # 首先确定每个角落被哪个房间占据
        for i, room in enumerate(self.rooms):
            r = 0.0
            r_components = {}
            room_poly = room.get_polygon()
            area_under_penalty_scale = float(self._reward_param("area_under_penalty_scale", 5.0))
            area_over_penalty_scale = float(self._reward_param("area_over_penalty_scale", 20.0))
            area_reward_bonus = float(self._reward_param("area_reward_bonus", 10.0))
            aspect_under_penalty_scale = float(self._reward_param("aspect_under_penalty_scale", 5.0))
            aspect_over_penalty_scale = float(self._reward_param("aspect_over_penalty_scale", 20.0))
            aspect_reward_bonus = float(self._reward_param("aspect_reward_bonus", 30.0))
            edge_touch_threshold = float(self._reward_param("edge_touch_threshold", 0.02))
            edge_near_threshold = float(self._reward_param("edge_near_threshold", 0.5))
            edge_touch_reward = float(self._reward_param("edge_touch_reward", 20.0))
            edge_near_reward_scale = float(self._reward_param("edge_near_reward_scale", 8.0))
            shared_wall_reward_per_meter = float(self._reward_param("shared_wall_reward_per_meter", 2.0))
            door_to_door_threshold = float(self._reward_param("door_to_door_threshold", 2.0))
            door_to_door_penalty_scale = float(self._reward_param("door_to_door_penalty_scale", 20.0))
            door_face_to_face_threshold = float(self._reward_param("door_face_to_face_threshold", 0.5))
            door_face_to_face_extra_penalty = float(self._reward_param("door_face_to_face_extra_penalty", 30.0))
            corner_touch_threshold = float(self._reward_param("corner_touch_threshold", 0.02))
            corner_near_threshold = float(self._reward_param("corner_near_threshold", 0.5))
            corner_touch_reward = float(self._reward_param("corner_touch_reward", 20.0))
            corner_near_reward_scale = float(self._reward_param("corner_near_reward_scale", 20.0))
            corner_dot_threshold = float(self._reward_param("corner_dot_threshold", 0.1))
            door_proximity_threshold = float(self._reward_param("door_proximity_threshold", 0.6))
            vertical_required_clearance = float(self._reward_param("vertical_required_clearance", 1.2))
            horizontal_required_clearance = float(self._reward_param("horizontal_required_clearance", 1.0))
            door_block_penalty_scale = float(self._reward_param("door_block_penalty_scale", 200.0))
            overlap_penalty_scale = float(self._reward_param("overlap_penalty_scale", 50.0))
            nesting_equal_dimension_tolerance = float(self._reward_param("nesting_equal_dimension_tolerance", 0.1))
            nesting_exact_match_tolerance = float(self._reward_param("nesting_exact_match_tolerance", 0.01))
            nesting_equal_dimension_penalty_scale = float(self._reward_param("nesting_equal_dimension_penalty_scale", 20.0))
            nesting_equal_dimension_exact_multiplier = float(self._reward_param("nesting_equal_dimension_exact_multiplier", 2.0))
            nesting_sigmoid_scale = float(self._reward_param("nesting_sigmoid_scale", 10.0))
            nesting_penalty_scale = float(self._reward_param("nesting_penalty_scale", 5.0))

            """ 1. 面积奖励/惩罚, 面积合适, 不扣分 """
            area = room.area
            min_area = MinAreas[room.room_type]
            max_area = self._get_room_max_area(room)
            area_success = False
            
            if area < min_area:
                area_value = -area_under_penalty_scale * abs(min_area - area) * self.base_reward_weight["面积奖励"]
                r += area_value
                r_components["area"] = area_value
            elif area > max_area:
                area_penalty = -area_over_penalty_scale * abs(area - max_area) * self.base_reward_weight["面积奖励"]
                r += area_penalty
                r_components["area"] = area_penalty
            else:
                area_success = True
                area_value = area_reward_bonus * self.base_reward_weight["面积奖励"]
                r += area_value
                r_components["area"] = area_value
            
            """ 2. 长宽比奖励/惩罚, 长宽比合适, 不扣分 """
            min_ratio, max_ratio = ASPECT_RATIO_RANGES[room.room_type]
            aspect_ratio = room.aspect_ratio
            if aspect_ratio < min_ratio:
                penalty = -aspect_under_penalty_scale * abs(min_ratio - aspect_ratio) * self.base_reward_weight["长宽比奖励"]
                r += penalty
                r_components["aspect_ratio"] = penalty
            elif aspect_ratio > max_ratio:
                ratio_penalty = -aspect_over_penalty_scale * abs(aspect_ratio - max_ratio) * self.base_reward_weight["长宽比奖励"]
                r += ratio_penalty
                r_components["aspect_ratio"] = ratio_penalty
            else:
                if area_success:
                    aspect_reward = aspect_reward_bonus * self.base_reward_weight["长宽比奖励"]
                    r += aspect_reward
                    r_components["aspect_ratio"] = aspect_reward
                else:
                    r += 0
                    r_components["aspect_ratio"] = 0
            
            """ 3. 贴边奖励 """
            # 计算到边界的最小距离
            distance_to_boundary = self.boundary_polygon.exterior.distance(room_poly)
            # 平滑的贴边奖励 - 适应不同边界形状
            if distance_to_boundary < edge_touch_threshold:
                edge_reward = edge_touch_reward
            elif distance_to_boundary < edge_near_threshold:
                edge_reward = edge_near_reward_scale * (1 - distance_to_boundary)
            else:
                edge_reward = 0 * distance_to_boundary
            r += edge_reward * self.base_reward_weight['贴边奖励'] 
            r_components["edge"] = edge_reward * self.base_reward_weight['贴边奖励'] 
            
            """ 4. 房间与房间之间的边界贴合奖励 """
            shared_wall_reward = 0.0
            door_to_door_penalty = 0.0
            
            for j, other_room in enumerate(self.rooms):
                if i != j:  # 不与自己比较
                    # 检查是否有共享的墙
                    if (abs(room.x1 - other_room.x2) < edge_touch_threshold or 
                        abs(room.x2 - other_room.x1) < edge_touch_threshold or
                        abs(room.y1 - other_room.y2) < edge_touch_threshold or
                        abs(room.y2 - other_room.y1) < edge_touch_threshold):
                        
                        # 计算共享墙的长度
                        x_overlap = max(0, min(room.x2, other_room.x2) - max(room.x1, other_room.x1))
                        y_overlap = max(0, min(room.y2, other_room.y2) - max(room.y1, other_room.y1))
                        shared_length = max(x_overlap, y_overlap)
                        
                        # 根据共享墙长度给予奖励
                        if shared_length > 0:
                            shared_wall_reward += shared_wall_reward_per_meter * shared_length
                            
                        # 检查门对门的情况（新增）
                        if (door_centers[i][0] is not None and door_centers[j][0] is not None and
                            door_centers[i][1] == door_centers[j][1]):  # 门在同一侧
                            
                            # 计算两个门中心点的距离
                            dist = math.sqrt((door_centers[i][0][0] - door_centers[j][0][0])**2 + 
                                        (door_centers[i][0][1] - door_centers[j][0][1])**2)
                            
                            if dist < door_to_door_threshold:
                                # 距离越近惩罚越大
                                penalty_factor = (door_to_door_threshold - dist) / max(door_to_door_threshold, 1e-9)
                                door_to_door_penalty -= door_to_door_penalty_scale * penalty_factor
                                
                                if dist < door_face_to_face_threshold:
                                    door_to_door_penalty -= door_face_to_face_extra_penalty
            
            r += shared_wall_reward * self.base_reward_weight["房间与房间之间的边界贴合奖励"]
            r_components["shared_wall"] = shared_wall_reward
            
            r += door_to_door_penalty
            r_components["door_to_door"] = door_to_door_penalty
            
            """ 5. 角落占领奖励 (修改后的版本) """
            boundary_points = list(self.boundary_polygon.exterior.coords)
            actual_corners = []
            
            # 检测实际角落（角度变化大的点）
            for j in range(len(boundary_points)-1):
                p1 = boundary_points[j-1]
                p2 = boundary_points[j]
                p3 = boundary_points[j+1]
                
                # 计算角度变化
                vec1 = (p2[0]-p1[0], p2[1]-p1[1])
                vec2 = (p3[0]-p2[0], p3[1]-p2[1])
                
                # 如果角度变化大，认为是角落
                if abs(vec1[0]*vec2[0] + vec1[1]*vec2[1]) < corner_dot_threshold:
                    actual_corners.append(p2)
            
            # 使用实际角落计算奖励
            corner_reward = 0
            for corner in actual_corners:
                distance = math.sqrt((room.center[0]-corner[0])**2 + 
                                    (room.center[1]-corner[1])**2)
                if distance < corner_touch_threshold:
                    corner_reward += corner_touch_reward
                if distance < corner_near_threshold:
                    corner_reward += corner_near_reward_scale * (1 - distance)
            r += corner_reward * self.base_reward_weight['角落占领奖励']
            r_components["corner"] = corner_reward * self.base_reward_weight['角落占领奖励']
                    
            """ 7. 外门遮挡惩罚 - 修正版，支持任意位置的门 """
            if self.entrance_door:
                door_x1, door_y1, door_x2, door_y2 = self.entrance_door
                door_penalty = 0.0
                
                # 判断门的类型（水平或垂直）
                is_vertical_door = abs(door_x1 - door_x2) < edge_touch_threshold
                is_horizontal_door = abs(door_y1 - door_y2) < edge_touch_threshold
                
                if is_vertical_door:
                    # 垂直门（通常在左右边界）
                    door_x = (door_x1 + door_x2) / 2
                    door_y_min = min(door_y1, door_y2)
                    door_y_max = max(door_y1, door_y2)
                    door_width = door_y_max - door_y_min
                    
                    # 检查房间是否靠近门的边界
                    if abs(room.x1 - door_x) < door_proximity_threshold or abs(room.x2 - door_x) < door_proximity_threshold:
                        # 计算房间在y轴上与门的重叠部分
                        overlap_low = max(room.y1, door_y_min)
                        overlap_high = min(room.y2, door_y_max)
                        overlap_length = max(0, overlap_high - overlap_low)
                        
                        # 计算需要预留的长度（至少1.2m）
                        required_clearance = vertical_required_clearance
                        if overlap_length > (door_width - required_clearance):
                            # 遮挡过多，给予重罚
                            door_penalty = -door_block_penalty_scale * (overlap_length - (door_width - required_clearance))
                
                elif is_horizontal_door:
                    # 水平门（通常在上边界）
                    door_y = (door_y1 + door_y2) / 2
                    door_x_min = min(door_x1, door_x2)
                    door_x_max = max(door_x1, door_x2)
                    door_width = door_x_max - door_x_min
                    
                    # 检查房间是否靠近门的边界
                    if abs(room.y1 - door_y) < door_proximity_threshold or abs(room.y2 - door_y) < door_proximity_threshold:
                        # 计算房间在x轴上与门的重叠部分
                        overlap_low = max(room.x1, door_x_min)
                        overlap_high = min(room.x2, door_x_max)
                        overlap_length = max(0, overlap_high - overlap_low)
                        
                        # 计算需要预留的长度（至少1.0m）
                        required_clearance = horizontal_required_clearance
                        if overlap_length > (door_width - required_clearance):
                            # 遮挡过多，给予重罚
                            door_penalty = -door_block_penalty_scale * (overlap_length - (door_width - required_clearance))
                
                r += door_penalty * self.base_reward_weight['外门遮挡惩罚']
                r_components["door_block"] = door_penalty * self.base_reward_weight['外门遮挡惩罚']
            else:
                r_components["door_block"] = 0
            
            """ 8. 先验知识布局奖励 """
            if self.prior_positions:
                prior_reward = prior_rewards[i]
                scaled_prior_reward = prior_reward * self.prior_layout_scale * self.base_reward_weight['先验知识布局奖励']
                r += scaled_prior_reward
                r_components["prior_layout"] = scaled_prior_reward
            else:
                r_components["prior_layout"] = 0
            
            # 9. 房间重叠惩罚（新增）
            overlap_penalty = 0.0
            for j, other_room in enumerate(self.rooms):
                if i != j:  # 不与自己比较
                    other_poly = other_room.get_polygon()
                    overlap_area = room_poly.intersection(other_poly).area
                    
                    if overlap_area > 0:
                        # 计算重叠面积相对于当前房间面积的比例
                        overlap_ratio = overlap_area / room.area
                        
                        # 根据重叠比例给予惩罚，重叠越多惩罚越大
                        # 使用指数增长函数：惩罚 = -50 * (重叠比例)^2
                        penalty = -overlap_penalty_scale * (overlap_ratio ** 2)
                        overlap_penalty += penalty
            r += overlap_penalty
            r_components["overlap"] = overlap_penalty
            
            """ A1. 房间嵌套惩罚 - 平滑版本 """
            nesting_penalty = 0
            for j, other_room in enumerate(self.rooms):
                if i != j:
                    other_poly = room_polygons[j]
                    intersection = room_poly.intersection(other_poly)
                    
                    # 新增：检查相交部分宽高相等惩罚
                    equal_dimension_penalty = 0.0
                    if not intersection.is_empty and intersection.area > 0:
                        # 计算相交区域的边界
                        min_x, min_y, max_x, max_y = intersection.bounds
                        overlap_width = max_x - min_x
                        overlap_height = max_y - min_y
                        
                        # 如果相交部分的宽度和高度相等（在一定容差范围内），给予惩罚
                        tolerance = nesting_equal_dimension_tolerance
                        if abs(overlap_width - overlap_height) < tolerance:
                            # 根据相交面积比例计算惩罚
                            overlap_ratio = intersection.area / min(room.area, other_room.area)
                            equal_dimension_penalty = -nesting_equal_dimension_penalty_scale * overlap_ratio
                            
                            # 如果完全相等，额外惩罚
                            if abs(overlap_width - overlap_height) < nesting_exact_match_tolerance:
                                equal_dimension_penalty *= nesting_equal_dimension_exact_multiplier
                    
                    nesting_penalty += equal_dimension_penalty
                    
                    overlap_area = intersection.area
                    if overlap_area > 0:
                        room_priority = NESTING_PRIORITY[room.room_type.name]
                        other_priority = NESTING_PRIORITY[other_room.room_type.name]
                        max_overlap = NESTING_WEIGHTS[other_room.room_type.name] if room_priority > other_priority else NESTING_WEIGHTS[room.room_type.name]
                        
                        overlap_ratio = overlap_area / min(room.area, other_room.area)
                        
                        # 平滑处理：使用S形函数计算惩罚
                        if overlap_ratio > max_overlap:
                            penalty_factor = min(1.0, (overlap_ratio - max_overlap) / (1 - max_overlap))
                            penalty = nesting_penalty_scale * (1 / (1 + math.exp(-nesting_sigmoid_scale * (penalty_factor - 0.5))))
                            nesting_penalty += penalty
            
            r -= nesting_penalty * self.base_reward_weight['房间嵌套惩罚']
            r_components['nesting_penalty'] = -nesting_penalty * self.base_reward_weight['房间嵌套惩罚']
            
            rewards.append((r, r_components))
        return rewards
    
    def _cal_extra_reward(self) -> List[Tuple[float, Dict[str, float]]]:
        """ 计算额外奖励 - 根据PDF要求重新实现 """
        rewards = []
        
        # 计算最短路径奖励, 这里因为每个房间都要加，所以取平均值
        shortest_path_reward = self._cal_shortest_path_reward() / self.num_agents
        
        for i, room in enumerate(self.rooms):
            r = 0.0
            r_components = {}

            """ 1. 邻接关系奖励 - 简化版本 """
            adj_reward = self._cal_simplified_adjacency_reward(room)
            r += adj_reward * self.extra_reward_weight['邻接关系奖励']
            r_components["adjacency"] = adj_reward * self.extra_reward_weight['邻接关系奖励']
            
            """ 2. 公私分区奖励 """
            privacy_reward = self._cal_privacy_partition_reward(room)
            r += privacy_reward * self.extra_reward_weight['隐私保护奖励']
            r_components["privacy"] = privacy_reward * self.extra_reward_weight['隐私保护奖励']
            
            """ 3. 南向采光奖励 """
            south_room_types = set(self._reward_param("south_lighting_room_types", ["BEDROOM", "LIVING_ROOM"]))
            if room.room_type.name in south_room_types:
                south_facing_reward = self._cal_south_lighting_reward(room)
                r += south_facing_reward * self.extra_reward_weight['南向采光奖励']
                r_components["south_facing"] = south_facing_reward * self.extra_reward_weight['南向采光奖励']
            
            """ 4. 最短路径奖励 - 所有房间都加上 """
            r += shortest_path_reward * self.extra_reward_weight['最短路径奖励']
            r_components["shortest_path"] = shortest_path_reward * self.extra_reward_weight['最短路径奖励']

            """ 5. 动静分区奖励 """
            noise_reward = self._cal_noise_partition_reward(room)
            r += noise_reward * self.extra_reward_weight['噪声干扰度奖励']
            r_components['noise_reward'] = noise_reward * self.extra_reward_weight['噪声干扰度奖励']

            rewards.append((r, r_components))

        return rewards

    def _cal_simplified_adjacency_reward(self, current_room: Room) -> float:
        """计算简化的邻接关系奖励"""
        total_score = 0
        
        # 计算所有房间的中心点
        room_centers = {}
        for room in self.rooms:
            room_centers[room.room_type.name] = room.center
            
        # 添加客厅中心点
        if self.living_room:
            room_centers['LIVING_ROOM'] = self.living_room.center
        
        # 计算最大距离D_max
        max_distance = 0
        if self.entrance_door:
            door_center = ((self.entrance_door[0] + self.entrance_door[2]) / 2, 
                        (self.entrance_door[1] + self.entrance_door[3]) / 2)
            for room_type, center in room_centers.items():
                if room_type != 'LIVING_ROOM':
                    dist = np.sqrt((center[0] - door_center[0])**2 + (center[1] - door_center[1])**2)
                    if dist > max_distance:
                        max_distance = dist
        
        if max_distance == 0:
            max_distance = 1  # 避免除零
        
        # 邻接关系定义
        synergistic_pairs = [tuple(pair) for pair in self._reward_param("adjacency_synergistic_pairs", [('BEDROOM', 'BATHROOM'), ('BEDROOM', 'BALCONY'), ('BATHROOM', 'BALCONY')])]
        repellent_pairs = [tuple(pair) for pair in self._reward_param("adjacency_repellent_pairs", [('BEDROOM', 'KITCHEN')])]
        
        # 计算每对关系的得分
        for room1, room2 in synergistic_pairs + repellent_pairs:
            if room1 in room_centers and room2 in room_centers:
                # 计算两个房间之间的距离
                center1 = room_centers[room1]
                center2 = room_centers[room2]
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                #*** 不要SDR了 ***#
                # 计算SDR
                # sdr = distance / max_distance
                # 根据关系类型计算得分
                # if (room1, room2) in synergistic_pairs or (room2, room1) in synergistic_pairs:
                #     # 协同关系：距离越近越好
                #     score = 2.0 * (1 - sdr)
                # else:  # 排斥关系
                #     # 排斥关系：距离越远越好
                #     score = 2.0 * sdr
                
                score = float(self._reward_param("adjacency_synergy_reward", 2.0))
                if (room1, room2) in repellent_pairs:
                    score = float(self._reward_param("adjacency_repellent_penalty", -2.0))
                
                # 如果当前房间参与了这个关系对，添加得分
                if room1 == current_room.room_type.name or room2 == current_room.room_type.name:
                    total_score += score
        
        return total_score

    def _cal_privacy_partition_reward(self, room: Room) -> float:
        """计算公私分区奖励"""
        if not self.entrance_door:
            return 0
        
        # 计算入口门中心点
        door_center = ((self.entrance_door[0] + self.entrance_door[2]) / 2, 
                    (self.entrance_door[1] + self.entrance_door[3]) / 2)
        
        # 计算房间中心到门的距离
        room_center = room.center
        distance = np.sqrt((room_center[0] - door_center[0])**2 + 
                        (room_center[1] - door_center[1])**2)
        
        # 找到最大距离用于归一化
        max_distance = 0
        for other_room in self.rooms:
            other_center = other_room.center
            other_distance = np.sqrt((other_center[0] - door_center[0])**2 + 
                                (other_center[1] - door_center[1])**2)
            if other_distance > max_distance:
                max_distance = other_distance
        
        if max_distance == 0:
            return 0
        
        # 归一化距离
        normalized_distance = distance / max_distance
        
        # 公私空间分类
        private_spaces = set(self._reward_param("privacy_private_spaces", ["BEDROOM", "BATHROOM"]))
        
        # 使用sigmoid函数计算奖励
        k = float(self._reward_param("privacy_sigmoid_k", 10.0))
        d0 = float(self._reward_param("privacy_sigmoid_midpoint", 0.5))
        privacy_base_reward = float(self._reward_param("privacy_base_reward", 2.0))
        privacy_reward_scale = float(self._reward_param("privacy_reward_scale", 5.0))
        
        # 阳台和厨房无所谓
        if room.room_type.name in private_spaces:
            # 私密空间：距离越远得分越高
            reward = privacy_base_reward + 1 / (1 + math.exp(-k * (normalized_distance - d0)))
        # elif room.room_type in public_spaces:
        #     # 公共空间：距离越近得分越高
        #     reward = 1 / (1 + math.exp(k * (normalized_distance - d0)))
        else:
            reward = 0
        
        return reward * privacy_reward_scale

    def _cal_south_lighting_reward(self, room: Room) -> float:
        """计算南向采光奖励"""
        if not self.boundary_polygon:
            return 0
        
        # 获取边界范围
        min_x, min_y, max_x, max_y = self.boundary_polygon.bounds
        
        south_region_ratio = float(self._reward_param("south_region_ratio", 0.3))
        south_position_reward_scale = float(self._reward_param("south_position_reward_scale", 5.0))
        south_balcony_bonus = float(self._reward_param("south_balcony_bonus", 2.0))
        south_threshold = min_y + (max_y - min_y) * south_region_ratio
        
        # 计算房间中心点的y坐标
        room_center_y = room.center[1]
        
        # 基础奖励：房间在南向区域
        base_reward = 0.0
        
        if room_center_y <= south_threshold:
            # 房间在南向区域，计算奖励
            # 距离南边界越近奖励越高
            distance_to_south = room_center_y - min_y
            max_possible_distance = south_threshold - min_y
            
            if max_possible_distance > 0:
                position_score = 1 - (distance_to_south / max_possible_distance)
                base_reward = south_position_reward_scale * position_score
                
                # 检查是否有阳台相邻且朝南
                if self._has_south_facing_balcony_adjacent(room):
                    base_reward += south_balcony_bonus
        
        return base_reward

    def _cal_noise_partition_reward(self, room: Room) -> float:
        """计算动静分区奖励（噪声奖励）
        根据PDF要求：
        1. 各房间基础噪声指数NI: 厨房、客厅取5.5，浴室2.5，卧室1.5，阳台设为最低
        2. 计算每个房间相邻房间中的最大噪声指数NI_max,i
        3. 计算噪声差异平方和
        4. 根据不同空间对噪声的敏感程度赋予权重：卧室0.5，浴室0.2，厨房0.1，客厅0.15，阳台0.05
        5. 综合噪声干扰度奖励 = (1/N) * Σ w_i * (NI_max,i - NI_i)^2
        """
        # 基础噪声指数
        noise_indices = {
            'LIVING_ROOM': float(NOISE_WEIGHTS.get('LIVING_ROOM', 5.5)),
            'KITCHEN': float(NOISE_WEIGHTS.get('KITCHEN', 5.5)),
            'BATHROOM': float(NOISE_WEIGHTS.get('BATHROOM', 2.5)),
            'BEDROOM': float(NOISE_WEIGHTS.get('BEDROOM', 1.5)),
            'BALCONY': float(NOISE_WEIGHTS.get('BALCONY', 0.0))
        }
        
        # 噪声敏感度权重
        weight_factors = {
            RoomType.BEDROOM: float(self._reward_param("noise_sensitivity_bedroom", 0.5)),
            RoomType.BATHROOM: float(self._reward_param("noise_sensitivity_bathroom", 0.2)),
            RoomType.KITCHEN: float(self._reward_param("noise_sensitivity_kitchen", 0.1)),
            RoomType.LIVING_ROOM: float(self._reward_param("noise_sensitivity_living_room", 0.15)),
            RoomType.BALCONY: float(self._reward_param("noise_sensitivity_balcony", 0.05))
        }
        
        # 获取当前房间的噪声指数
        current_noise = noise_indices.get(room.room_type.name, 0.0)
        
        # 获取相邻房间（包括客厅）
        adjacent_rooms = []
        room_poly = room.get_polygon()
        
        # 检查与其他房间的相邻关系
        for other_room in self.rooms:
            if other_room != room:
                other_poly = other_room.get_polygon()
                if room_poly.touches(other_poly) or room_poly.distance(other_poly) < float(self._reward_param("noise_adjacent_distance_threshold", 0.1)):
                    adjacent_rooms.append(other_room)
        
        # 检查与客厅的相邻关系
        if self.living_room:
            living_poly = self.living_room.get_polygon()
            if room_poly.touches(living_poly) or room_poly.distance(living_poly) < float(self._reward_param("noise_adjacent_distance_threshold", 0.1)):
                adjacent_rooms.append(self.living_room)
        
        # 计算相邻房间中的最大噪声指数
        ni_max_i = 0
        for adj_room in adjacent_rooms:
            if adj_room == self.living_room:
                adj_noise = noise_indices['LIVING_ROOM']
            else:
                adj_noise = noise_indices.get(adj_room.room_type.name, 0.0)
            
            if adj_noise > ni_max_i:
                ni_max_i = adj_noise
        
        # 计算噪声差异平方
        noise_difference_square = (ni_max_i - current_noise) ** 2
        
        # 获取权重
        weight = weight_factors.get(room.room_type, 0.1)
        
        # 计算房间总数N（不包括阳台，因为阳台在噪声计算中权重最低）
        room_types_in_noise_calc = set(self._reward_param("noise_calc_room_types", ["BEDROOM", "BATHROOM", "KITCHEN", "LIVING_ROOM"]))
        N = sum(1 for r in self.rooms if r.room_type.name in room_types_in_noise_calc)
        if self.living_room and "LIVING_ROOM" in room_types_in_noise_calc:
            N += 1
        
        # 避免除零
        if N == 0:
            N = 1
        
        # 计算综合噪声干扰度奖励
        # 根据PDF公式：R_noise = (1/N) * Σ w_i * (NI_max,i - NI_i)^2
        # 注意：这里是惩罚，所以返回负值
        reward = -(weight * noise_difference_square) / N
        
        return reward

    def _has_south_facing_balcony_adjacent(self, room: Room) -> bool:
        """检查房间是否有朝南的阳台相邻"""
        if not self.boundary_polygon:
            return False
        
        # 获取边界范围
        min_x, min_y, max_x, max_y = self.boundary_polygon.bounds
        
        # 定义南向区域（下方30%的区域）
        south_threshold = min_y + (max_y - min_y) * 0.3
        
        room_poly = room.get_polygon()
        
        # 检查所有阳台
        for other_room in self.rooms:
            if other_room.room_type == RoomType.BALCONY:
                # 检查阳台是否朝南
                balcony_center_y = other_room.center[1]
                if balcony_center_y <= south_threshold:
                    # 检查阳台是否与当前房间相邻
                    balcony_poly = other_room.get_polygon()
                    if room_poly.touches(balcony_poly) or room_poly.distance(balcony_poly) < 0.1:
                        return True
        
        return False
