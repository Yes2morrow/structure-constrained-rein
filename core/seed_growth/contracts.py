"""CP1: immutable seeds, full-thickness obstacles and precise-output scheduling.

This module does not perform growth or replace the existing MAPPO environment.
"""
from dataclasses import dataclass, field
import hashlib
import json
import math

from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union

from core.envs.structure_geometry import fixed_polygon, normalize_structures


def finite_pair(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f'{label} 必须是两个数值')
    result = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f'{label} 必须是有限数值')
    return result


@dataclass(frozen=True)
class ShapeLimits:
    # Experimental geometric policy, not building-code defaults.
    max_reflex: int = 4
    min_rectangularity: float = .75
    max_notch_depth_ratio: float = .25
    max_extra_perimeter: float = .25
    min_segment_ratio: float = .02
    min_width_ratio: float = .10
    allow_long_edge_split: bool = True
    long_edge_ratio: float = 1.5


@dataclass(frozen=True)
class RoomSpec:
    id: str
    seed: tuple[float, float]
    target_area: float
    area_range: tuple[float, float]
    aspect_range: tuple[float, float]
    shape_policy: str = 'regular'
    min_width: float | None = None
    shape_limits: ShapeLimits = field(default_factory=ShapeLimits)
    role: str = 'agent'


@dataclass(frozen=True)
class Relation:
    source: str
    target: str
    kind: str
    # Explicit thresholds are deliberately not invented by this contract layer.
    min_shared_length: float | None = None
    min_clear_width: float | None = None
    min_distance: float | None = None


@dataclass(frozen=True)
class Problem:
    rooms: tuple[RoomSpec, ...]
    relations: tuple[Relation, ...]
    boundary: Polygon
    fixed: object
    free_space: object
    grid_size: float
    precise_every_episodes: int = 250
    entrance: object = None
    entrance_clearance: object = None
    fitting_area_tolerance: float = .12

    @property
    def active_rooms(self):
        return tuple(r for r in self.rooms if r.role != 'residual')

    @property
    def residual_room(self):
        return next((r for r in self.rooms if r.role == 'residual'), None)

    def validate_seeds(self, seeds):
        if set(seeds) != {r.id for r in self.active_rooms}:
            raise ValueError('种子 ID 必须与目标房间一一对应')
        seen = set()
        for identifier, raw in seeds.items():
            point = finite_pair(raw, f'{identifier}.seed')
            if point in seen:
                raise ValueError('两个房间不能使用重合的种子')
            seen.add(point)
            p = Point(point)
            if not self.boundary.contains(p) or self.fixed.intersects(p):
                raise ValueError(f'{identifier} 的种子位于边界外、边界上或固定结构内/表面')
            if self.entrance_clearance is not None and self.entrance_clearance.covers(p):
                raise ValueError(f'{identifier} 的种子占用客厅户门内侧预留区域')


def build_problem(config):
    """Compile fixed geometry once. Preserve the source config without mutation."""
    building = config['ExistingBuilding']
    vertices = [finite_pair(p, 'boundary') for p in building['boundary']]
    boundary = Polygon(vertices)
    if not boundary.is_valid or boundary.area <= 0:
        raise ValueError('建筑边界必须有效且有面积')
    objects = normalize_structures(building.get('fixed_objects', []))
    polygons = [fixed_polygon(item) for item in objects]
    if any(not p.is_valid or p.is_empty for p in polygons):
        raise ValueError('固定结构多边形无效')
    fixed = unary_union(polygons) if polygons else Polygon()
    free = boundary.difference(fixed)
    rooms = []
    for item in config['TargetSpaces']:
        identifier = str(item['id']).strip()
        if not identifier:
            raise ValueError('房间 ID 不能为空')
        role = item.get('role', 'agent')
        if role not in ('agent', 'residual'):
            raise ValueError('空间 role 必须为 agent 或 residual')
        if role == 'residual':
            seed = (free.representative_point().x, free.representative_point().y)
        elif 'seed' in item:
            seed = finite_pair(item['seed'], identifier)
        else:
            x1,y1,x2,y2 = map(float,item['initial_rect'])
            seed = finite_pair([(x1+x2)/2,(y1+y2)/2], identifier)
        area = float(item['target_area'])
        limits = finite_pair(item.get('area_range',[area*.85,area*1.15]), 'area_range')
        if role == 'residual':
            minimum = float(item.get('residual_min_area', limits[0]))
            area = max(minimum, free.area-sum(float(r['target_area']) for r in config['TargetSpaces'] if r.get('role') != 'residual'))
            limits = (minimum, free.area)
        aspects = finite_pair(item.get('aspect_range',[1,2.5]), 'aspect_range')
        if not math.isfinite(area) or area <= 0 or not 0 < limits[0] <= area <= limits[1]:
            raise ValueError('目标面积必须为正，且位于有效面积范围中')
        if not 1 <= aspects[0] <= aspects[1]:
            raise ValueError('比例采用长边/短边，范围须满足 1 <= min <= max')
        policy = item.get('shape_policy','regular')
        if policy not in ('regular','limited_recess'):
            raise ValueError('shape_policy 必须为 regular 或 limited_recess')
        min_width = item.get('min_width')
        if min_width is not None:
            min_width = float(min_width)
            if not math.isfinite(min_width) or min_width <= 0:
                raise ValueError('min_width 必须为正的有限数值')
        raw_limits = item.get('shape_limits',{})
        if not isinstance(raw_limits,dict) or set(raw_limits)-set(ShapeLimits.__dataclass_fields__):
            raise ValueError('shape_limits 包含未知参数或不是字典')
        shape_limits = ShapeLimits(**raw_limits)
        if type(shape_limits.max_reflex) is not int or shape_limits.max_reflex < 0:
            raise ValueError('max_reflex 必须是非负整数')
        if type(shape_limits.allow_long_edge_split) is not bool:
            raise ValueError('allow_long_edge_split 必须是布尔值')
        for name in ('min_rectangularity','max_notch_depth_ratio','max_extra_perimeter',
                     'min_segment_ratio','min_width_ratio','long_edge_ratio'):
            value = getattr(shape_limits,name)
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} 必须为正有限数')
        if shape_limits.min_rectangularity > 1 or shape_limits.min_width_ratio > 1:
            raise ValueError('min_rectangularity / min_width_ratio 不得大于 1')
        rooms.append(RoomSpec(identifier,seed,area,limits,aspects,policy,min_width,shape_limits,role))
    ids = {r.id for r in rooms}
    if not rooms or len(ids) != len(rooms):
        raise ValueError('房间列表不能为空，ID 不能重复')
    residual = [r for r in rooms if r.role == 'residual']
    if len(residual) > 1 or len(residual) == len(rooms):
        raise ValueError('最多设置一个剩余公共空间，并至少保留一个独立房间')
    if sum(r.area_range[0] for r in rooms) > free.area + 1e-9:
        raise ValueError('房间最小面积总和超过扣除固定结构后的自由面积')
    relations = []
    for edge in config.get('FunctionalRelations',[]):
        source,target,kind = str(edge['from']),str(edge['to']),edge['type']
        if source not in ids or target not in ids or source == target:
            raise ValueError('图关系必须连接两个不同的有效房间')
        if kind not in ('none','adjacent','connected','via_circulation','separate'):
            raise ValueError(f'不支持的关系类型：{kind}')
        if kind == 'none':
            continue
        values = []
        for key in ('min_shared_length','min_clear_width','min_distance'):
            value = edge.get(key)
            if value is not None:
                value = float(value)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f'{key} 必须为正的有限数值')
            values.append(value)
        relations.append(Relation(source,target,kind,*values))
    grid = float(config.get('SeedGrowth',{}).get('grid_size',config['AdaptiveReuseEnvironment'].get('grid_size',.5)))
    if not math.isfinite(grid) or grid <= 0:
        raise ValueError('种子网格尺寸必须为正的有限数值')
    entrance = clearance = None
    if residual:
        raw = building.get('door_positions')
        if raw is None or len(raw) != 2:
            raise ValueError('剩余客厅需要 ExistingBuilding.door_positions 指定户门两个端点')
        entrance = LineString([finite_pair(p, 'door_positions') for p in raw])
        if entrance.length <= 1e-8 or entrance.difference(boundary.boundary.buffer(1e-7)).length > 1e-7:
            raise ValueError('户门整段必须位于建筑边界上')
        depth = float(config.get('SeedGrowth', {}).get('entrance_depth', .9))
        if not math.isfinite(depth) or depth <= 0:
            raise ValueError('entrance_depth 必须为正有限数')
        clearance = entrance.buffer(depth, cap_style=2).intersection(boundary)
        if clearance.area <= 1e-8 or clearance.intersection(fixed).area > 1e-8:
            raise ValueError('户门内侧预留区域被固定结构阻挡，请调整户门位置或预留深度')
    fitting_tolerance = float(config.get('SeedGrowth', {}).get('fitting_area_tolerance', .12))
    if not math.isfinite(fitting_tolerance) or not 0 <= fitting_tolerance <= .5:
        raise ValueError('fitting_area_tolerance 必须在 0 到 0.5 之间')
    result = Problem(tuple(rooms),tuple(relations),boundary,fixed,free,grid,
                     entrance=entrance, entrance_clearance=clearance,
                     fitting_area_tolerance=fitting_tolerance)
    result.validate_seeds({r.id:r.seed for r in result.active_rooms})
    return result


@dataclass(frozen=True)
class SeedSnapshot:
    completed_episodes: int
    step: int
    seeds: tuple[tuple[str, tuple[float, float]], ...]

    @classmethod
    def capture(cls, problem, seeds, completed_episodes, step=0):
        if type(completed_episodes) is not int or type(step) is not int or min(completed_episodes,step) < 0:
            raise ValueError('轮次和步数必须是非负整数')
        problem.validate_seeds(seeds)
        return cls(completed_episodes,step,tuple(sorted((k,finite_pair(v,k)) for k,v in seeds.items())))

    def to_dict(self):
        return dict(schema_version=1,completed_episodes=self.completed_episodes,step=self.step,
                    seeds={k:list(v) for k,v in self.seeds})

    @property
    def key(self):
        return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True).encode()).hexdigest()


class PreciseSchedule:
    """Pure scheduling only: no expensive decoding is called from training steps."""
    interval = 250

    def __init__(self, completed_keys=()):
        self.completed_keys = set(completed_keys)

    def due(self, snapshot, reason='periodic'):
        if reason not in ('periodic','finished','stop_requested','interrupted'):
            raise ValueError('未知的精确成图触发原因')
        if snapshot.key in self.completed_keys:
            return False
        if reason != 'periodic':
            return True
        return snapshot.step == 0 and snapshot.completed_episodes > 0 and snapshot.completed_episodes % self.interval == 0

    def mark_completed(self, snapshot):
        """Call only after artifacts are safely written; failed work remains due."""
        self.completed_keys.add(snapshot.key)

    def to_dict(self):
        return dict(interval=self.interval,completed_keys=sorted(self.completed_keys))
