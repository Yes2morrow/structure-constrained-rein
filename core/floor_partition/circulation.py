"""住宅楼层分区的走道生成。"""

from __future__ import annotations

from dataclasses import dataclass
import math

from shapely.geometry import Polygon, box, LineString
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
    return box(start, max_y, end, max_y + width)


def generate_residential_circulation(problem: FloorPartitionProblem) -> CirculationLayout:
    """Full feasible contour envelope; the partition solver trims its extent."""
    return list(circulation_candidates(problem))[-1]


def circulation_candidates(problem):
    """Offset the actual core contour, then try short-to-long connected bands.

    Width is never reduced to squeeze around obstacles. Opening is the real
    core/corridor interface. Candidates still need doors and unit validation.
    """
    side = _choose_opening_side(problem)
    core = problem.traffic_core
    if core.geom_type != 'Polygon':
        raise ValueError('当前自动走道要求交通核连通；分离交通核需先指定连接方案')
    # Generate centre-line territories wide enough for two half-wall deductions.
    width = problem.profile.corridor_width + problem.profile.wall_thickness
    width = math.ceil(width/problem.profile.grid_size-1e-8)*problem.profile.grid_size
    blocked = unary_union([fixed_polygon(i) for i in problem.fixed_objects
                           if i['type'] not in TRAFFIC_CORE_TYPES])
    ring = core.buffer(width, join_style=2).difference(core).intersection(problem.boundary).difference(blocked)
    bx,by,ex,ey = problem.boundary.bounds
    cx,cy = core.centroid.coords[0]
    masks = {'south':box(bx-1,by-1,ex+1,cy), 'north':box(bx-1,cy,ex+1,ey+1),
             'west':box(bx-1,by-1,cx,ey+1), 'east':box(cx,by-1,ex+1,ey+1)}
    ring = ring.intersection(masks[side]).buffer(0)
    opening = problem.entrance
    if opening is None or opening.difference(core.boundary.buffer(1e-7)).length > 1e-6:
        edges=[]
        for a,b in zip(core.exterior.coords,list(core.exterior.coords)[1:]):
            edge=LineString([a,b]); mid=edge.interpolate(.5,normalized=True)
            horizontal=abs(a[1]-b[1]) < 1e-7
            if horizontal != (side in ('south','north')) or edge.length < problem.profile.opening_width:
                continue
            if edge.intersection(ring.buffer(1e-7)).length >= problem.profile.opening_width:
                edges.append(edge)
        if not edges:
            raise ValueError('交通核没有足宽且可连接走道的开口边')
        edge=min(edges,key=lambda e:e.distance(core.centroid))
        mid=edge.length/2; half=problem.profile.opening_width/2
        opening=LineString([edge.interpolate(mid-half),edge.interpolate(mid+half)])
    horizontal=side in ('south','north')
    center=opening.centroid.x if horizontal else opening.centroid.y
    lo,hi=(ring.bounds[0],ring.bounds[2]) if horizontal else (ring.bounds[1],ring.bounds[3])
    maximum=2*max(center-lo,hi-center)
    minimum=min(maximum,max(problem.profile.opening_width,_required_length(problem)*.65))
    seen=set(); yielded=False
    # One-metre span increments keep the search small; the final extent is exact.
    lengths=[minimum,maximum]+list(range(math.ceil(minimum),math.ceil(maximum)))
    for length in sorted(set(lengths)):
        clip=box(center-length/2,by-1,center+length/2,ey+1) if horizontal else box(bx-1,center-length/2,ex+1,center+length/2)
        g=ring.intersection(clip).buffer(0).simplify(1e-8,preserve_topology=True)
        if g.is_empty or g.geom_type != 'Polygon' or g.wkb in seen:
            continue
        if not g.buffer(1e-7).covers(opening):
            continue
        # A connected centre-space is required; disconnected slivers/narrow
        # obstacle bypasses are not silently accepted as circulation.
        centre=g.buffer(-width/2+1e-5,join_style=2)
        if centre.is_empty or centre.geom_type != 'Polygon':
            continue
        seen.add(g.wkb); yielded=True
        yield CirculationLayout(g,opening,side)
    if not yielded:
        raise ValueError('无法生成连接交通核且满足净宽的连续走道；请检查柱位、边界和开口')
