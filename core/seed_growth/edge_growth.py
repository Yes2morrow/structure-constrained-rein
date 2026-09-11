"""CP4 additive edge candidates. Never subtract obstacles and keep a fragment."""
from dataclasses import dataclass
import math
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from .shape_rules import polygon_parts, long_edge_eligible, check_shape


@dataclass(frozen=True)
class Edge:
    normal: tuple
    low: float
    high: float
    level: float

    def local(self, point):
        nx,ny = self.normal
        x,y = point
        return ny*x-nx*y,nx*x+ny*y

    def world(self, u, v):
        nx,ny = self.normal
        return ny*u+nx*v,-nx*u+ny*v


def exposed_edges(polygon):
    coords = list(orient(polygon.simplify(0),sign=1).exterior.coords)
    for a,b in zip(coords,coords[1:]):
        dx,dy = b[0]-a[0],b[1]-a[1]
        if abs(dx) > 1e-8 and abs(dy) > 1e-8:
            continue  # Slanted faces already aligned to a wall stay fixed.
        length = math.hypot(dx,dy)
        if length <= 1e-8:
            continue
        normal = (round(dy/length),round(-dx/length))
        frame = Edge(normal,0,0,0)
        u0,v = frame.local(a); u1,_ = frame.local(b)
        yield Edge(normal,min(u0,u1),max(u0,u1),v)


def _roof_polygon(edge, roof):
    points = [(edge.low,edge.level),(edge.high,edge.level)] + list(reversed(roof))
    return Polygon([edge.world(u,v) for u,v in points])


def slant_candidates(polygon, edge, fixed):
    """A: supporting face; B: cap at the finite far endpoint projection.

    All endpoints come from the thick obstacle exterior. The low contact end
    must lie on the finite face. Only the high end may extend beyond that face.
    The caller checks the entire added wedge against all geometry.
    """
    for obstacle in polygon_parts(fixed):
        vertices = [edge.local(p) for p in obstacle.exterior.coords]
        for a,b in zip(vertices,vertices[1:]):
            if abs(a[0]-b[0]) < 1e-8 or abs(a[1]-b[1]) < 1e-8:
                continue
            left,right = sorted((a,b))
            if right[0] <= edge.low or left[0] >= edge.high:
                continue
            slope = (right[1]-left[1])/(right[0]-left[0])
            def height(u):
                return left[1]+slope*(u-left[0])
            low_height,high_height = height(edge.low),height(edge.high)
            first_u = edge.low if low_height <= high_height else edge.high
            if not left[0]-1e-8 <= first_u <= right[0]+1e-8:
                continue
            if min(low_height,high_height) < edge.level-1e-8:
                continue
            roof = [(edge.low,max(edge.level,low_height)),(edge.high,max(edge.level,high_height))]
            strip = _roof_polygon(edge,roof)
            if strip.is_valid and strip.area > 1e-8:
                yield 'slant_A',polygon.union(strip)
            far = right if slope > 0 else left
            if not edge.low+1e-8 < far[0] < edge.high-1e-8:
                continue
            roof = [(edge.low,min(roof[0][1],far[1])),far,
                    (edge.high,min(roof[1][1],far[1]))]
            strip = _roof_polygon(edge,roof)
            if strip.is_valid and strip.area > 1e-8:
                yield 'slant_B',polygon.union(strip)


def split_candidate(polygon, edge, fixed, depth, room):
    """First-hit rectangular obstacle projections; blocked intervals stay put."""
    blocks = []
    for obstacle in polygon_parts(fixed):
        local = Polygon([edge.local(p) for p in obstacle.exterior.coords])
        if not local.equals(local.envelope):
            continue
        x0,y0,x1,y1 = local.bounds
        if x1 <= edge.low or x0 >= edge.high or y1 <= edge.level or y0 >= edge.level+depth:
            continue
        blocks.append((max(edge.low,x0),min(edge.high,x1),max(edge.level,y0)))
    if not blocks:
        return None
    if room.shape_policy != 'limited_recess' and not long_edge_eligible(room,edge.high-edge.low,len(blocks)):
        return None
    events = sorted({edge.low,edge.high} | {v for b in blocks for v in b[:2]})
    parts = [polygon]
    for lo,hi in zip(events,events[1:]):
        mid = (lo+hi)/2
        top = min([edge.level+depth]+[v for a,b,v in blocks if a < mid < b])
        if top > edge.level+1e-8:
            parts.append(Polygon([edge.world(lo,edge.level),edge.world(hi,edge.level),
                                  edge.world(hi,top),edge.world(lo,top)]))
    return unary_union(parts).simplify(0)


def _area_limit(base, candidate, edge, target):
    if candidate.area <= target+1e-9:
        return candidate,False
    # Truncate only the added normal extent, not the existing room. This gives
    # a continuous monotone area search even for a wall-aligned trapezoid.
    hi = max(edge.local(p)[1] for p in candidate.exterior.coords)-edge.level
    if hi <= 0:
        return base,True
    lo = 0.
    accepted = base
    for _ in range(48):
        mid = (lo+hi)/2
        strip = _roof_polygon(edge,[(edge.low,edge.level+mid),(edge.high,edge.level+mid)])
        proposal = base.union(candidate.intersection(strip))
        if proposal.area <= target:
            accepted,lo = proposal,mid
        else:
            hi = mid
    return accepted.simplify(0),True


def _whole_candidate(base, edge, depth, problem, others):
    def at(distance):
        strip = _roof_polygon(edge,[(edge.low,edge.level+distance),(edge.high,edge.level+distance)])
        return base.union(strip)
    def safe(p):
        return (p.difference(problem.boundary).area <= 1e-10
                and p.intersection(problem.fixed).area <= 1e-10
                and all(p.intersection(other).area <= 1e-10 for other in others))
    full=at(depth)
    if safe(full):
        return full
    lo,hi,accepted=0.,depth,base
    for _ in range(48):
        mid=(lo+hi)/2
        p=at(mid)
        if safe(p):
            lo,accepted=mid,p
        else:
            hi=mid
    return accepted.simplify(0)


def refine_layout(problem, seeds, polygons, max_rounds=32):
    """Bounded joint refinement with symmetric rejection of conflicting claims."""
    current = dict(polygons)
    records = []
    rooms = {r.id:r for r in problem.rooms}
    for iteration in range(max_rounds):
        proposals, kinds = {}, {}
        for key in sorted(current):
            base,room = current[key],rooms[key]
            if base.area >= room.target_area-1e-7:
                continue
            best = None
            _,base_metrics=check_shape(base,room,problem.fixed)
            def shape_cost(m):
                return .10*(1-m['rectangularity'])+.05*m['extra_perimeter']+.01*m['reflex_count']
            for edge in exposed_edges(base):
                remaining = room.target_area-base.area
                depth = min(math.sqrt(room.target_area),max(problem.grid_size,remaining/(edge.high-edge.low)))
                candidates = list(slant_candidates(base,edge,problem.fixed))
                candidates.append(('whole_segment',_whole_candidate(base,edge,depth,problem,
                    [p for k,p in current.items() if k != key])))
                split = split_candidate(base,edge,problem.fixed,depth,room)
                if split is not None:
                    candidates.append(('split',split))
                for kind,candidate in candidates:
                    if candidate.geom_type != 'Polygon' or not candidate.is_valid or candidate.interiors:
                        continue
                    candidate,limited = _area_limit(base,candidate,edge,room.target_area)
                    if candidate.geom_type != 'Polygon' or not candidate.is_valid or candidate.interiors:
                        continue
                    gain = candidate.area-base.area
                    if gain < 1e-7 or not candidate.covers(Point(seeds[key])):
                        continue
                    if (candidate.difference(problem.boundary).area > 1e-8
                            or candidate.intersection(problem.fixed).area > 1e-8
                            or any(candidate.intersection(p).area > 1e-8 for k,p in current.items() if k != key)):
                        continue
                    errors,metrics = check_shape(candidate,room,problem.fixed)
                    if errors:
                        continue
                    # Explicit experimental utility: small area gains cannot
                    # buy many notches or large perimeter/rectangularity loss.
                    utility = gain/room.target_area-max(0.,shape_cost(metrics)-shape_cost(base_metrics))
                    if utility <= 0:
                        continue
                    score = (round(utility,10),-metrics['reflex_count'])
                    if best is None or score > best[0]:
                        best = (score,candidate,kind+('_area_limited' if limited else ''))
            if best is not None:
                _,proposals[key],kinds[key] = best
        rejected = set()
        keys = sorted(proposals)
        for i,a in enumerate(keys):
            for b in keys[i+1:]:
                if proposals[a].intersection(proposals[b]).area > 1e-8:
                    rejected.update((a,b))
        accepted = [k for k in keys if k not in rejected]
        # Refinement must not destroy a graph relation already proven true.
        # Keep the transaction atomic when several independent additions alter
        # a separation distance together.
        rollback = []
        if accepted:
            from .validation import validate_layout
            trial = dict(current)
            trial.update({k:proposals[k] for k in accepted})
            before = validate_layout(problem,seeds,current)
            after = validate_layout(problem,seeds,trial)
            rollback = [f'{a["source"]}/{a["target"]}:{a["kind"]}'
                        for a,b in zip(before['relations'],after['relations'])
                        if a['satisfied'] is True and b['satisfied'] is not True]
            if set(after['errors'])-set(before['errors']):
                rollback.append('new_validation_errors')
            if rollback:
                accepted = []
        records.append(dict(round=iteration+1,accepted={k:kinds[k] for k in accepted},
                            conflict_rejected=sorted(rejected),validation_rollback=rollback))
        if not accepted:
            break
        current.update({k:proposals[k] for k in accepted})
    return current,records
