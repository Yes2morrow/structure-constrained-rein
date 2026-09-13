"""Graph-directed face fitting followed by an exact residual-space partition.

Moves whole faces (including area-preserving opposite-face adjustment), even
after target area is reached. All proposals are transactions checked against
the full layout; shared length is measured on polygons, never on graph lines.
"""
from itertools import product
from shapely.geometry import box, Point
from shapely.ops import unary_union
from .validation import validate_layout


def complete_partition(problem, polygons):
    result = dict(polygons)
    residual = problem.residual_room
    if residual:
        private = [p for k,p in result.items() if k != residual.id]
        result[residual.id] = problem.free_space.difference(unary_union(private))
    return result


def _face_options(room, polygon, axis, level, positive, seed, tolerance):
    """Keep the tangential interval; offer extension or area-preserving fitting."""
    b = list(polygon.bounds)
    side = axis+2 if positive else axis
    opposite = axis if positive else axis+2
    tangent = 1-axis
    span = b[tangent+2]-b[tangent]
    trials = []
    extended = list(b); extended[side] = level; trials.append(extended)
    for area in (room.target_area, polygon.area, room.area_range[0]):
        candidate = list(extended)
        candidate[opposite] = level+(-1 if positive else 1)*area/span
        trials.append(candidate)
    seen = set()
    for candidate in trials:
        if candidate[2] <= candidate[0] or candidate[3] <= candidate[1]: continue
        key = tuple(round(v,9) for v in candidate)
        if key in seen: continue
        seen.add(key)
        p = box(*candidate)
        minimum=max(room.area_range[0],room.target_area*(1-tolerance))
        maximum=min(room.area_range[1],room.target_area*(1+tolerance))
        if minimum-1e-7 <= p.area <= maximum+1e-7 and p.covers(Point(seed)):
            yield p


def _cost(problem, polygons, report):
    failures = sum(e['satisfied'] is False for e in report['relations'])
    gap = sum(e['distance'] or 0 for e in report['relations'] if e['kind']=='adjacent' and e['satisfied'] is False)
    area = sum(abs(polygons[r.id].area-r.target_area)/r.target_area for r in problem.active_rooms)
    return len(report['errors']), failures, round(gap,7), round(area,7)


def fit_graph_faces(problem, seeds, polygons, passes=4):
    current = complete_partition(problem, polygons)
    report = validate_layout(problem, seeds, current)
    rooms = {r.id:r for r in problem.active_rooms}
    records = []
    for iteration in range(passes):
        best = (_cost(problem,current,report), current, report, None)
        for edge in sorted(problem.relations, key=lambda e:(e.source,e.target,e.kind)):
            if edge.kind != 'adjacent' or edge.source not in rooms or edge.target not in rooms: continue
            a,b = edge.source,edge.target
            pa,pb = current[a],current[b]
            if not pa.equals(pa.envelope) or not pb.equals(pb.envelope): continue
            for axis in (0,1):
                positive = pa.centroid.coords[0][axis] <= pb.centroid.coords[0][axis]
                first = pa.bounds[axis+2 if positive else axis]
                second = pb.bounds[axis if positive else axis+2]
                levels = sorted({first, second, (first+second)/2,
                                 (2*first+second)/3, (first+2*second)/3})
                for level in levels:
                    aa = list(_face_options(rooms[a],pa,axis,level,positive,seeds[a],problem.fitting_area_tolerance))
                    bb = list(_face_options(rooms[b],pb,axis,level,not positive,seeds[b],problem.fitting_area_tolerance))
                    for qa,qb in product(aa,bb):
                        trial = complete_partition(problem,dict(current,**{a:qa,b:qb}))
                        checked = validate_layout(problem,seeds,trial)
                        # Do not trade a new hard error or a proven relation for
                        # nominal contact elsewhere.
                        if set(checked['errors'])-set(report['errors']): continue
                        if any(old['satisfied'] is True and new['satisfied'] is not True
                               for old,new in zip(report['relations'],checked['relations'])): continue
                        score = _cost(problem,trial,checked)
                        if score < best[0]: best = (score,trial,checked,(a,b,axis,level))
        # Close peripheral slivers by fitting the whole room to a nearby
        # boundary face. Do not turn a thin disconnected strip into a lounge.
        if problem.residual_room:
            width=problem.residual_room.min_width or problem.grid_size
            for key,room in rooms.items():
                p=current[key]
                if not p.equals(p.envelope): continue
                for side in range(4):
                    level=problem.boundary.bounds[side]
                    if abs(p.bounds[side]-level)>width or abs(p.bounds[side]-level)<1e-8: continue
                    for q in _face_options(room,p,side%2,level,side>=2,seeds[key],problem.fitting_area_tolerance):
                        trial=complete_partition(problem,dict(current,**{key:q}))
                        checked=validate_layout(problem,seeds,trial)
                        if set(checked['errors'])-set(report['errors']): continue
                        if any(old['satisfied'] is True and new['satisfied'] is not True
                               for old,new in zip(report['relations'],checked['relations'])): continue
                        score=_cost(problem,trial,checked)
                        if score<best[0]: best=(score,trial,checked,(key,'boundary',side%2,level))
        if best[1] is current: break
        _,current,report,change = best
        records.append(dict(pass_index=iteration+1, edge=list(change[:2]),axis=change[2],level=change[3]))
    return current,records
