"""Independent geometric acceptance; never treats a proxy score as evidence."""
from shapely.geometry import Point
from shapely.ops import unary_union
from .shape_rules import polygon_parts
import math
from .shape_rules import check_shape


def _shared_length(a, b, tolerance):
    """Collinear boundary overlap, allowing only numerical coordinate error.

    Unlike buffered intersection length, corner-only contact contributes zero.
    """
    total = 0.
    def segments(g):
        return [(p,q) for poly in polygon_parts(g) for ring in (poly.exterior,*poly.interiors)
                for p,q in zip(ring.coords,list(ring.coords)[1:])]
    aa, bb = segments(a), segments(b)
    for p, q in aa:
        dx, dy = q[0]-p[0], q[1]-p[1]
        length = math.hypot(dx,dy)
        if length <= tolerance:
            continue
        ux,uy = dx/length,dy/length
        for r,s in bb:
            if max(abs((v[0]-p[0])*uy-(v[1]-p[1])*ux) for v in (r,s)) > tolerance:
                continue
            t0,t1 = sorted((v[0]-p[0])*ux+(v[1]-p[1])*uy for v in (r,s))
            total += max(0.,min(length,t1)-max(0.,t0))
    return total


def validate_layout(problem, seeds, polygons, tolerance=1e-7):
    problem.validate_seeds(seeds)
    errors, metrics, relations = [], {}, []
    ids = {r.id for r in problem.rooms}
    if set(polygons) != ids:
        errors.append('room_ids_mismatch')
    usable = {}
    for room in problem.rooms:
        p = polygons.get(room.id)
        residual = room.role == 'residual'
        if p is None or p.is_empty or not p.is_valid or (not residual and (p.geom_type != 'Polygon' or len(p.interiors))):
            errors.append(f'{room.id}:invalid_or_non_simple_polygon')
            continue
        usable[room.id] = p
        if not residual and not p.covers(Point(seeds[room.id])):
            errors.append(f'{room.id}:seed_not_contained')
        if p.difference(problem.boundary).area > tolerance:
            errors.append(f'{room.id}:outside_boundary')
        if p.intersection(problem.fixed).area > tolerance:
            errors.append(f'{room.id}:fixed_structure_overlap')
        if p.area < room.area_range[0] - tolerance:
            errors.append(f'{room.id}:area_below_minimum')
        if p.area > room.area_range[1] + tolerance:
            errors.append(f'{room.id}:area_above_maximum')
        x0, y0, x1, y1 = p.bounds
        aspect = max(x1-x0, y1-y0)/min(x1-x0, y1-y0)
        if residual:
            shape_errors,shape = [],dict(policy='residual',components=len(polygon_parts(p)))
            if p.geom_type != 'Polygon': shape_errors.append('public_space_disconnected')
            if room.min_width:
                core=p.buffer(-room.min_width/2+tolerance,join_style=2)
                if core.is_empty or len(polygon_parts(core)) != 1:
                    shape_errors.append('public_space_narrow_or_disconnected')
            if problem.entrance is not None and not p.buffer(tolerance).covers(problem.entrance):
                shape_errors.append('entrance_not_in_public_space')
            if problem.entrance_clearance is not None and problem.entrance_clearance.difference(p).area > tolerance:
                shape_errors.append('entrance_clearance_blocked')
        else:
            shape_errors,shape = check_shape(p,room,problem.fixed,tolerance)
            if not room.aspect_range[0]-tolerance <= aspect <= room.aspect_range[1]+tolerance:
                shape_errors.append('aspect_out_of_range')
        errors.extend(f'{room.id}:{error}' for error in shape_errors)
        metrics[room.id] = dict(area=p.area, target_area=room.target_area,
                               area_error=p.area-room.target_area,
                               rectangularity=p.area/p.envelope.area, aspect=aspect,
                               aspect_in_range=room.aspect_range[0] <= aspect <= room.aspect_range[1],shape=shape)
    keys = sorted(usable)
    for i, first in enumerate(keys):
        for second in keys[i+1:]:
            if usable[first].intersection(usable[second]).area > tolerance:
                errors.append(f'{first}/{second}:room_overlap')
    for edge in problem.relations:
        a, b = usable.get(edge.source), usable.get(edge.target)
        shared, distance, satisfied = None, None, None
        if a is not None and b is not None:
            shared = _shared_length(a,b,tolerance)
            distance = a.distance(b)
            if edge.kind == 'adjacent':
                satisfied = shared > tolerance if edge.min_shared_length is None else shared >= edge.min_shared_length-tolerance
            elif edge.kind == 'separate':
                satisfied = distance > tolerance if edge.min_distance is None else distance >= edge.min_distance-tolerance
        # Door connectivity and corridor access require additional evidence (CP5+).
        relations.append(dict(source=edge.source, target=edge.target, kind=edge.kind,
                              shared_length=shared, distance=distance, satisfied=satisfied))
    uncovered = problem.free_space.difference(unary_union(list(usable.values()))).area
    if problem.residual_room and uncovered > tolerance: errors.append('unassigned_free_space')
    public=usable.get(problem.residual_room.id) if problem.residual_room else None
    entrance_ok=None if problem.entrance is None else bool(public is not None and public.buffer(tolerance).covers(problem.entrance) and problem.entrance_clearance.difference(public).area <= tolerance)
    return dict(estimated=False, unassigned_area=uncovered, entrance_in_public_space=entrance_ok, geometry_valid=not errors, errors=errors, rooms=metrics,
                relations=relations, relations_satisfied=all(r['satisfied'] is True for r in relations))
