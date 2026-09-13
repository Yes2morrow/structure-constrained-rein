"""Continuous-coordinate, simultaneous whole-edge rectangular growth (CP3 base).

An offline decoder, deliberately absent from the training step. Numerical search
clips complete edges against full-thickness geometry, never rasterized obstacles.
This bounded heuristic may stall on feasible problems; it does not prove infeasibility.
"""
import math
from dataclasses import replace
from shapely.ops import unary_union
from shapely.geometry import Point, box

from .validation import validate_layout


def _interpolate(a, b, fraction):
    return tuple(x+(y-x)*fraction for x, y in zip(a, b))


def _clip(predicate):
    if predicate(1.):
        return 1.
    lo, hi = 0., 1.
    for _ in range(48):
        mid = (lo+hi)/2
        if predicate(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _initial(problem, seeds):
    result = {}
    for room in problem.rooms:
        p = Point(seeds[room.id])
        clearance = p.distance(problem.free_space.boundary)
        neighbors = [p.distance(Point(v)) for k, v in seeds.items() if k != room.id]
        radius = min([clearance/4, math.sqrt(room.target_area)/20, problem.grid_size/20]
                     + [d/8 for d in neighbors])
        if radius <= 1e-10:
            raise ValueError('种子间距或结构净距低于精确生长数值精度')
        x, y = seeds[room.id]
        result[room.id] = (x-radius, y-radius, x+radius, y+radius)
    return result


def _grow(problem, seeds, phase, step, max_rounds, initial=None, caps=None):
    rooms = {r.id:r for r in problem.rooms}
    keys = sorted(rooms)
    bounds = _initial(problem, seeds) if initial is None else dict(initial)
    caps = caps or {}
    stalled = 0
    for iteration in range(max_rounds):
        direction = (iteration+phase) % 4
        proposals = {}
        for key in keys:
            old = bounds[key]
            width, height = old[2]-old[0], old[3]-old[1]
            remaining = max(0., rooms[key].target_area-width*height)
            delta = min(step, remaining/(height if direction in (0,2) else width))
            new = list(old)
            new[direction] += delta * (-1 if direction < 2 else 1)
            if (key,direction) in caps:
                cap = caps[key,direction]
                new[direction] = max(new[direction],cap) if direction < 2 else min(new[direction],cap)
            def allowed(f):
                p = box(*_interpolate(old, new, f))
                return (problem.boundary.covers(p) and p.intersection(problem.fixed).area == 0
                        and all(p.intersection(box(*bounds[other])).area == 0
                                for other in keys if other != key))
            proposals[key] = _interpolate(old, new, _clip(allowed))
        # All pair limits are computed from the same state. Neither input order
        # nor sequential claiming gives one room first access to contested space.
        factors = dict.fromkeys(keys, 1.)
        for i, a in enumerate(keys):
            for b in keys[i+1:]:
                def disjoint(f):
                    pa = box(*_interpolate(bounds[a], proposals[a], f))
                    pb = box(*_interpolate(bounds[b], proposals[b], f))
                    return pa.intersection(pb).area == 0
                factor = _clip(disjoint)
                factors[a] = min(factors[a], factor)
                factors[b] = min(factors[b], factor)
        updated = {k:_interpolate(bounds[k], proposals[k], factors[k]) for k in keys}
        progress = sum(box(*updated[k]).area-box(*bounds[k]).area for k in keys)
        bounds = updated
        stalled = stalled+1 if progress < 1e-8 else 0
        if stalled >= 4:
            break
    return {k:box(*v) for k,v in bounds.items()}, iteration+1


def _score(problem, polygons, report):
    deficit = sum(abs(polygons[r.id].area-r.target_area)/r.target_area for r in problem.rooms)
    aspect_loss = sum(max(0., r.aspect_range[0]-report['rooms'][r.id]['aspect'],
                          report['rooms'][r.id]['aspect']-r.aspect_range[1]) for r in problem.rooms)
    relation_failures = sum(r['satisfied'] is False for r in report['relations'])
    # Sub-precision area noise must not trigger a nominal improvement/retreat.
    return (len(report['errors']), relation_failures, round(deficit,10), round(aspect_loss,10))


def _retreat_candidates(problem, seeds, polygons, step):
    """Offer a strip from a donor that can retain its minimum area and seed."""
    for room in sorted(problem.rooms, key=lambda r:r.id):
        p = polygons[room.id]
        x0,y0,x1,y1 = p.bounds
        spare = p.area-room.area_range[0]
        if spare <= 1e-8:
            continue
        for side in range(4):
            old = list(p.bounds)
            depth = min(step, spare/(y1-y0 if side in (0,2) else x1-x0))
            new = list(old)
            new[side] += depth*(1 if side < 2 else -1)
            q = box(*new)
            if not q.contains(Point(seeds[room.id])) or q.area < room.area_range[0]-1e-8:
                continue
            if room.min_width is not None and min(new[2]-new[0],new[3]-new[1]) < room.min_width:
                continue
            initial = {k:v.bounds for k,v in polygons.items()}
            initial[room.id] = tuple(new)
            yield initial, {(room.id,side):new[side]}


def decode_layout(problem, snapshot, step=None, max_rounds=2000, retreat_passes=2, refine=True):
    """Try four direction phases; discard inferior trials atomically.

    Returned candidates can be invalid in area/relations and are explicitly
    reported as such. Structural collisions are never repaired by clipping away
    pieces, which would incorrectly introduce CP4 notches or disconnected rooms.
    """
    if problem.residual_room:
        from .graph_partition import fit_graph_faces
        active_ids = {r.id for r in problem.active_rooms}
        reserved = unary_union([problem.fixed,problem.entrance_clearance])
        private_problem = replace(problem,rooms=problem.active_rooms,
            relations=tuple(e for e in problem.relations if e.source in active_ids and e.target in active_ids),
            fixed=reserved,free_space=problem.boundary.difference(reserved))
        result = decode_layout(private_problem,snapshot,step,max_rounds,retreat_passes,refine)
        polygons,records = fit_graph_faces(problem,dict(snapshot.seeds),result['polygons'])
        report = validate_layout(problem,dict(snapshot.seeds),polygons)
        result.update(polygons=polygons,validation=report,graph_fitting=records,
            decoder_version='graph_faces_residual_v1',
            limitations=['bounded_heuristic_not_feasibility_proof','no_interior_door_or_corridor_proof'],
            status='valid' if report['geometry_valid'] and report['relations_satisfied'] else 'unresolved')
        return result
    seeds = dict(snapshot.seeds)
    problem.validate_seeds(seeds)
    step = problem.grid_size if step is None else float(step)
    if not math.isfinite(step) or step <= 0 or type(max_rounds) is not int or max_rounds < 4:
        raise ValueError('step 必须为正有限数，max_rounds 必须为至少 4 的整数')
    if type(retreat_passes) is not int or not 0 <= retreat_passes <= 10:
        raise ValueError('retreat_passes 必须是 0 到 10 的整数')
    trials = []
    for phase in range(4):
        polygons, rounds = _grow(problem, seeds, phase, step, max_rounds)
        report = validate_layout(problem, seeds, polygons)
        score = _score(problem, polygons, report)
        trials.append((score, phase, polygons, report, rounds))
    _, phase, polygons, report, rounds = min(trials, key=lambda t:t[0])
    accepted = 0
    attempted = 0
    for _ in range(retreat_passes):
        best = (_score(problem,polygons,report),polygons,report)
        # No allocation to repair when every room reaches its target and no
        # known geometric/graph requirement fails.
        if not report['errors'] and best[0][1] == 0 and best[0][2] < 1e-7:
            break
        for initial, caps in _retreat_candidates(problem,seeds,polygons,step):
            candidate, _ = _grow(problem,seeds,phase,step,max_rounds,initial,caps)
            checked = validate_layout(problem,seeds,candidate)
            score = _score(problem,candidate,checked)
            attempted += 1
            if score < best[0]:
                best = (score,candidate,checked)
        if best[1] is polygons:
            break
        _,polygons,report = best
        accepted += 1
    from .graph_partition import fit_graph_faces
    polygons,graph_fitting = fit_graph_faces(problem,seeds,polygons)
    report = validate_layout(problem,seeds,polygons)
    refinement = []
    if refine:
        from .edge_growth import refine_layout
        polygons,refinement = refine_layout(problem,seeds,polygons)
        report = validate_layout(problem,seeds,polygons)
    return dict(decoder_version='whole_edge_plus_segments_v2' if refine else 'whole_edge_v1', estimated=False, snapshot=snapshot.to_dict(),
                polygons=polygons, validation=report, selected_phase=phase, rounds=rounds,
                status='valid' if report['geometry_valid'] and report['relations_satisfied'] else 'unresolved',
                retreat_attempts=attempted, retreat_accepted=accepted,
                refinement=refinement,graph_fitting=graph_fitting,
                limitations=([] if refine else ['rectangles_only'])+['bounded_heuristic_not_feasibility_proof', 'no_door_or_corridor_proof'],
                trials=[dict(phase=t[1], score=list(t[0]), rounds=t[4]) for t in trials])
