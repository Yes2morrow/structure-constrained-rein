"""Bounded joint replacement of a core-side band by a shared entrance lobby.

Candidates follow the actual entrance edge and setbacks, never an A1 template.
Released corridor components, dwelling boundaries and doors change together.
The caller must perform full physical validation before accepting a proposal.
"""
from itertools import product, islice
import math

from shapely.geometry import box, LineString


def lobby_candidates(problem, opening, side):
    if not problem.search_settings.get('compact_lobby', True):
        return []
    core = problem.traffic_core
    if core.geom_type != 'Polygon':
        return []
    horizontal = side in ('north', 'south')
    tangent, normal = (0, 1) if horizontal else (1, 0)
    sign = -1 if side in ('south', 'west') else 1
    vertices = list(core.exterior.coords)
    edge = next((LineString([a,b]) for a,b in zip(vertices,vertices[1:])
                 if abs(a[normal]-b[normal]) < 1e-7
                 and opening.difference(LineString([a,b]).buffer(1e-7)).length < 1e-6), None)
    if edge is None:
        return []
    lo, hi = edge.bounds[tangent], edge.bounds[tangent+2]
    base = edge.centroid.coords[0][normal]
    q = problem.settings
    width = problem.profile.corridor_width + problem.profile.wall_thickness
    clearance = max(problem.profile.door_width, q.get('door_clearance_width',1.2))
    # Include the depth needed to put doors on both side faces below a setback.
    setbacks = [sign*(v[normal]-base) for v in vertices
                if abs(v[tangent]-lo)<1e-6 or abs(v[tangent]-hi)<1e-6]
    depths = [width, width*1.5, width*2]
    # A deeper variant may be needed for the configured separation between
    # perpendicular side/front doors; it is scored against the extra area.
    depths.append(max(width*2,problem.profile.min_door_spacing+clearance+problem.profile.wall_thickness))
    depths += [s+clearance+problem.profile.wall_thickness for s in setbacks if 0<s<width*2]
    step = problem.profile.grid_size
    free = problem.boundary.difference(problem.fixed_union)
    out = []; seen = set()
    for depth in sorted({math.ceil(d/step-1e-8)*step for d in depths}):
        end = base+sign*depth
        g = (box(lo,min(base,end),hi,max(base,end)) if horizontal else
             box(min(base,end),lo,max(base,end),hi))
        # Do not silently bend around columns or amputate a disconnected piece.
        if g.difference(free).area > 1e-7 or g.wkb in seen:
            continue
        if not g.buffer(1e-7).covers(opening):
            continue
        seen.add(g.wkb); out.append(g)
    return out


def lobby_reassignments(result, lobby, limit=32, problem=None):
    """Each released connected wing goes to an adjacent dwelling as one piece."""
    released = result.corridor.difference(lobby)
    parts = [released] if released.geom_type == 'Polygon' else list(getattr(released,'geoms',[]))
    parts = [g for g in parts if g.area > 1e-7]
    units = {u:p.difference(lobby) for u,p in result.unit_polygons.items()}
    choices = []
    for part in parts:
        adjacent = [u for u,p in units.items() if p.boundary.intersection(part.boundary).length > 1e-7]
        # Give first consideration to units that otherwise lose all lobby access.
        adjacent.sort(key=lambda u: units[u].boundary.intersection(lobby.boundary).length)
        if not adjacent:
            return
        choices.append(adjacent)
    for recipients in islice(product(*choices), limit):
        proposed = dict(units)
        for part, uid in zip(parts, recipients):
            proposed[uid] = proposed[uid].union(part)
        if all(p.geom_type == 'Polygon' and not p.is_empty for p in proposed.values()):
            yield [lobby, *proposed.values()]
            if problem is not None and all(p.boundary.intersection(lobby.boundary).length>=problem.profile.door_width for p in proposed.values()):
                adjusted=rebalance_below_lobby(problem,result,lobby,proposed)
                if adjusted != proposed:
                    yield [lobby,*adjusted.values()]


def rebalance_below_lobby(problem, result, lobby, units):
    """Re-cut adjacent units away from the lobby, retaining new entrance arms.

    Two bounded sweeps, structural candidate lines and one-dimensional cuts;
    this does not attempt an unconstrained global plan repair.
    """
    from .quality import structural_axes
    horizontal=result.opening_side in ('north','south')
    dim=0 if horizontal else 1
    bx,by,ex,ey=problem.boundary.bounds
    side=result.opening_side
    old=result.corridor.bounds
    from .walls import structural_center_axes
    normal=1-dim
    negative=side in ('south','west')
    cut=(min(lobby.bounds[normal],old[normal]) if negative else
         max(lobby.bounds[normal+2],old[normal+2]))
    references=structural_center_axes(problem)[normal] or structural_axes(problem,include_wall_faces=False)[normal]
    away=[v for v in references if v<cut-1e-7] if negative else [v for v in references if v>cut+1e-7]
    if away: cut=max(away) if negative else min(away)
    masks={'south':box(bx-1,by-1,ex+1,cut),
           'north':box(bx-1,cut,ex+1,ey+1),
           'west':box(bx-1,by-1,cut,ey+1),
           'east':box(cut,by-1,ex+1,ey+1)}
    window=masks[side]
    out=dict(units)
    total=sum(g.area for g in out.values())
    weights={u:result.target_areas[u]/sum(result.target_areas.values()) for u in out}
    def error(geometries):
        return sum(abs(p.area-total*weights[u])/(total*weights[u]) for u,p in geometries.items())
    for _ in range(2):
        previous=error(out)
        best=out; score=previous
        ids=list(out)
        for i,a in enumerate(ids):
            for b in ids[i+1:]:
                if out[a].boundary.intersection(out[b].boundary).intersection(window).length<1e-6: continue
                low,high=sorted([a,b],key=lambda u:out[u].centroid.coords[0][dim])
                merged=out[a].union(out[b]).intersection(window)
                for cut in structural_axes(problem,include_wall_faces=False)[dim]:
                    if not merged.bounds[dim]<cut<merged.bounds[dim+2]: continue
                    mask=box(bx-1,by-1,cut,ey+1) if dim==0 else box(bx-1,by-1,ex+1,cut)
                    proposed=dict(out)
                    proposed[low]=out[low].difference(window).union(merged.intersection(mask))
                    proposed[high]=out[high].difference(window).union(merged.difference(mask))
                    if any(p.is_empty or p.geom_type!='Polygon' for p in proposed.values()): continue
                    current=error(proposed)
                    if current < score-1e-7: best,score=proposed,current
        out=best
        if score>=previous-1e-7: break
    return out


def compact_starts(problem, seeds):
    """Alternative feasible starts, independently checked including moved doors."""
    from dataclasses import replace
    from .circulation import CirculationLayout
    from .structured import unit_doors
    from .quality import validate_partition
    from .walls import update_net_targets
    for seed, _ in seeds:
        for lobby in lobby_candidates(problem, seed.opening, seed.opening_side):
            for pieces in lobby_reassignments(seed, lobby,problem=problem):
                units = dict(zip(seed.unit_polygons,pieces[1:]))
                candidate = update_net_targets(problem,replace(seed,corridor=lobby,
                    unit_polygons=units,allocatable_space=problem.boundary.difference(problem.fixed_union).difference(lobby)))
                pre = validate_partition(problem,candidate)
                if any('door' not in e for e in pre['errors']):
                    continue
                try:
                    candidate = replace(candidate,doors=unit_doors(problem,
                        CirculationLayout(lobby,seed.opening,seed.opening_side),units))
                except ValueError:
                    continue
                report = validate_partition(problem,candidate)
                if report['valid']:
                    yield candidate,report
