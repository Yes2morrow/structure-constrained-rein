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
    # Search intermediate depths too: jumping from 2.75 to 3.5 m can miss
    # the first feasible door arrangement and unnecessarily reserve floor area.
    first = math.ceil(min(depths)/step-1e-8)
    last = math.ceil(max(depths)/step-1e-8)
    depths = [i*step for i in range(first, last+1)]
    free = problem.boundary.difference(problem.fixed_union)
    from core.envs.structure_geometry import fixed_polygon
    obstacles = [fixed_polygon(o) for o in problem.fixed_objects
                 if o['type'] not in ('traffic_core','core')]
    out = []; seen = set()
    half = problem.profile.wall_thickness/2
    for depth in sorted({math.ceil(d/step-1e-8)*step for d in depths}):
        end = base+sign*depth
        start, stop = lo, hi
        # Columns at the two jambs are real obstacles, not part of the core.
        # Coordinates describe wall CENTRES. Move half a wall into the column
        # so the corridor-facing finished surface is flush with its inner face.
        for obstacle in obstacles:
            b = obstacle.bounds
            if b[normal+2] <= min(base,end)+1e-7 or b[normal] >= max(base,end)-1e-7:
                continue
            if b[tangent+2] <= opening.bounds[tangent]+1e-7 and b[tangent+2] > start and b[tangent] < stop:
                start = max(start,b[tangent+2]-half)
            elif b[tangent] >= opening.bounds[tangent+2]-1e-7 and b[tangent] < stop and b[tangent+2] > start:
                stop = min(stop,b[tangent]+half)
        if stop-start < width: continue
        g = (box(start,min(base,end),stop,max(base,end)) if horizontal else
             box(min(base,end),start,max(base,end),stop))
        # Keep all space outside the immutable structure. Never keep only the
        # largest component when an obstacle disconnects a proposed lobby.
        g = g.intersection(free)
        if g.geom_type!='Polygon' or g.is_empty or g.wkb in seen:
            continue
        # A side-column setback may leave a small pocket trapped between the
        # column, core and lobby. It cannot be assigned to a dwelling; retain
        # it as part of the public floor instead of silently losing its area.
        remaining = free.difference(g)
        for part in getattr(remaining,'geoms',[]):
            if (part.geom_type=='Polygon' and part.area < q.get('min_unit_width',1.5)**2
                    and part.boundary.intersection(g.boundary).length>1e-7):
                g = g.union(part)
        if not g.buffer(1e-7).covers(opening):
            continue
        seen.add(g.wkb); out.append(g)
    return out


def lobby_reassignments(result, lobby, limit=32, problem=None):
    """Each released connected wing goes to an adjacent dwelling as one piece."""
    released = result.corridor.difference(lobby)
    parts = [released] if released.geom_type == 'Polygon' else list(getattr(released,'geoms',[]))
    parts = [g for g in parts if g.area > 1e-7]
    # A shallower lobby can release one U-shaped connected strip. Treating that
    # strip as indivisible wrongly gives both entrance wings to the same unit.
    horizontal = result.opening_side in ('north','south')
    dim = 0 if horizontal else 1
    middle = result.opening.centroid.coords[0][dim]
    if parts:
        bx,by,ex,ey = released.bounds
        mask = box(bx-1,by-1,middle,ey+1) if horizontal else box(bx-1,by-1,ex+1,middle)
        split = []
        for part in parts:
            for piece in (part.intersection(mask),part.difference(mask)):
                split.extend([piece] if piece.geom_type=='Polygon' else
                             [g for g in getattr(piece,'geoms',[]) if g.geom_type=='Polygon'])
        parts = [g for g in split if g.area>1e-7]
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
                yield from straighten_entrance_wings(problem,result,lobby,proposed)


def straighten_entrance_wings(problem,result,lobby,units):
    """Re-cut entrance arms on a column row or bay midline, not old band steps.

    Initialization only. Dynamic actions remain confined to their local window.
    """
    from .walls import structural_center_axes
    dim=0 if result.opening_side in ('north','south') else 1
    normal=1-dim
    negative=result.opening_side in ('south','west')
    row_axes=structural_center_axes(problem)[normal]
    rows=sorted(set(row_axes+[(a+b)/2 for a,b in zip(row_axes,row_axes[1:])]))
    edge=lobby.bounds[normal if negative else normal+2]
    rows=([v for v in rows if v<edge-1e-6][-2:] if negative else
          [v for v in rows if v>edge+1e-6][:2])
    bx,by,ex,ey=problem.boundary.bounds
    mid=(lobby.bounds[normal]+lobby.bounds[normal+2])/2
    section=lobby.intersection(LineString([(bx-1,mid),(ex+1,mid)] if dim==0 else [(mid,by-1),(mid,ey+1)]))
    for row in rows:
        near=(box(bx-1,row,ex+1,ey+1) if normal==1 and negative else
              box(bx-1,by-1,ex+1,row) if normal==1 else
              box(row,by-1,ex+1,ey+1) if negative else box(bx-1,by-1,row,ey+1))
        out=dict(units)
        for high in (False,True):
            v=section.bounds[dim+2 if high else dim]
            flank=(box(v,by-1,ex+1,ey+1) if dim==0 and high else
                   box(bx-1,by-1,v,ey+1) if dim==0 else
                   box(bx-1,v,ex+1,ey+1) if high else box(bx-1,by-1,ex+1,v))
            region=near.intersection(flank)
            receivers=[u for u,p in out.items() if p.boundary.intersection(lobby.boundary).intersection(region.buffer(1e-6)).length>=problem.profile.door_width]
            if not receivers: continue
            # The outer entrance owner keeps its arm; all changes stay within
            # the union of existing dwellings, so columns/core cannot be filled.
            receiver=(max if high else min)(receivers,key=lambda u:out[u].centroid.coords[0][dim])
            for u in list(out):
                if u==receiver: continue
                take=out[u].intersection(region)
                out[u]=out[u].difference(take).buffer(0)
                out[receiver]=out[receiver].union(take).buffer(0)
        if any(g.geom_type!='Polygon' or g.is_empty for g in out.values()): continue
        out=rebalance_below_lobby(problem,result,lobby,out,cut_override=row)
        yield [lobby,*out.values()]


def rebalance_below_lobby(problem, result, lobby, units, cut_override=None):
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
    if cut_override is not None: cut=cut_override
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
    seen=set()
    for seed, _ in seeds:
        for lobby in lobby_candidates(problem, seed.opening, seed.opening_side):
            for pieces in lobby_reassignments(seed, lobby,problem=problem):
                units = dict(zip(seed.unit_polygons,pieces[1:]))
                key=tuple(g.simplify(1e-7).normalize().wkb for g in [lobby,*units.values()])
                if key in seen: continue
                seen.add(key)
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
