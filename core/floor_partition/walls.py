"""Centre-line territories, physical partition walls, clear floor and thresholds.

Territories remain a complete partition of existing free space. Only INTERNAL
shared boundaries acquire new walls; existing structure and exterior envelopes
are not thickened again. Net floor excludes the entire wall reservation, with
door thresholds reported separately from the solid wall.
"""
from dataclasses import dataclass, replace
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union, linemerge
from shapely.affinity import translate

from .contracts import target_areas_for_area
from core.envs.structure_geometry import fixed_polygon


@dataclass(frozen=True)
class WallGeometry:
    centerlines: object
    reservation: object
    solid: object
    thresholds: object
    units: dict
    corridor: object


def wall_geometry(problem, corridor, units, doors=()):
    parts=[corridor,*units.values()]
    shared=[a.boundary.intersection(b.boundary) for i,a in enumerate(parts) for b in parts[i+1:]]
    def line_parts(g):
        if g.geom_type=='LineString': return [g]
        return [p for child in getattr(g,'geoms',[]) for p in line_parts(child)]
    centerlines=unary_union([line for g in shared for line in line_parts(g)])
    if centerlines.geom_type=='MultiLineString': centerlines=linemerge(centerlines)
    half=problem.profile.wall_thickness/2
    reserve=centerlines.buffer(half,cap_style=2,join_style=2).intersection(
        problem.boundary.difference(problem.fixed_union)) if half else Polygon()
    # Square jambs; opening width is measured along the centre line.
    holes=unary_union([LineString(d.points).buffer(half+1e-7,cap_style=2,join_style=2) for d in doors]) if half else Polygon()
    thresholds=reserve.intersection(holes)
    return WallGeometry(centerlines,reserve,reserve.difference(holes),thresholds,
        {u:p.difference(reserve) for u,p in units.items()},corridor.difference(reserve))


def update_net_targets(problem,result):
    geometry=wall_geometry(problem,result.corridor,result.unit_polygons)
    areas,scale=target_areas_for_area(problem,sum(p.area for p in geometry.units.values()))
    return replace(result,target_areas=dict(zip((t.unit_id for t in problem.targets),areas)),area_scale=scale)


def shifted_door(points, distance, sign):
    line=LineString(points)
    (ax,ay),(bx,by)=points
    return translate(line,xoff=-(by-ay)/line.length*distance*sign,
                     yoff=(bx-ax)/line.length*distance*sign)


def unit_face_door(problem, result, door):
    """Map the centre-line opening onto the clear inner face for room export."""
    net=wall_geometry(problem,result.corridor,result.unit_polygons).units[door.unit_id]
    for sign in (-1,1):
        face=shifted_door(door.points,problem.profile.wall_thickness/2,sign)
        if face.difference(net.boundary.buffer(1e-6)).length<1e-6:
            return tuple(face.coords)
    raise ValueError(f'{door.unit_id}: no clear inner door face')


def structural_center_axes(problem):
    """Preferred centre axes; column faces and bay mid-lines are not axes."""
    axes=[set(),set()]
    for obj in problem.fixed_objects:
        kind=obj.get('type')
        p=fixed_polygon(obj)
        if kind=='column':
            axes[0].add(p.centroid.x); axes[1].add(p.centroid.y)
        elif kind in ('wall','shear_wall','bearing_wall'):
            x0,y0,x1,y1=p.bounds
            dim=0 if x1-x0<y1-y0 else 1
            axes[dim].add((p.centroid.x,p.centroid.y)[dim])
    return [sorted(a) for a in axes]


def structural_reference_axes(problem):
    """Wall CENTRE coordinates for each consistent structural alignment rule.

    low/high means left/right for a vertical wall, bottom/top for a horizontal
    wall. inner/outer describes which side of the existing structural face the
    new wall occupies. Equal physical faces require a half-thickness offset.
    """
    modes={name:[set(),set()] for name in ('center','low_inner','high_inner','low_outer','high_outer')}
    half=problem.profile.wall_thickness/2
    for obj in problem.fixed_objects:
        kind=obj.get('type')
        if kind not in ('column','wall','shear_wall','bearing_wall'): continue
        p=fixed_polygon(obj); bounds=p.bounds
        dims=(0,1) if kind=='column' else (0,) if bounds[2]-bounds[0]<bounds[3]-bounds[1] else (1,)
        for d in dims:
            modes['center'][d].add(p.centroid.coords[0][d])
            modes['low_inner'][d].add(bounds[d]+half)
            modes['high_inner'][d].add(bounds[d+2]-half)
            modes['low_outer'][d].add(bounds[d]-half)
            modes['high_outer'][d].add(bounds[d+2]+half)
    return {k:[sorted(v) for v in axes] for k,axes in modes.items()}
