"""Independent geometric acceptance of floor partitions (metres).

Facade access is a daylight opportunity proxy, not a daylight simulation.
"""
import math
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from core.envs.structure_geometry import fixed_polygon


def settings(problem):
    values = dict(area_tolerance=.30, min_facade_length=3., facade_per_area=.12,
                  min_unit_width=1.5, max_corners=24, door_clearance_depth=1.2,
                  door_clearance_width=1.2, entrance_depth=.9, beam_width=24,
                  max_candidates=24, alignment_tolerance=1e-6)
    values.update(problem.settings)
    for key,value in values.items():
        if not isinstance(value,(float,int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'FloorPartition.quality.{key} 必须是正有限数')
    if values['area_tolerance'] >= 1:
        raise ValueError('area_tolerance 必须小于1')
    return values


def structural_axes(problem):
    axes=[set(),set()]
    for item in problem.fixed_objects:
        p=fixed_polygon(item)
        for x,y in p.exterior.coords:
            axes[0].add(round(x,8)); axes[1].add(round(y,8))
        if item['type']=='column':
            for d,v in enumerate(p.centroid.coords[0]): axes[d].add(round(v,8))
    # Bay centre lines are derived from the actual column grid, not pixel cells.
    for d in (0,1):
        centers=sorted({round(fixed_polygon(i).centroid.coords[0][d],8)
                        for i in problem.fixed_objects if i['type']=='column'})
        axes[d].update(round((a+b)/2,8) for a,b in zip(centers,centers[1:]))
        axes[d].add(round((problem.boundary.bounds[d]+problem.boundary.bounds[d+2])/2,8))
    return [sorted(a) for a in axes]


def facade(problem):
    # Core and solid columns touching the outside are not window frontage.
    return problem.boundary.exterior.difference(problem.fixed_union)


def corners(poly):
    points=list(poly.simplify(1e-7,preserve_topology=True).exterior.coords)[:-1]
    return len(points)


def frontage_requirement(area, q):
    return max(q['min_facade_length'],area*q['facade_per_area'])


def rectangle_at_door(points, depth, width, sign):
    (ax,ay),(bx,by)=points
    length=math.hypot(bx-ax,by-ay); tx,ty=(bx-ax)/length,(by-ay)/length
    nx,ny=-ty*sign,tx*sign; cx,cy=(ax+bx)/2,(ay+by)/2
    a=(cx-tx*width/2,cy-ty*width/2); b=(cx+tx*width/2,cy+ty*width/2)
    return Polygon([a,b,(b[0]+nx*depth,b[1]+ny*depth),(a[0]+nx*depth,a[1]+ny*depth)])


def door_clearance(door, unit, corridor, q):
    for sign in (-1,1):
        front=rectangle_at_door(door.points,q['door_clearance_depth'],
                                max(q['door_clearance_width'],LineString(door.points).length),sign)
        inside=rectangle_at_door(door.points,q['entrance_depth'],LineString(door.points).length,-sign)
        if corridor.buffer(1e-7).covers(front) and unit.buffer(1e-7).covers(inside):
            return True
    return False


def validate_partition(problem, result):
    q=settings(problem); errors=[]; metrics={}; exterior=facade(problem)
    units=result.unit_polygons; doors={d.unit_id:d for d in result.doors}
    expected={t.unit_id for t in problem.targets}
    if set(units)!=expected or set(doors)!=expected: errors.append('unit_or_door_ids')
    corridor=result.corridor
    if corridor.geom_type!='Polygon' or not corridor.is_valid: errors.append('corridor_disconnected')
    if corridor.intersection(problem.fixed_union).area>1e-6: errors.append('corridor_structure_overlap')
    if corridor.difference(problem.boundary).area>1e-6: errors.append('corridor_outside')
    if result.opening.is_empty or not corridor.buffer(1e-7).covers(result.opening): errors.append('core_opening_disconnected')
    if result.opening.difference(problem.traffic_core.boundary.buffer(1e-7)).length>1e-6: errors.append('opening_not_on_core')
    centre=corridor.buffer(-problem.profile.corridor_width/2+1e-5,join_style=2)
    if centre.is_empty or centre.geom_type!='Polygon': errors.append('corridor_width')
    for uid,poly in units.items():
        if poly.geom_type!='Polygon' or not poly.is_valid or poly.is_empty:
            errors.append(f'{uid}:disconnected'); continue
        if poly.intersection(problem.fixed_union).area>1e-6: errors.append(f'{uid}:structure_overlap')
        if poly.difference(result.allocatable_space).area>1e-6: errors.append(f'{uid}:outside_free_space')
        length=poly.boundary.intersection(exterior).length
        required=frontage_requirement(poly.area,q)
        error=abs(poly.area-result.target_areas[uid])/result.target_areas[uid]
        count=corners(poly)
        if length+1e-6<required: errors.append(f'{uid}:facade_deficit')
        if error>q['area_tolerance']+1e-6: errors.append(f'{uid}:area')
        if count>q['max_corners']: errors.append(f'{uid}:too_many_corners')
        eroded=poly.buffer(-q['min_unit_width']/2+1e-5,join_style=2)
        if eroded.is_empty or eroded.geom_type!='Polygon': errors.append(f'{uid}:narrow_or_disconnected')
        door=doors.get(uid)
        if door:
            line=LineString(door.points)
            if abs(line.length-problem.profile.door_width)>1e-6: errors.append(f'{uid}:door_width')
            if line.difference(poly.boundary.buffer(1e-7)).length>1e-6 or line.difference(corridor.boundary.buffer(1e-7)).length>1e-6:
                errors.append(f'{uid}:door_contact')
            if not door_clearance(door,poly,corridor,q): errors.append(f'{uid}:door_clearance')
        metrics[uid]=dict(area=poly.area,target_area=result.target_areas[uid],area_error_ratio=error,
                          facade_length=length,required_facade_length=required,facade_per_area=length/poly.area,
                          corners=count)
    polygons=list(units.values()); merged=unary_union(polygons)
    if sum(p.area for p in polygons)-merged.area>1e-6: errors.append('unit_overlap')
    missing=merged.symmetric_difference(result.allocatable_space).area
    if missing>1e-6: errors.append('coverage')
    for i,a in enumerate(result.doors):
        for b in result.doors[i+1:]:
            if LineString(a.points).distance(LineString(b.points))+1e-6<problem.profile.min_door_spacing:
                errors.append('door_spacing')
    # Only shared unit boundaries are checked: external outlines may be irregular.
    axes=structural_axes(problem); aligned=0.; shared_total=0.
    for i,a in enumerate(polygons):
        for b in polygons[i+1:]:
            shared=a.boundary.intersection(b.boundary)
            lines=[shared] if shared.geom_type=='LineString' else list(getattr(shared,'geoms',[]))
            for line in lines:
                if line.geom_type!='LineString': continue
                for p,r in zip(line.coords,list(line.coords)[1:]):
                    seg=LineString([p,r]); shared_total+=seg.length
                    ok=any(abs(p[d]-r[d])<1e-7 and any(abs(p[d]-v)<=q['alignment_tolerance'] for v in axes[d]) for d in (0,1))
                    if ok: aligned+=seg.length
                    else: errors.append('partition_edge_off_structure_grid')
    return dict(valid=not errors,errors=sorted(set(errors)),units=metrics,
                unassigned_area=missing,corridor_area=corridor.area,
                structure_alignment_ratio=aligned/shared_total if shared_total else 1.,
                daylight_method='exposed_facade_length_per_net_area_proxy')
