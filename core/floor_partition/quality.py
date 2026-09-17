"""Independent geometric acceptance of floor partitions (metres).

Facade access is a daylight opportunity proxy, not a daylight simulation.
"""
import math
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union
from core.envs.structure_geometry import fixed_polygon
from .walls import wall_geometry, structural_center_axes, structural_reference_axes, shifted_door


def settings(problem):
    values = dict(area_tolerance=.30, min_facade_length=3., facade_per_area=.12,
                  min_unit_width=1.5, max_corners=24, door_clearance_depth=1.2,
                  door_clearance_width=1.2, entrance_depth=.9, beam_width=24,
                  max_candidates=24, alignment_tolerance=1e-6,
                  daylight_depth=6., min_daylight_coverage=.25,
                  min_structure_alignment=1., max_unusable_ratio=.20, area_balance_deadband=.01)
    values.update(problem.settings)
    for key,value in values.items():
        if key=='area_balance_deadband' and value==0: continue
        if not isinstance(value,(float,int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'FloorPartition.quality.{key} 必须是正有限数')
    if values['area_tolerance'] >= 1:
        raise ValueError('area_tolerance 必须小于1')
    if values['area_balance_deadband']>values['area_tolerance']:
        raise ValueError('area_balance_deadband must not exceed area_tolerance')
    for key in ('min_daylight_coverage', 'min_structure_alignment', 'max_unusable_ratio'):
        if values[key] > 1: raise ValueError(f'{key} must be <= 1')
    return values


def structural_axes(problem, include_wall_faces=True):
    axes=[set(),set()]
    references=structural_reference_axes(problem)
    for refs in (references.values() if include_wall_faces else [references['center']]):
        for d in (0,1): axes[d].update(round(v,8) for v in refs[d])
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


def corners(poly, problem=None):
    points=list(poly.simplify(1e-7,preserve_topology=True).exterior.coords)[:-1]
    if problem is None:
        return len(points)
    # Existing facade columns indent the free floor; those mandatory corners
    # must not be treated as jagged NEW partition walls.
    return max(4, sum(Point(p).distance(problem.fixed_union)>1e-6 for p in points))


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
        half=q.get('wall_thickness',0.)/2
        front_points=tuple(shifted_door(door.points,half,sign).coords)
        inside_points=tuple(shifted_door(door.points,half,-sign).coords)
        front=rectangle_at_door(front_points,q['door_clearance_depth'],
                                max(q['door_clearance_width'],LineString(door.points).length),sign)
        inside=rectangle_at_door(inside_points,q['entrance_depth'],LineString(door.points).length,-sign)
        if corridor.buffer(1e-7).covers(front) and unit.buffer(1e-7).covers(inside):
            return True
    return False


def validate_partition(problem, result):
    q=settings(problem); errors=[]; metrics={}; exterior=facade(problem)
    units=result.unit_polygons; doors={d.unit_id:d for d in result.doors}
    expected={t.unit_id for t in problem.targets}
    if set(units)!=expected or set(doors)!=expected: errors.append('unit_or_door_ids')
    corridor=result.corridor
    physical=wall_geometry(problem,corridor,units,result.doors)
    net_corridor=physical.corridor
    q['wall_thickness']=problem.profile.wall_thickness
    actual_free=problem.boundary.difference(problem.fixed_union).difference(corridor)
    if actual_free.symmetric_difference(result.allocatable_space).area>1e-6:
        errors.append('allocatable_space_mismatch')
    if len(result.doors)!=len(expected): errors.append('duplicate_or_missing_doors')
    if corridor.geom_type!='Polygon' or not corridor.is_valid: errors.append('corridor_disconnected')
    if corridor.intersection(problem.fixed_union).area>1e-6: errors.append('corridor_structure_overlap')
    if corridor.difference(problem.boundary).area>1e-6: errors.append('corridor_outside')
    if result.opening.is_empty or not corridor.buffer(1e-7).covers(result.opening): errors.append('core_opening_disconnected')
    if not net_corridor.buffer(1e-7).covers(result.opening): errors.append('net_core_opening_disconnected')
    if result.opening.difference(problem.traffic_core.boundary.buffer(1e-7)).length>1e-6: errors.append('opening_not_on_core')
    if net_corridor.geom_type!='Polygon' or net_corridor.is_empty:
        errors.append('net_corridor_disconnected')
    recess_area=0.
    centre=net_corridor.buffer(-problem.profile.corridor_width/2+1e-5,join_style=2)
    if centre.is_empty or centre.geom_type!='Polygon': errors.append('corridor_width')
    else:
        residual=net_corridor.difference(centre.buffer(problem.profile.corridor_width/2,join_style=2))
        # A half-wall shoulder where a new wall terminates at existing structure
        # is not a passage. The connected clear-width centre-space and actual
        # door-front rectangles must still pass independently.
        residual=residual.difference(problem.fixed_union.buffer(problem.profile.wall_thickness/2+1e-7,join_style=2))
        # Small three-sided recesses trapped by EXISTING structure are floor
        # area, not passages. Do not enlarge the whole lobby to give such a
        # recess a corridor-width turning circle. Door-front checks still apply.
        pieces=[residual] if residual.geom_type=='Polygon' else list(getattr(residual,'geoms',[]))
        for part in pieces:
            if part.is_empty: continue
            structural_contact=part.boundary.intersection(
                problem.fixed_union.buffer(problem.profile.wall_thickness/2+2e-7,join_style=2)).length
            if part.area <= (problem.profile.corridor_width/2)**2 and structural_contact >= .6*part.length:
                recess_area += part.area
        if result.opening.difference(centre.buffer(problem.profile.corridor_width/2+1e-5,join_style=2)).length>1e-6:
            errors.append('core_opening_outside_clear_passage')
        if residual.area-recess_area>1e-5: errors.append('corridor_thin_appendage')
    for uid,poly in units.items():
        if poly.geom_type!='Polygon' or not poly.is_valid or poly.is_empty:
            errors.append(f'{uid}:disconnected'); continue
        if poly.intersection(problem.fixed_union).area>1e-6: errors.append(f'{uid}:structure_overlap')
        if poly.difference(result.allocatable_space).area>1e-6: errors.append(f'{uid}:outside_free_space')
        net=physical.units[uid]
        if net.is_empty or net.geom_type!='Polygon':
            errors.append(f'{uid}:wall_thickness_disconnects_unit'); continue
        length=net.boundary.intersection(exterior).length
        required=frontage_requirement(net.area,q)
        error=abs(net.area-result.target_areas[uid])/result.target_areas[uid]
        count=corners(poly,problem)
        coords=list(poly.simplify(1e-7,preserve_topology=True).exterior.coords)
        short_edges=sum(1 for a,b in zip(coords,coords[1:])
                        if LineString([a,b]).length<q['min_unit_width']-1e-6
                        and LineString([a,b]).distance(problem.fixed_union)>1e-6)
        if length+1e-6<required: errors.append(f'{uid}:facade_deficit')
        if error>q['area_tolerance']+1e-6: errors.append(f'{uid}:area')
        if count>q['max_corners']: errors.append(f'{uid}:too_many_corners')
        eroded=net.buffer(-q['min_unit_width']/2+1e-5,join_style=2)
        if eroded.is_empty or eroded.geom_type!='Polygon': errors.append(f'{uid}:narrow_or_disconnected')
        unusable=net.difference(eroded.buffer(q['min_unit_width']/2,join_style=2)).area/net.area
        if unusable>q['max_unusable_ratio']+1e-6: errors.append(f'{uid}:thin_appendage')
        # Geometric opportunity only: no sun, glazing or obstruction simulation.
        frontage=net.boundary.intersection(exterior)
        lit=net.intersection(frontage.buffer(q['daylight_depth'])).area/net.area
        if lit+1e-6<q['min_daylight_coverage']: errors.append(f'{uid}:daylight_depth_deficit')
        door=doors.get(uid)
        if door:
            line=LineString(door.points)
            if abs(line.length-problem.profile.door_width)>1e-6: errors.append(f'{uid}:door_width')
            if line.difference(poly.boundary.buffer(1e-7)).length>1e-6 or line.difference(corridor.boundary.buffer(1e-7)).length>1e-6:
                errors.append(f'{uid}:door_contact')
            if not door_clearance(door,net,net_corridor,q): errors.append(f'{uid}:door_clearance')
            passage=line.buffer(problem.profile.wall_thickness/2,cap_style=2)
            if passage.intersection(problem.fixed_union).area>1e-6: errors.append(f'{uid}:door_structure_overlap')
        metrics[uid]=dict(area=net.area,territory_area=poly.area,wall_share_area=poly.area-net.area,
                          target_area=result.target_areas[uid],area_error_ratio=error,
                          facade_length=length,required_facade_length=required,facade_per_area=length/net.area,
                          corners=count,short_edges=short_edges,daylight_coverage_proxy=lit,unusable_area_ratio=unusable)
    polygons=list(units.values()); merged=unary_union(polygons)
    if sum(p.area for p in polygons)-merged.area>1e-6: errors.append('unit_overlap')
    missing=merged.symmetric_difference(result.allocatable_space).area
    if missing>1e-6: errors.append('coverage')
    for i,a in enumerate(result.doors):
        for b in result.doors[i+1:]:
            if LineString(a.points).distance(LineString(b.points))+1e-6<problem.profile.min_door_spacing:
                errors.append('door_spacing')
    # Only shared unit boundaries are checked: external outlines may be irregular.
    axes=structural_axes(problem); center_axes=structural_center_axes(problem)
    reference_axes=structural_reference_axes(problem)
    reference_lengths={k:0. for k in reference_axes}
    aligned=0.; shared_total=0.; center_aligned=0.; reference_supported=0.
    for i,a in enumerate(polygons):
        for b in polygons[i+1:]:
            shared=a.boundary.intersection(b.boundary)
            lines=[shared] if shared.geom_type=='LineString' else list(getattr(shared,'geoms',[]))
            for line in lines:
                if line.geom_type!='LineString': continue
                for p,r in zip(line.coords,list(line.coords)[1:]):
                    seg=LineString([p,r]); shared_total+=seg.length
                    supported=False
                    for mode,refs in reference_axes.items():
                        if any(abs(p[d]-r[d])<1e-7 and any(abs(p[d]-v)<1e-6 for v in refs[d]) for d in (0,1)):
                            reference_lengths[mode]+=seg.length
                            supported=True
                    if supported: reference_supported+=seg.length
                    if any(abs(p[d]-r[d])<1e-7 and any(abs(p[d]-v)<1e-6 for v in center_axes[d]) for d in (0,1)):
                        center_aligned+=seg.length
                    ok=any(abs(p[d]-r[d])<1e-7 and any(abs(p[d]-v)<=q['alignment_tolerance'] for v in axes[d]) for d in (0,1))
                    if ok: aligned+=seg.length
                    else:
                        # Supplementary modular lines allow useful concavity, while
                        # a separate ratio retains structural preference.
                        modular=any(abs(p[d]-r[d])<1e-7 and
                            abs((p[d]-problem.boundary.bounds[d])/problem.profile.grid_size-
                                round((p[d]-problem.boundary.bounds[d])/problem.profile.grid_size))<1e-6
                            for d in (0,1))
                        if not modular: errors.append('partition_edge_off_structure_grid')
    alignment=aligned/shared_total if shared_total else 1.
    reference_mode=max(reference_lengths,key=reference_lengths.get)
    reference_ratio=reference_lengths[reference_mode]/shared_total if shared_total else 1.
    if alignment+1e-6<q['min_structure_alignment']: errors.append('structure_alignment_deficit')
    net_area=sum(p.area for p in physical.units.values())+net_corridor.area
    accounting_error=abs(net_area+physical.solid.area+physical.thresholds.area-
                         problem.boundary.difference(problem.fixed_union).area)
    if accounting_error>1e-6: errors.append('physical_area_accounting')
    return dict(valid=not errors,errors=sorted(set(errors)),units=metrics,
                unassigned_area=missing,corridor_area=net_corridor.area,corridor_territory_area=corridor.area,
                corridor_structural_recess_area=recess_area,
                wall_thickness=problem.profile.wall_thickness,wall_reservation_area=physical.reservation.area,
                solid_wall_area=physical.solid.area,door_threshold_area=physical.thresholds.area,
                physical_area_accounting_error=accounting_error,
                net_floor_area=net_area,
                area_balance_deadband=q['area_balance_deadband'],
                center_axis_alignment_ratio=center_aligned/shared_total if shared_total else 1.,
                consistent_reference_alignment_ratio=reference_ratio,
                dominant_structure_reference=reference_mode if reference_supported else 'none',
                reference_consistency_ratio=reference_lengths[reference_mode]/reference_supported if reference_supported else 1.,
                structure_reference_lengths=reference_lengths,
                structure_alignment_ratio=alignment,
                daylight_method='per_unit_exposed_facade_and_depth_coverage_proxy')
