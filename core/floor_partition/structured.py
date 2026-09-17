"""Bounded beam search over structural straight cuts, with exact acceptance.

No staircase raster growth, disconnected-cell fallback, or largest-part loss.
Concave polygons and column holes are retained exactly.
"""
from dataclasses import replace
from shapely.geometry import box, LineString
from shapely.ops import linemerge, unary_union
from .contracts import target_areas_for_area, Door, PartitionResult
from .circulation import circulation_candidates
from .walls import wall_geometry, update_net_targets
from .quality import (settings, structural_axes, facade,
                      corners, door_clearance, validate_partition)


def lines(geometry):
    if geometry.geom_type=='LineString': return [geometry]
    return [p for g in getattr(geometry,'geoms',[]) for p in lines(g)]


def unit_doors(problem, circulation, polygons, locked=(), allowed_window=None):
    q=settings(problem); width=problem.profile.door_width; choices={}
    q['wall_thickness']=problem.profile.wall_thickness
    physical=wall_geometry(problem,circulation.corridor,polygons)
    for uid,poly in polygons.items():
        existing=next((d for d in locked if d.unit_id==uid),None)
        if existing is not None:
            choices[uid]=[existing]
            continue
        candidates=[]
        contact=poly.boundary.intersection(circulation.corridor.boundary)
        contact=unary_union(lines(contact))
        if contact.geom_type=='MultiLineString': contact=linemerge(contact)
        for line in lines(contact):
            # GEOS may retain collinear nodes; simplify before sampling segments.
            line=line.simplify(1e-7)
            for a,b in zip(line.coords,list(line.coords)[1:]):
                edge=LineString([a,b]); length=edge.length
                margin=max(width,q['door_clearance_width'])/2
                if length<2*margin-1e-6: continue
                positions=[length/2]
                count=int((length-2*margin)/.25)+1
                positions += [margin+j*.25 for j in range(count)]
                for position in positions:
                    p=edge.interpolate(position-width/2).coords[0]
                    r=edge.interpolate(position+width/2).coords[0]
                    label='horizontal' if abs(p[1]-r[1])<1e-7 else 'vertical'
                    d=Door(uid,(p,r),label,position)
                    if allowed_window is not None and not allowed_window.buffer(1e-7).covers(LineString(d.points)):
                        continue
                    if door_clearance(d,physical.units[uid],physical.corridor,q): candidates.append(d)
        # Keep geographically spread candidates, rather than only one edge.
        unique={tuple(round(v,6) for p in d.points for v in p):d for d in candidates}
        candidates=list(unique.values())
        if not candidates: raise ValueError(f'{uid} 无足宽户门及完整门前/门内净空')
        if len(candidates)>24:
            candidates=[candidates[round(i*(len(candidates)-1)/23)] for i in range(24)]
        choices[uid]=candidates
    ordered=sorted(choices,key=lambda u:len(choices[u])); selected=[]
    attempts=0
    def choose(index):
        nonlocal attempts
        attempts+=1
        if attempts>5000: return False
        if index==len(ordered): return True
        for door in choices[ordered[index]]:
            if any(LineString(door.points).distance(LineString(d.points))+1e-7<problem.profile.min_door_spacing for d in selected): continue
            selected.append(door)
            if choose(index+1): return True
            selected.pop()
        return False
    if not choose(0):
        if attempts>5000: raise ValueError('door_search_budget_exhausted')
        raise ValueError('当前门位候选中未找到满足最小净间距的组合')
    return tuple(sorted(selected,key=lambda d:d.unit_id))


from .scoring import partition_score as score_report


def candidate_partitions(problem):
    q=settings(problem); axes=structural_axes(problem,include_wall_faces=False); exterior=facade(problem)
    accepted=[]; seen=set(); diagnostics=[]; repair_seeds=[]
    for circulation in circulation_candidates(problem):
        free=problem.boundary.difference(problem.fixed_union).difference(circulation.corridor)
        if free.geom_type!='Polygon':
            diagnostics.append('可分配空间不连通'); continue
        targets,scale=target_areas_for_area(problem,free.area)
        if exterior.length+1e-6<max(len(targets)*q['min_facade_length'],free.area*q['facade_per_area']):
            diagnostics.append('整层可用采光外墙总长不足'); continue
        # A state contains (geometry, contiguous target-index group) leaves.
        beam=[[(free,tuple(range(len(targets))))]]
        def rank(state):
            return sum(abs(p.area-sum(targets[i] for i in ids))/sum(targets[i] for i in ids)
                       +.004*corners(p,problem) for p,ids in state)
        for _ in range(len(targets)-1):
            next_states=[]; unique=set()
            for state in beam:
                unresolved=[i for i,(_,ids) in enumerate(state) if len(ids)>1]
                if not unresolved: next_states.append(state); continue
                at=max(unresolved,key=lambda i:len(state[i][1]))
                poly,ids=state[at]; bx,by,ex,ey=poly.bounds
                for d in (0,1):
                    for cut in axes[d]:
                        if not poly.bounds[d]+1e-6<cut<poly.bounds[d+2]-1e-6: continue
                        mask=box(bx-1,by-1,cut,ey+1) if d==0 else box(bx-1,by-1,ex+1,cut)
                        a=poly.intersection(mask).buffer(0).simplify(1e-7,preserve_topology=True)
                        b=poly.difference(mask).buffer(0).simplify(1e-7,preserve_topology=True)
                        if any(p.is_empty or p.geom_type!='Polygon' for p in (a,b)): continue
                        for n in range(1,len(ids)):
                            groups=[(a,ids[:n]),(b,ids[n:])]; valid=True
                            for p,group in groups:
                                target=sum(targets[i] for i in group)
                                if abs(p.area-target)/target>q['area_tolerance']: valid=False; break
                                # The area-dependent frontage requirement uses NET
                                # area after walls. Gross-area pruning here can
                                # discard a valid completed partition prematurely.
                                if p.boundary.intersection(exterior).length+1e-6 < q['min_facade_length']*len(group): valid=False; break
                                if p.boundary.intersection(circulation.corridor.boundary).length+1e-6 < problem.profile.door_width*len(group): valid=False; break
                                inset=p.buffer(-q['min_unit_width']/2+1e-5,join_style=2)
                                if inset.is_empty or inset.geom_type!='Polygon': valid=False; break
                                if len(group)==1 and corners(p,problem)>q['max_corners']: valid=False; break
                            if not valid: continue
                            new=state[:at]+groups+state[at+1:]
                            key=tuple((p.wkb,g) for p,g in new)
                            if key not in unique: next_states.append(new); unique.add(key)
            beam=sorted(next_states,key=rank)[:int(q['beam_width'])]
            if not beam: break
        for state in beam:
            if any(len(ids)!=1 for _,ids in state): continue
            units={problem.targets[ids[0]].unit_id:p for p,ids in state}
            result=PartitionResult(circulation.corridor,circulation.opening,circulation.opening_side,
                free,(),units,{t.unit_id:targets[i] for i,t in enumerate(problem.targets)},
                {t.unit_id:t.requested_area for t in problem.targets},scale)
            result=update_net_targets(problem,result)
            if state is beam[0]:
                repair_seeds.append((result,{'corridor_area':circulation.corridor.area}))
            try: doors=unit_doors(problem,circulation,units)
            except ValueError as exc: diagnostics.append(str(exc)); continue
            result=replace(result,doors=doors)
            report=validate_partition(problem,result)
            if not report['valid']:
                diagnostics.extend(report['errors']); continue
            key=tuple(p.wkb for _,p in sorted(units.items()))
            if key not in seen: accepted.append((result,report)); seen.add(key)
    if accepted or repair_seeds:
        from .lobby import compact_starts
        # Short bands avoid inheriting gratuitous branches/column bypasses when
        # testing an alternative compact topology.
        seeds=sorted(accepted,key=lambda pair:(pair[1]['corridor_area'],-score_report(pair[1])))[:2]
        # Door placement must not be a prerequisite for trying another
        # circulation topology. Provisional starts are NEVER returned as valid.
        if not seeds: seeds=repair_seeds[-4:]
        accepted.extend(compact_starts(problem,seeds))
    if not accepted:
        raise ValueError('结构分户搜索未找到通过面积、采光外墙、连通性和门前净空验收的方案。'
                         '请调整目标面积/户数/约束或扩大beam_width；这不代表全局无解。 '+ '; '.join(sorted(set(diagnostics))[:4]))
    return sorted(accepted,key=lambda pair:score_report(pair[1]),reverse=True)[:int(q['max_candidates'])]
