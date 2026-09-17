"""Portable first-stage geometry in the original floor coordinate system."""
import hashlib
import json
from copy import deepcopy
from shapely.geometry import mapping
from .walls import wall_geometry


def snapshot_id(data):
    payload={k:v for k,v in data.items() if k != 'plan_id'}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def floor_snapshot(config, problem, result):
    physical=wall_geometry(problem,result.corridor,result.unit_polygons,result.doors)
    data=dict(schema='floor-handoff-v1', coordinate_system='original_floor_metres',
        source_config_id=config.get('ConfigID',''),
        floor_outline=mapping(problem.boundary),
        fixed_objects=deepcopy(list(problem.fixed_objects)),
        corridor=mapping(physical.corridor), partition_walls=mapping(physical.solid),
        wall_thickness=problem.profile.wall_thickness,
        doors=[dict(unit_id=d.unit_id,points=d.points) for d in result.doors],
        units={uid:dict(net_boundary=mapping(p),territory=mapping(result.unit_polygons[uid]))
               for uid,p in physical.units.items()})
    data['plan_id']=snapshot_id(data)
    return data
