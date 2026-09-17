"""Sequential unit handoff and validated assembly; no image stitching."""
from copy import deepcopy
from pathlib import Path
import json
import re

import yaml
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from core.envs.structure_geometry import fixed_polygon
from .snapshot import snapshot_id


def read_floor_snapshot(path):
    data=json.loads(Path(path).read_text(encoding='utf8'))
    if data.get('schema')!='floor-handoff-v1' or data.get('coordinate_system')!='original_floor_metres':
        raise ValueError('该结果缺少统一坐标的楼层交接文件，请用新版第一阶段重新导出。')
    if data.get('plan_id') != snapshot_id(data):
        raise ValueError('楼层交接文件内容与方案编号不一致')
    return data


def prepare_units(exports, config_root, workflow_root):
    """Create distinct configs once; reopening never overwrites room edits."""
    exports=Path(exports).resolve(); config_root=Path(config_root); workflow_root=Path(workflow_root)
    floor=read_floor_snapshot(exports/'floor_plan.json')
    batch=floor['plan_id']; directory=workflow_root/batch
    manifest_path=directory/'workflow.json'
    if manifest_path.exists():
        return manifest_path
    prepared=[]
    for i,uid in enumerate(floor['units'],1):
        if not uid or uid in ('.','..') or any(c in uid for c in '/\\:'):
            raise ValueError('户型编号无效')
        c=yaml.safe_load((exports/uid/'config.yaml').read_text(encoding='utf8'))
        if c.get('FloorPartitionResult',{}).get('floor_plan_id')!=batch:
            raise ValueError(f'{uid} 不属于当前楼层方案')
        parent=re.sub('[^a-z0-9_-]','_',str(floor.get('source_config_id') or 'floor').lower())
        identifier=f'{parent}_{batch[:12]}_u{i:02d}'
        dest=config_root/identifier/'config.yaml'
        if dest.exists():
            existing=yaml.safe_load(dest.read_text(encoding='utf8')) or {}
            if existing.get('FloorWorkflow',{}).get('plan_id')!=batch:
                raise ValueError(f'配置编号冲突：{identifier}')
        c=deepcopy(c)
        c['ConfigID']=identifier
        c['FloorWorkflow']=dict(plan_id=batch,unit_id=uid,workflow_path=str(manifest_path.resolve()))
        c.setdefault('Training',{}).update(training_stage='room_training',ckpt_path='',resume_mode='fresh')
        # This route produces exact polygon artifacts consumed by assembly.
        c.setdefault('SeedGrowth',{}).update(enabled=True,resume_run='')
        prepared.append((uid,identifier,dest,c))
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'floor_plan.json').write_text(json.dumps(floor,ensure_ascii=False,indent=2),encoding='utf8')
    units={}
    for uid,identifier,dest,c in prepared:
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            dest.write_text(yaml.safe_dump(c,allow_unicode=True,sort_keys=False),encoding='utf8')
        units[uid]=dict(config_id=identifier,config_path=str(dest.resolve()))
    manifest=dict(schema='floor-workflow-v1',plan_id=batch,source_exports=str(exports),units=units)
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    return manifest_path


def matching_runs(results_root, plan_id, unit_id):
    runs=[]
    for path in Path(results_root).glob('seed_experimental/*/config.yaml'):
        try:
            c=yaml.safe_load(path.read_text(encoding='utf8')) or {}
            ref=c.get('FloorWorkflow',{})
            if ref.get('plan_id')==plan_id and ref.get('unit_id')==unit_id:
                runs.append(path.parent)
        except (OSError,ValueError,yaml.YAMLError):
            continue
    return sorted(runs,key=lambda p:p.stat().st_mtime,reverse=True)


def read_unit_result(floor, uid, run, current_config):
    """Reject stale, foreign, partial or invalid room geometry independently."""
    from core.seed_growth.artifacts import config_key
    from core.seed_growth.contracts import build_problem
    from core.seed_growth.validation import validate_layout
    run=Path(run).resolve()
    c=yaml.safe_load((run/'config.yaml').read_text(encoding='utf8'))
    ref=c.get('FloorWorkflow',{})
    if ref.get('plan_id')!=floor['plan_id'] or ref.get('unit_id')!=uid:
        raise ValueError('户型结果来自其他楼层方案或其他户型')
    # Training controls may change for a future run; spatial requirements may not.
    for key in ('ExistingBuilding','TargetSpaces','FunctionalRelations','SeedGrowth'):
        old=deepcopy(c.get(key,{})); new=deepcopy(current_config.get(key,{}))
        if key=='SeedGrowth':
            for data in (old,new): data.pop('resume_run',None)
        if old!=new:
            raise ValueError('房间需求或边界已修改，该训练结果已过期')
    pointer=json.loads((run/'last_valid.json').read_text(encoding='utf8'))
    path=(run/pointer['path']).resolve()
    if run not in path.parents: raise ValueError('结果路径超出训练目录')
    result=json.loads(path.read_text(encoding='utf8'))
    if result.get('status')!='valid' or result.get('config_key')!=config_key(c):
        raise ValueError('训练结果尚未通过验收或配置不匹配')
    problem=build_problem(c)
    expected=shape(floor['units'][uid]['net_boundary'])
    if problem.free_space.symmetric_difference(expected).area>1e-5:
        raise ValueError('户内训练边界与第一阶段净边界不一致')
    polygons={key:shape(value) for key,value in result['polygons'].items()}
    report=validate_layout(problem,result['snapshot']['seeds'],polygons)
    if not report['geometry_valid'] or not report['relations_satisfied']:
        raise ValueError('房间结果未通过重新验收')
    if any(p.difference(expected).area>1e-5 for p in polygons.values()):
        raise ValueError('房间越出所属户型的净边界')
    merged=unary_union(list(polygons.values()))
    if expected.symmetric_difference(merged).area>1e-5:
        raise ValueError('房间划分尚未覆盖全部户型净面积')
    return polygons,dict(run=str(run),snapshot_key=result['snapshot_key'])


def composition(floor, unit_rooms):
    if set(unit_rooms)-set(floor['units']): raise ValueError('存在未知户型')
    features=[]; all_rooms=[]
    for uid,rooms in unit_rooms.items():
        boundary=shape(floor['units'][uid]['net_boundary'])
        for rid,p in rooms.items():
            if p.is_empty or not p.is_valid or p.difference(boundary).area>1e-5:
                raise ValueError(f'{uid}/{rid} 超出户型净边界或几何无效')
            all_rooms.append(p)
            features.append(dict(type='Feature',properties=dict(unit_id=uid,room_id=rid),geometry=mapping(p)))
        if unary_union(list(rooms.values())).symmetric_difference(boundary).area>1e-5:
            raise ValueError(f'{uid} 未覆盖全部净面积')
    if all_rooms and sum(p.area for p in all_rooms)-unary_union(all_rooms).area>1e-5:
        raise ValueError('合成房间存在重叠')
    missing=sorted(set(floor['units'])-set(unit_rooms))
    return dict(type='FeatureCollection',features=features,plan_id=floor['plan_id'],
                coordinate_system='original_floor_metres',complete=not missing,missing_units=missing)


def composition_figure(floor, unit_rooms):
    from matplotlib.figure import Figure
    from .rendering import _draw_geometry
    from common.plan_styles import PARTITION_FILL,STRUCTURE_FILL,CORE_FILL,CORE_EDGE,CORE_HATCH,DOOR_COLOR
    composition(floor,unit_rooms)
    fig=Figure(figsize=(12,8)); ax=fig.subplots()
    _draw_geometry(ax,shape(floor['floor_outline']),'white','#172126')
    colors=['#90caf9','#a5d6a7','#ffcc80','#ce93d8','#80cbc4','#ef9a9a']
    for uid,item in floor['units'].items():
        net=shape(item['net_boundary'])
        _draw_geometry(ax,net,'#eeeeee','#777777')
        if uid not in unit_rooms:
            p=net.representative_point(); ax.text(p.x,p.y,f'{uid}\nPending',ha='center',fontsize=8)
        for i,(rid,p) in enumerate(unit_rooms.get(uid,{}).items()):
            _draw_geometry(ax,p,colors[i%len(colors)],'#555555',linewidth=.6)
            point=p.representative_point(); ax.text(point.x,point.y,f'{uid}\n{rid}',ha='center',fontsize=6)
    _draw_geometry(ax,shape(floor['corridor']),'#20a4ed','#0571ad')
    _draw_geometry(ax,shape(floor['partition_walls']),PARTITION_FILL,PARTITION_FILL)
    for item in floor['fixed_objects']:
        core=item['type'] in ('core','traffic_core')
        _draw_geometry(ax,fixed_polygon(item),CORE_FILL if core else STRUCTURE_FILL,CORE_EDGE,
                       hatch=CORE_HATCH if core else None)
    for door in floor['doors']:
        a,b=door['points']; ax.plot([a[0],b[0]],[a[1],b[1]],color=DOOR_COLOR,lw=4)
    ax.set_aspect('equal'); ax.autoscale_view()
    ax.set_title('Whole floor | '+('Complete' if len(unit_rooms)==len(floor['units']) else 'Draft: pending units'))
    fig.tight_layout()
    return fig
