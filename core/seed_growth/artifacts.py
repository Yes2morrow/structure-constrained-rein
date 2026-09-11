"""Atomic offline layout artifacts, explicit failures and same-snapshot reuse."""
import hashlib
import json
import os
import time
from pathlib import Path
from uuid import uuid4
from html import escape

import yaml
from shapely.geometry import mapping
from .contracts import SeedSnapshot, PreciseSchedule
from .geometry import decode_layout
from .proxy import LayoutProxy
from .validation import validate_layout
from .shape_rules import polygon_parts


def config_key(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,allow_nan=False).encode()).hexdigest()


def atomic_text(path, text):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    try:
        with temp.open('w',encoding='utf-8') as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
        for attempt in range(6):
            try:
                os.replace(temp,path)
                break
            except PermissionError:
                if os.name!='nt' or attempt==5: raise
                # Windows indexers/readers can briefly deny atomic replacement.
                # Never fall back to truncating the committed destination.
                time.sleep(.02*(2**attempt))
    finally:
        if temp.exists(): temp.unlink()


def atomic_json(path, value):
    atomic_text(path,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def read_snapshot(problem, data):
    if data.get('schema_version') != 1:
        raise ValueError('不支持的种子快照版本')
    return SeedSnapshot.capture(problem,data['seeds'],data['completed_episodes'],data['step'])


def compare_proxy(problem, estimate, report):
    areas=[]
    for i,room in enumerate(problem.rooms):
        precise=report['rooms'].get(room.id,{}).get('area')
        value=float(estimate['areas'][i])
        areas.append(dict(id=room.id,estimated_area=value,precise_area=precise,
                          error=None if precise is None else value-precise))
    relations=[]
    for estimated,actual in zip(estimate['relations'],report['relations']):
        value=estimated['estimated']
        predicted=None if value is None else bool(value>=1-1e-9)
        proved=actual['satisfied']
        relations.append(dict(source=actual['source'],target=actual['target'],kind=actual['kind'],
            estimated_score=None if value is None else float(value),estimated_satisfied=predicted,
            precise_satisfied=proved,shared_length=actual['shared_length'],distance=actual['distance'],
            false_positive=predicted is True and proved is False,
            false_negative=predicted is False and proved is True))
    return dict(areas=areas,relations=relations,
                false_positives=sum(r['false_positive'] for r in relations),
                false_negatives=sum(r['false_negative'] for r in relations),
                unknown=sum(r['precise_satisfied'] is None for r in relations),
                note='代理接触及距离奖励均不是精确邻接或通行证明')


def layout_svg(problem, snapshot, polygons, status):
    x0,y0,x1,y1=problem.boundary.bounds
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0-1} {-y1-2} {x1-x0+2} {y1-y0+3}">']
    def draw(p,color):
        for poly in polygon_parts(p):
            commands=[]
            for ring in [poly.exterior,*poly.interiors]:
                coords=list(ring.coords)
                commands.append('M '+' L '.join(f'{x},{-y}' for x,y in coords)+' Z')
            parts.append(f'<path d="{" ".join(commands)}" fill="{color}" fill-rule="evenodd" stroke="#263238" stroke-width="0.035"/>')
    draw(problem.boundary,'#fafafa')
    palette=['#90caf9','#a5d6a7','#ffcc80','#ce93d8','#80cbc4','#ef9a9a']
    for i,(key,p) in enumerate(polygons.items()):
        draw(p,palette[i%len(palette)])
        x,y=dict(snapshot.seeds)[key]
        parts.append(f'<circle cx="{x}" cy="{-y}" r="0.08" fill="#111"/><text x="{x+.1}" y="{-y}" font-size="0.25">{escape(key)}</text>')
    draw(problem.fixed,'#455a64')
    parts.append(f'<text x="{x0}" y="{-y1-.7}" font-size="0.35">Precise: {status}; episodes={snapshot.completed_episodes}, step={snapshot.step}</text></svg>')
    return '\n'.join(parts)


class RunArtifacts:
    def __init__(self, root, config, problem, proxy=None, decoder=decode_layout, export_root=None):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.config=config; self.problem=problem
        self.export_root=Path(export_root) if export_root is not None else None
        self.fingerprint=config_key(config)
        manifest=self.root/'manifest.json'
        if manifest.exists():
            previous=json.loads(manifest.read_text(encoding='utf-8'))
            if previous['config_key']!=self.fingerprint:
                raise ValueError('运行配置与已保存结果不一致，不能覆盖或混合续作')
        else:
            atomic_text(self.root/'config.yaml',yaml.safe_dump(config,allow_unicode=True))
            atomic_json(manifest,dict(schema_version=1,config_key=self.fingerprint,mode='seed_proxy_v3'))
        self.proxy=proxy or LayoutProxy(problem)
        self.decoder=decoder
        self.schedule=PreciseSchedule(p.parent.name for p in (self.root/'precise').glob('*/result.json'))

    def save_snapshot(self, snapshot, force=False):
        path=self.root/'latest_snapshot.json'
        if path.exists() and not force:
            previous=json.loads(path.read_text(encoding='utf-8'))
            if (previous['completed_episodes'],previous['step'])>(snapshot.completed_episodes,snapshot.step):
                return
        atomic_json(self.root/'latest_snapshot.json',snapshot.to_dict())

    def _publish(self, snapshot, result):
        pointer=dict(snapshot_key=snapshot.key,path=f'precise/{snapshot.key}/result.json',status=result['status'],
                     completed_episodes=snapshot.completed_episodes,step=snapshot.step)
        for name in ['latest_precise.json']+(['last_valid.json'] if result['status']=='valid' else []):
            path=self.root/name
            if path.exists():
                old=json.loads(path.read_text(encoding='utf-8'))
                if 'completed_episodes' not in old:
                    old=json.loads((self.root/old['path']).read_text(encoding='utf-8'))['snapshot']
                if (old['completed_episodes'],old['step'])>(snapshot.completed_episodes,snapshot.step):
                    continue
            atomic_json(path,pointer)
        self.schedule.mark_completed(snapshot)
        atomic_json(self.root/'schedule.json',self.schedule.to_dict())
        if self.export_root is not None:
            # Rendering failure must not invalidate committed geometry/model data.
            output=self.export_root
            try:
                latest=json.loads((self.root/'latest_precise.json').read_text(encoding='utf-8'))
                rendered=json.loads((output/'result.json').read_text(encoding='utf-8')) if (output/'result.json').exists() else {}
                if latest['snapshot_key']==snapshot.key and (rendered.get('snapshot_key')!=snapshot.key
                        or not (output/'layout.png').exists() or not (output/'layout.svg').exists()):
                    from .rendering import render_result
                    render_result(self.config,result,output)
            except Exception as exc:
                atomic_json(self.root/'render_failure.json',dict(snapshot_key=snapshot.key,error=str(exc)))

    def emit(self, snapshot, reason):
        if not self.schedule.due(snapshot,reason) and snapshot.key not in self.schedule.completed_keys:
            return None
        self.save_snapshot(snapshot)
        folder=self.root/'precise'/snapshot.key
        completed=folder/'result.json'
        # result.json is the commit marker. Recover after a crash between this
        # marker and publishing the pointers without decoding a second time.
        if completed.exists():
            result=json.loads(completed.read_text(encoding='utf-8'))
            if result['config_key']!=self.fingerprint or result['snapshot_key']!=snapshot.key:
                raise ValueError('精确结果与配置/快照不匹配')
            self._publish(snapshot,result)
            return result
        atomic_json(folder/'request.json',dict(config_key=self.fingerprint,snapshot=snapshot.to_dict(),reason=reason))
        try:
            decoded=self.decoder(self.problem,snapshot)
            polygons=decoded.pop('polygons')
            report=validate_layout(self.problem,dict(snapshot.seeds),polygons)
            decoded['validation']=report
            decoded['status']='valid' if report['geometry_valid'] and report['relations_satisfied'] else 'unresolved'
            decoded.update(config_key=self.fingerprint,snapshot_key=snapshot.key,trigger=reason,
                           polygons={k:mapping(v) for k,v in polygons.items()},
                           comparison=compare_proxy(self.problem,self.proxy.evaluate(dict(snapshot.seeds)),report))
            atomic_text(folder/'layout.svg',layout_svg(self.problem,snapshot,polygons,decoded['status']))
            atomic_json(completed,decoded)
            self._publish(snapshot,decoded)
            return decoded
        except Exception as exc:
            failure=dict(status='decode_failed',error_type=type(exc).__name__,error=str(exc),reason=reason,
                         snapshot=snapshot.to_dict(),snapshot_key=snapshot.key)
            atomic_json(folder/'failure.json',failure)
            return failure
