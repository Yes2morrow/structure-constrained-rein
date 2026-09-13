"""Reproducible retrofit decoder audit, without claiming a trained RL policy."""
import copy
import json
import time
from pathlib import Path
import yaml
from shapely.geometry import mapping
from core.seed_growth.contracts import build_problem
from core.seed_growth.environment import SeedLayoutEnv
from core.seed_growth.geometry import decode_layout, _grow
from core.seed_growth.validation import validate_layout
from core.seed_growth.rendering import render_result


def main():
    root=Path(__file__).resolve().parents[2]
    config=yaml.safe_load((root/'config/retrofit/config.yaml').read_text(encoding='utf-8'))
    legacy=copy.deepcopy(config)
    for room in legacy['TargetSpaces']:
        if room.pop('role',None)=='residual': room.pop('min_width',None)
    old=build_problem(legacy)
    old_seeds={r.id:r.seed for r in old.active_rooms}
    trials=[]
    start=time.perf_counter()
    for phase in range(4):
        polygons,_=_grow(old,old_seeds,phase,old.grid_size,2000)
        report=validate_layout(old,old_seeds,polygons)
        trials.append((len(report['errors']),sum(abs(polygons[r.id].area-r.target_area) for r in old.rooms),report))
    previous=min(trials,key=lambda x:x[:2])[2]
    baseline=dict(seconds=time.perf_counter()-start,validation=previous,method='independent_growth_four_phases')
    env=SeedLayoutEnv(config)
    start=time.perf_counter(); result=decode_layout(env.problem,env.snapshot(0))
    duration=time.perf_counter()-start
    result['polygons']={k:mapping(p) for k,p in result['polygons'].items()}
    out=root/'output/graph_partition_review_20260913'; out.mkdir(parents=True,exist_ok=True)
    (out/'config.yaml').write_text(yaml.safe_dump(config,allow_unicode=True),encoding='utf-8')
    render_result(config,result,out)
    summary=dict(agent_ids=list(env.ids),graph_node_ids=list(env.proxy.ids),
        observation=list(env.observation_space.shape),seconds=duration,baseline=baseline,
        validation=result['validation'],status=result['status'],note='Initial configured seeds; not a trained RL comparison')
    (out/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
