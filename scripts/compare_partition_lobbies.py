"""Explicit A1 sensitivity example: existing spacing vs a smaller spacing.

This writes experiment-local configurations only, never changes A1's settings.
"""
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import copy
import json
from datetime import datetime
import yaml
from core.floor_partition.contracts import build_floor_partition_problem
from core.floor_partition.structured import candidate_partitions
from core.floor_partition.lobby import compact_starts
from core.floor_partition.export import export_unit_configs
from core.floor_partition.rendering import render_partition


def main():
    root=Path(__file__).resolve().parents[1]
    config=yaml.safe_load((root/'config/a1/config.yaml').read_text(encoding='utf8'))
    p=build_floor_partition_problem(config)
    pool=candidate_partitions(p)
    seeds=sorted(pool,key=lambda item:item[1]['corridor_area'])[:2]
    experiment=root/'results2/floor_partition'/('a1_lobby_comparison_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    experiment.mkdir(parents=True,exist_ok=True)
    summary=[]
    for spacing in sorted({p.profile.min_door_spacing,1.2,2.4}):
        c=copy.deepcopy(config)
        c['FloorPartition']['residential']['min_door_spacing']=spacing
        pp=build_floor_partition_problem(c)
        candidates=list(compact_starts(pp,seeds))
        if not candidates:
            summary.append(dict(min_door_spacing=spacing,status='not_found_within_candidate_budget',valid=False))
            continue
        r,q=min(candidates,key=lambda item:item[1]['corridor_area'])
        out=experiment/f'spacing_{spacing:g}'
        out.mkdir(parents=True,exist_ok=True)
        (out/'config.yaml').write_text(yaml.safe_dump(c,allow_unicode=True,sort_keys=False),encoding='utf8')
        (out/'validation.json').write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf8')
        export_unit_configs(c,r,out/'exports')
        render_partition(c,r,out)
        summary.append(dict(min_door_spacing=spacing,corridor_net_area=q['corridor_area'],
                            walls=q['solid_wall_area'],valid=q['valid'],output=str(out)))
    (experiment/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
