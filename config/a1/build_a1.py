"""Validate and render A1 from config.yaml without overwriting user settings."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from core.envs.structure_geometry import fixed_polygon
from core.floor_partition import run_residential_floor_partition
from core.floor_partition.quality import validate_partition
from core.floor_partition.rendering import render_partition, _draw_geometry
from common.plan_styles import STRUCTURE_FILL, CORE_FILL, CORE_EDGE, CORE_HATCH


def main():
    config = yaml.safe_load((HERE/'config.yaml').read_text(encoding='utf8'))
    problem, result = run_residential_floor_partition(config)
    report = validate_partition(problem, result)
    if not report['valid']:
        raise ValueError(report['errors'])
    (HERE/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    render_partition(config, result, HERE)
    fig, ax = plt.subplots(figsize=(12,7))
    _draw_geometry(ax,problem.boundary,'#f4f1e9','#233747',alpha=1.)
    _draw_geometry(ax,result.corridor,'#f4cc74','#b28123')
    for item in problem.fixed_objects:
        core=item['type'] in ('traffic_core','core')
        _draw_geometry(ax,fixed_polygon(item),CORE_FILL if core else STRUCTURE_FILL,
                       CORE_EDGE,linewidth=.7,alpha=1.,hatch=CORE_HATCH if core else None)
    grid=config['ModularDesign']
    for x in grid['column_axes_x']: ax.axvline(x,color='#99a',ls='--',alpha=.35,lw=.7)
    for y in grid['column_axes_y']: ax.axhline(y,color='#99a',ls='--',alpha=.35,lw=.7)
    bx,by,ex,ey=problem.boundary.bounds
    ax.set(xlim=(bx-1,ex+1),ylim=(by-1,ey+1),xlabel='x (m)',ylabel='y (m)',
           title='A1 | Photo-checked symmetric abstraction | 18 columns: 0.6 x 0.8 m\n'
                 'Perimeter columns flush with outline | Yellow: generated circulation')
    ax.set_aspect('equal'); fig.tight_layout()
    fig.savefig(HERE/'A1_environment.png',dpi=160)
    fig.savefig(HERE/'A1_environment.svg'); plt.close(fig)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
