"""Rebuild modular A1; geometry is a design abstraction, not a survey."""
from pathlib import Path
import sys, json
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from core.envs.structure_geometry import column, zone, fixed_polygon
from core.floor_partition import run_residential_floor_partition
from core.floor_partition.quality import validate_partition
from scripts.train_floor_partition import _render_partition

def main():
    config=yaml.safe_load((HERE/'config.yaml').read_text(encoding='utf-8'))
    cores=[('lifts_stairs',[6,12,24,15]),('left_services',[8,9,11,12]),
           ('lift_lobby',[11,10,19,12]),('right_services',[19,9,22,12])]
    objects=[zone('A1_core_'+name,'traffic_core',*r) for name,r in cores]
    for y in [1,5,9]:
        for x in [1,6,11,19,24,29]:
            if y==9 and x in [11,19]: continue
            objects.append(column(f'A1_column_x{x}_y{y}',x,y,.6,.8))
    config['ExistingBuilding']=dict(structure_schema_version=2,
        boundary=[[0,0],[30,0],[30,10],[22,10],[22,12],[24,12],[24,15],
                  [6,15],[6,12],[8,12],[8,10],[0,10]],
        door_positions=[[14.25,10],[15.75,10]],fixed_objects=objects)
    config['FloorPartition'].update(enabled=True,grid_size=.25,
        quality=dict(area_tolerance=.25,min_facade_length=3.,facade_per_area=.12,
                     min_unit_width=1.5,max_corners=24,door_clearance_depth=1.2,
                     door_clearance_width=1.2,entrance_depth=.9,beam_width=32,max_candidates=24),
        rl=dict(enabled=True,episodes=128,learning_rate=.01,entropy_coef=.02,seed=42))
    config['FloorPartition']['residential'].update(unit_count=4,target_area_mode='equal',target_areas=[],
        corridor_width=1.5,door_width=.9,min_door_spacing=2.4,opening_width=1.5,opening_side='south')
    config['Training']['training_stage']='floor_partition'
    config['Training']['episodes']=128
    config['SurveyReference'].update(status='modular_design_abstraction_not_survey',
        assumptions=['已按用户要求模数化、对称化，不再逐像素拟合；像素取点仅保留为来源记录。',
                     '主体30×10m，交通核关于x=15m对称；轮廓主坐标为整米。',
                     '柱统一0.6×0.8m；x轴网1/6/11/19/24/29m，y轴网1/5/9m。',
                     '交通核内的两个上排柱位合并在核固定区域内，不重复设置。',
                     '不设置固定走道；沿实际交通核轮廓自动生成1.5m宽走道。',
                     '逐户有效外墙长度至少max(3m,0.12×净面积)，为几何采光机会指标。'])
    config['ModularDesign']=dict(symmetry_axis_x=15,column_size=[.6,.8],
        column_axes_x=[1,6,11,19,24,29],column_axes_y=[1,5,9],primary_dimension_module=1,
        note='走道、门宽、柱截面保留实用小数；主轮廓及柱中心为整数')
    (HERE/'config.yaml').write_text('# A1 v2: 对称模数化设计环境；单位m，非实测。\n'+
        yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    problem,result=run_residential_floor_partition(config)
    report=validate_partition(problem,result)
    assert report['valid'],report
    (HERE/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    _render_partition(config,result,HERE)
    fig,ax=plt.subplots(figsize=(12,7))
    ax.add_patch(Patch(list(problem.boundary.exterior.coords),facecolor='#f4f1e9',edgecolor='#233747',lw=2))
    ax.add_patch(Patch(list(result.corridor.exterior.coords),facecolor='#f4cc74',edgecolor='#b28123',lw=1.5))
    for item in objects:
        ax.add_patch(Patch(list(fixed_polygon(item).exterior.coords),facecolor='#77a8e5' if item['type']=='traffic_core' else '#303640',edgecolor='white',lw=.6))
    for x in [1,6,11,19,24,29]: ax.axvline(x,color='#99a',ls='--',alpha=.35,lw=.7)
    for y in [1,5,9]: ax.axhline(y,color='#99a',ls='--',alpha=.35,lw=.7)
    ax.text(15,13.5,'TRAFFIC CORE',ha='center',weight='bold')
    ax.set(xlim=(-1,31),ylim=(-1,16),xlabel='x (m)',ylabel='y (m)',
           title='A1 | Modular symmetric environment | columns 0.6 x 0.8 m\nBlue: fixed core; yellow: automatically generated corridor (1.5 m)')
    ax.set_aspect('equal'); fig.tight_layout()
    fig.savefig(HERE/'A1_environment.png',dpi=160); fig.savefig(HERE/'A1_environment.svg'); plt.close(fig)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
