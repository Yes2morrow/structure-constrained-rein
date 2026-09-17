"""Rebuild A1's measured environment and preview; run from any directory."""
from pathlib import Path
import json
import sys

import yaml
from PIL import Image
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch, Patch as LegendPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from core.envs.structure_geometry import column, zone, fixed_polygon, normalize_structures
from core.floor_partition.contracts import build_floor_partition_problem

# Pixel coordinates refer to the supplied 1000 x 750 images. Orthogonalized
# tracing: the photographed page is slightly skewed; no survey accuracy implied.
SCALE = 1.5 / (550 - 512)
def point(x, y):
    return [round((x - 170) * SCALE, 6), round((559 - y) * SCALE, 6)]

def rect(x1, y1, x2, y2):
    return point(x1, y2) + point(x2, y1)

outline = [(170,559),(894,559),(894,308),(743,308),(743,288),
           (701,288),(701,250),(735,250),(735,180),(322,180),
           (322,250),(354,250),(354,315),(170,315)]
core_boxes = [
    ('A1_core_lifts_stairs', (322,180,735,250)),
    ('A1_core_left_services', (354,250,432,315)),
    ('A1_core_lift_lobby', (432,250,626,298)),
    ('A1_core_right_services_upper', (626,250,701,288)),
    ('A1_core_right_services_lower', (626,288,743,334)),
]
column_boxes = [
    (170,315,184,334),(174,431,188,451),(175,539,190,559),
    (300,315,316,339),(301,428,319,451),(303,535,319,559),
    (425,315,440,343),(425,428,441,451),(426,535,443,559),
    (624,425,642,450),(625,532,643,558),
    (748,425,765,449),(748,532,766,558),
    (879,308,894,325),(879,424,894,444),(879,535,894,556),
]
fixed = [zone(name, 'traffic_core', *rect(*pixels)) for name, pixels in core_boxes]
fixed.append(zone('A1_retained_corridor', 'retained_circulation', *rect(512,298,550,476)))
for i, pixels in enumerate(column_boxes, 1):
    x1,y1,x2,y2 = rect(*pixels)
    fixed.append(column(f'A1_column_{i:02}', (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1))

config = yaml.safe_load((ROOT/'config/retrofit/config.yaml').read_text(encoding='utf-8'))
config.pop('InteriorTrainingEnvironment', None)
config.update(ProjectName='A1 北角上润中心 14F', ConfigID='A1',
              BuildingTypology='existing_office_floor', ConversionGoal='adaptive_reuse',
              TargetSpaces=[], FunctionalRelations=[])
config['ExistingBuilding'] = dict(
    boundary=[point(*p) for p in outline],
    door_positions=[point(512,298), point(550,298)],
    fixed_objects=fixed, structure_schema_version=2,
)
# This task defines the existing environment, not a new residential programme.
# Disable rule-generated corridors: the current solver uses the core bounding
# box and would invent a transverse corridor for this stepped core.
config['FloorPartition'].update(enabled=False, grid_size=0.25)
config['FloorPartition']['residential'].update(
    unit_count=4, target_area_mode='equal', target_areas=[], corridor_width=1.5,
    opening_width=1.5, opening_side='south', export_config_prefix='A1_unit')
config['SeedGrowth']['enabled'] = False
config['Training'].update(training_stage='room_training', ckpt_path='', resume_mode='fresh')
config['SurveyReference'] = dict(
    config_id='A1', units='m', status='image_estimate_not_survey', scope='whole_floor',
    source_images=['../../../北角上润中心尺寸(1).jpg', '../../../北角上润中心尺寸（2).jpg'],
    image_size_px=[1000,750], reference_dimension_mm=1500,
    reference_corridor_x_px=[512,550], reference_width_px=38,
    metres_per_pixel=SCALE, origin_pixel=[170,559],
    coordinates='x right, y up; image-aligned, not true north',
    boundary_pixels=[list(p) for p in outline],
    traffic_core_rectangles_px={name:list(p) for name,p in core_boxes},
    retained_corridor_rect_px=[512,298,550,476],
    column_rectangles_px=[list(p) for p in column_boxes],
    assumptions=[
        '以通道两侧内边线 x=512 与 x=550 的约38像素间距标定1.5m。',
        '整层建模；橙色Unit A不是整层边界。图二蓝色区域包含电梯、楼梯、大堂及附属卫生间。',
        '交通核用五个相接矩形表达，中央纵向通道单列为retained_circulation。',
        '外轮廓作正交化近似；楼梯端部圆角按外包矩形简化；未做透视校正。',
        '16个核外黑色柱位按可见填充估计；核内黑色构件已由交通核整体保护。',
        '原Unit A/B/C/D隔墙不认定为承重墙，未加入固定结构约束。',
        '仅新建环境，未指定室内房间任务；四户均分为可编辑占位参数，未启用分区生成。',
        '入口线表示交通核大堂与中央通道的连接截面，不是室外入口。',
    ])

class Dumper(yaml.SafeDumper):
    pass
def represent_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data,
        flow_style=bool(data) and all(isinstance(x, (int,float)) for x in data))
Dumper.add_representer(list, represent_list)
(HERE/'config.yaml').write_text(
    '# A1 | 北角上润中心14F | 长度:m，面积:m² | 影像估算，非实测\n'
    '# 1500mm通道为唯一比例基准；建模说明见README.md。\n'
    + yaml.dump(config, Dumper=Dumper, allow_unicode=True, sort_keys=False), encoding='utf-8')

# Round-trip through the actual project geometry compiler.
loaded = yaml.safe_load((HERE/'config.yaml').read_text(encoding='utf-8'))
problem = build_floor_partition_problem(loaded)
boundary = problem.boundary
assert boundary.is_valid
for item in normalize_structures(loaded['ExistingBuilding']['fixed_objects']):
    shape = fixed_polygon(item)
    assert shape.is_valid and shape.area > 0, item['id']
    assert shape.difference(boundary).area < 1e-6, item['id']
corridor = fixed_polygon(next(x for x in fixed if x['type']=='retained_circulation'))
assert abs(corridor.bounds[2]-corridor.bounds[0]-1.5) < 1e-6
assert corridor.intersection(problem.traffic_core).area < 1e-6
assert abs(corridor.boundary.intersection(problem.traffic_core.boundary).length-1.5) < 1e-6
assert problem.traffic_core.geom_type == 'Polygon'
assert abs(LineString(loaded['ExistingBuilding']['door_positions']).length-1.5) < 1e-6
for source in loaded['SurveyReference']['source_images']:
    assert Image.open(HERE/source).size == (1000,750)

metrics = dict(config_id='A1', geometry_validation='passed', length_unit='m', area_unit='m2',
    width=boundary.bounds[2], height=boundary.bounds[3],
    gross_footprint_area=boundary.area, traffic_core_area=problem.traffic_core.area,
    retained_corridor_area=corridor.area, retained_corridor_width=1.5,
    retained_corridor_length=corridor.bounds[3]-corridor.bounds[1],
    column_count=len(column_boxes), free_area=boundary.difference(problem.fixed_union).area,
    training_executed=False)
(HERE/'validation.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')

fig, ax = plt.subplots(figsize=(12,7))
ax.add_patch(Patch(list(boundary.exterior.coords), facecolor='#f5f2e9',edgecolor='#172b3a',lw=2))
colors={'traffic_core':'#76a9ed','retained_circulation':'#f7cc6f','column':'#303640'}
for item in fixed:
    ax.add_patch(Patch(list(fixed_polygon(item).exterior.coords), facecolor=colors[item['type']],
                       edgecolor='white' if item['type']=='traffic_core' else '#303640', lw=.6))
ax.text(14.2,12.8,'TRAFFIC CORE',ha='center',weight='bold',fontsize=13)
x1,y1,x2,y2 = corridor.bounds
ax.annotate('',xy=(x1, y2-1),xytext=(x2,y2-1),arrowprops=dict(arrowstyle='<->',color='#172b3a'))
ax.text((x1+x2)/2,y2-1.6,'1.50 m',ha='center',fontsize=9)
ax.text((x1+x2)/2,(y1+y2)/2-1,'RETAINED',rotation=90,ha='center',fontsize=8)
ax.set(xlim=(-1,30),ylim=(-1,16.5),xlabel='x (m)',ylabel='y (m)',
       title='A1 | Max Share Centre, 14F | Existing building environment\nImage-derived dimensions; reference corridor = 1.500 m')
ax.set_aspect('equal'); ax.grid(alpha=.15); ax.set_axisbelow(True)
ax.legend(handles=[LegendPatch(color=v,label=k.replace('_',' ').title()) for k,v in colors.items()],
          loc='upper right',fontsize=8)
fig.tight_layout()
fig.savefig(HERE/'A1_environment.png',dpi=160)
fig.savefig(HERE/'A1_environment.svg')
plt.close(fig)
print(json.dumps(metrics,indent=2))
