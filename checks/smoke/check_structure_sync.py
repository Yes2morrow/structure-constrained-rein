"""Serialization, movement, scaling, deletion and legacy migration regressions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from copy import deepcopy
import math
import yaml
from tempfile import TemporaryDirectory
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
from core.envs.structure_geometry import column, wall, normalize_structures, parameter_rows
from gui.structure_canvas import canvas_objects, parse_canvas, signature, viewport
from gui.adaptive_reuse_page import CONSTRAINT_STYLES


def test_roundtrip():
    boundary = [[-2, -4],[30,-4],[30,18],[-2,18]]
    items = normalize_structures([column('c1',3,4,.45,.65),wall('w1','shear_wall',4,5,9,8,.15,.08)])
    draw = canvas_objects(items,boundary,900,600,CONSTRAINT_STYLES)
    restored = parse_canvas(yaml.safe_load(yaml.safe_dump(draw)),items,boundary,900,600,'column')
    assert signature(restored)==signature(items)
    def fabric_round(v):
        if isinstance(v,float): return round(v,2)
        if isinstance(v,list): return [fabric_round(x) for x in v]
        if isinstance(v,dict): return {k:fabric_round(x) for k,x in v.items()}
        return v
    serialized = fabric_round(draw)
    assert signature(parse_canvas(serialized,items,boundary,900,600,'column'))==signature(items)
    serialized[1]['left'] += 10
    shifted=parse_canvas(serialized,items,boundary,900,600,'column')
    assert next(i for i in shifted if i['id']=='w1')['left_thickness']==.15
    moved = deepcopy(draw)
    scale,_,_ = viewport(boundary,900,600)
    moved[1]['left'] += 2*scale
    moved[1]['top'] -= scale
    moved[1]['scaleX'] = 1.5
    moved[1]['scaleY'] = 2
    updated = parse_canvas(moved,items,boundary,900,600,'column')
    w = next(i for i in updated if i['id']=='w1')
    assert math.isclose(w['left_thickness'],.3)
    assert math.isclose(w['right_thickness'],.16)
    assert math.isclose(math.dist(w['start'],w['end']),math.sqrt(34)*1.5)
    assert w['type']=='shear_wall'
    assert len(parse_canvas(draw[:1],items,boundary,900,600,'column'))==1
    assert parse_canvas([],items,boundary,900,600,'column')==[]
    # New lines in each drag direction must preserve the chosen start/end.
    for x1,y1,x2,y2 in [(100,100,300,200),(300,200,100,100),(100,200,300,100),(300,100,100,200),(100,100,100,300)]:
        line=dict(type='line',left=min(x1,x2),top=min(y1,y2),width=abs(x2-x1),height=abs(y2-y1),x1=x1,y1=y1,x2=x2,y2=y2)
        got=parse_canvas([line],[],boundary,900,600,'shear_wall',.15,.1)[0]
        _,ox,oy=viewport(boundary,900,600)
        assert math.dist(got['start'],[(x1-ox)/scale,(oy-y1)/scale])<1e-8
        assert math.dist(got['end'],[(x2-ox)/scale,(oy-y2)/scale])<1e-8
    # Rebuilding a closed core after moving/removing its walls.
    pts=[(0,0),(4,0),(4,4),(0,4),(0,0)]
    ring=normalize_structures([wall(f'w{i}','shear_wall',*a,*b,.1,.1) for i,(a,b) in enumerate(zip(pts,pts[1:]))])
    rendered=canvas_objects(ring,boundary,900,600,CONSTRAINT_STYLES)
    assert signature(parse_canvas(rendered,ring,boundary,900,600,'column'))==signature(ring)
    del rendered[0]
    assert not any(i.get('derived') for i in parse_canvas(rendered,ring,boundary,900,600,'column'))
    legacy=normalize_structures([dict(id='old',type='shear_wall',rect=[0,0,4,.2])])
    assert legacy[0]['start']==[0,.1]
    assert parameter_rows(updated)[1][0]['left']==w['left_thickness']
    from gui.config_store import load_config, save_config
    with TemporaryDirectory() as folder:
        path = str(Path(folder)/'config.yaml')
        config = {'ProjectType':'adaptive_reuse','ExistingBuilding':{'boundary':boundary,'fixed_objects':items}}
        with patch('gui.config_store.get_config_path',return_value=path):
            save_config(config,'qa')
            persisted=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
            assert persisted['ExistingBuilding']['structure_schema_version']==2
            assert signature(load_config('qa')['ExistingBuilding']['fixed_objects'])==signature(items)
            persisted['ExistingBuilding']['fixed_objects'][1]['rect']=[0,0,1,1]
            Path(path).write_text(yaml.safe_dump(persisted),encoding='utf-8')
            assert signature(load_config('qa')['ExistingBuilding']['fixed_objects'])==signature(items)
    print('PASS: stable IDs, YAML roundtrip, drag directions, asymmetric scaling, deletion, core refresh and legacy migration')


if __name__=='__main__':
    test_roundtrip()
