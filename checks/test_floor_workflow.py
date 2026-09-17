import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml
from shapely.geometry import box, mapping, shape
from streamlit.testing.v1 import AppTest

from core.floor_partition.workflow import prepare_units, read_floor_snapshot, read_unit_result, composition
from core.floor_partition.snapshot import snapshot_id
from gui.partition_overlay import partition_overlay


class FloorWorkflowTests(unittest.TestCase):
    def test_preview_hidden_does_not_solve_and_visible_is_read_only_pink(self):
        from checks.test_joint_partition import case
        c,p,r=case(); c['Training']={'training_stage':'floor_partition'}
        original=copy.deepcopy(c)
        with patch('gui.partition_overlay.run_residential_floor_partition',return_value=(p,r)) as solve:
            self.assertEqual(partition_overlay(c,680,400),[])
            solve.assert_not_called()
            objects=partition_overlay(c,680,400,True,50)
            self.assertTrue(objects)
            self.assertTrue(all(not o['selectable'] and not o['evented'] for o in objects))
            fills={o['fill'] for o in objects if o['type']=='path'}
            self.assertEqual(fills,{'#f9a8d4','#db2777'})
            self.assertTrue(all(o['opacity']==.325 for o in objects if o['type']=='path'))
            self.assertTrue(all(o['stroke']=='#facc15' for o in objects if o['type']=='polyline'))
            c['Training']['training_stage']='room_training'
            self.assertEqual(partition_overlay(c,680,400,True),[])
            self.assertEqual(solve.call_count,1)
        c['Training']['training_stage']='floor_partition'
        self.assertEqual(c,original)

    def test_export_handoff_keeps_net_coordinates_and_never_overwrites_edits(self):
        from checks.test_joint_partition import case
        from core.floor_partition.export import export_unit_configs
        from core.floor_partition.walls import update_net_targets
        c,p,r=case();c.update(ConfigID='test',ProjectType='adaptive_reuse',Training={})
        r=update_net_targets(p,r)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            exports=root/'exports'
            export_unit_configs(c,r,exports)
            floor=read_floor_snapshot(exports/'floor_plan.json')
            path=prepare_units(exports,root/'config',root/'workflows')
            manifest=json.loads(path.read_text())
            self.assertEqual(len(manifest['units']),2)
            for uid,item in manifest['units'].items():
                target=Path(item['config_path'])
                config=yaml.safe_load(target.read_text(encoding='utf8'))
                self.assertEqual(config['Training']['training_stage'],'room_training')
                self.assertEqual(config['FloorPartitionResult']['floor_outline'],c['ExistingBuilding']['boundary'])
                self.assertTrue(config['SeedGrowth']['enabled'])
                from shapely.geometry import Polygon
                from core.envs.structure_geometry import fixed_polygon
                from shapely.ops import unary_union
                actual=Polygon(config['ExistingBuilding']['boundary']).difference(unary_union([
                    fixed_polygon(i) for i in config['ExistingBuilding']['fixed_objects']]))
                self.assertLess(actual.symmetric_difference(shape(floor['units'][uid]['net_boundary'])).area,1e-5)
                config['ProjectName']='user room edits'
                target.write_text(yaml.safe_dump(config),encoding='utf8')
            self.assertEqual(prepare_units(exports,root/'config',root/'workflows'),path)
            self.assertEqual(yaml.safe_load(target.read_text())['ProjectName'],'user room edits')
            floor['wall_thickness']=9
            (exports/'floor_plan.json').write_text(json.dumps(floor))
            with self.assertRaisesRegex(ValueError,'方案编号'):
                read_floor_snapshot(exports/'floor_plan.json')

    def test_exact_result_validation_and_complete_assembly(self):
        from core.seed_growth.artifacts import config_key
        from core.seed_growth.contracts import build_problem,SeedSnapshot
        a,b=box(0,0,6,8),box(6.2,0,12.2,8)
        floor=dict(units={'a':{'net_boundary':mapping(a)},'b':{'net_boundary':mapping(b)}})
        floor['plan_id']=snapshot_id(floor)
        c=dict(ExistingBuilding=dict(boundary=list(a.exterior.coords),fixed_objects=[]),
               AdaptiveReuseEnvironment={'grid_size':.5},SeedGrowth={'enabled':True},
               FloorWorkflow=dict(plan_id=floor['plan_id'],unit_id='a'),
               TargetSpaces=[dict(id='room',seed=[3,4],initial_rect=[1,1,2,2],target_area=48,
                                  area_range=[47,49],aspect_range=[1,3])],FunctionalRelations=[])
        c=json.loads(json.dumps(c))
        problem=build_problem(c)
        snapshot=SeedSnapshot.capture(problem,{'room':(3,4)},1,0)
        result=dict(status='valid',config_key=config_key(c),snapshot_key=snapshot.key,
                    snapshot=snapshot.to_dict(),polygons={'room':mapping(a)})
        with tempfile.TemporaryDirectory() as td:
            run=Path(td)
            (run/'config.yaml').write_text(yaml.safe_dump(c),encoding='utf8')
            (run/'result.json').write_text(json.dumps(result))
            (run/'last_valid.json').write_text(json.dumps({'path':'result.json'}))
            rooms,source=read_unit_result(floor,'a',run,c)
            self.assertEqual(source['snapshot_key'],snapshot.key)
            self.assertFalse(composition(floor,{'a':rooms})['complete'])
            self.assertTrue(composition(floor,{'a':rooms,'b':{'room':b}})['complete'])
            with self.assertRaises(ValueError): composition(floor,{'a':{'room':b}})
            with self.assertRaises(ValueError): composition(floor,{'a':{'room':box(0,0,3,8)}})
            changed=copy.deepcopy(c);changed['TargetSpaces'][0]['target_area']=47
            with self.assertRaisesRegex(ValueError,'过期'): read_unit_result(floor,'a',run,changed)
            with self.assertRaisesRegex(ValueError,'其他户型'): read_unit_result(floor,'b',run,c)

    def test_empty_unit_editor_and_training_guard(self):
        from gui.training_service import training_command
        c=dict(ProjectType='adaptive_reuse',Training={'training_stage':'room_training'},
               TargetSpaces=[],SeedGrowth={'enabled':True},FloorWorkflow={'plan_id':'test'})
        with self.assertRaisesRegex(ValueError,'添加房间'): training_command(c,'test')
        with patch('gui.config_store.load_config',return_value=c):
            app=AppTest.from_string('''
from gui.config_store import load_config
from gui.adaptive_reuse_page import _render_target_editor
_render_target_editor(load_config('test'),'workflow_test')
''').run()
        self.assertFalse(app.exception,app.exception)
        self.assertTrue(any('尚未配置房间' in item.value for item in app.info))
        self.assertIn('id',app.dataframe[0].value.columns)

    def test_preview_adoption_and_pending_units_page(self):
        from checks.test_joint_partition import case
        from core.floor_partition.walls import update_net_targets
        import gui.floor_workflow_page as page
        c,p,r=case();c.update(ConfigID='test',ProjectType='adaptive_reuse',Training={})
        r=update_net_targets(p,r)
        source='''
import streamlit as st
from gui.config_store import load_config
from gui.floor_workflow_page import render_floor_results,render_floor_workflow
route=st.session_state.get('workflow_route',{})
if route.get('view')=='整体合成':
    render_floor_workflow()
else:
    render_floor_results(load_config('test'),'test')
'''
        with tempfile.TemporaryDirectory() as td, \
             patch.object(page,'RESULTS_DIR',Path(td)/'results'), \
             patch.object(page,'CONFIG_ROOT',Path(td)/'config'), \
             patch.object(page,'WORKFLOW_ROOT',Path(td)/'workflows'), \
             patch('gui.config_store.load_config',return_value=c), \
             patch('core.floor_partition.run_residential_floor_partition',return_value=(p,r)):
            app=AppTest.from_string(source,default_timeout=30).run()
            self.assertFalse(app.exception,app.exception)
            next(b for b in app.button if b.label=='生成当前参数推演（非训练结果）').click().run()
            self.assertFalse(app.exception,app.exception)
            self.assertFalse(app.error,[e.value for e in app.error])
            next(b for b in app.button if b.label=='采用此方案，创建逐户训练清单').click().run()
            self.assertFalse(app.exception,app.exception)
            self.assertFalse(app.error,[e.value for e in app.error])
            self.assertEqual(len(app.dataframe[0].value),2)
            self.assertTrue(any('过程草图' in w.value for w in app.warning))
            self.assertFalse(app.get('download_button'))


if __name__=='__main__': unittest.main()
