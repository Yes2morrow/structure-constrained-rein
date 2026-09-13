import copy
import unittest
from unittest.mock import patch
from pathlib import Path
import pandas as pd
from streamlit.testing.v1 import AppTest
from gui.config_store import load_config
import gui.adaptive_reuse_page as page
from core.seed_growth.contracts import build_problem
from gui.seed_page import seed_rows,apply_seed_rows,read_precise
from gui.training_service import training_command


class SeedUITest(unittest.TestCase):
    def test_target_editor_preserves_node_extensions(self):
        rooms=[dict(id='r',name='room',initial_rect=[0,0,4,4],target_area=16,area_range=[12,20],
                    seed=[1,2],min_width=1.2,shape_policy='limited_recess',shape_limits={'max_reflex':2})]
        result=page._records_to_targets(pd.DataFrame(page._target_records(rooms)),rooms)
        for key in ('seed','min_width','shape_policy','shape_limits'): self.assertEqual(result[0][key],rooms[0][key])

    def test_seed_edit_keeps_ids_and_checks_mapping(self):
        rooms=[dict(id='r',initial_rect=[0,0,4,4],target_area=16)]
        rows=seed_rows(rooms); rows[0]['x']=1.5
        result=apply_seed_rows(rooms,rows,{'r':{'max_reflex':2}})
        self.assertEqual(result[0]['id'],'r'); self.assertEqual(result[0]['seed'],[1.5,2.])
        self.assertNotIn('seed',rooms[0])
        with self.assertRaises(ValueError): apply_seed_rows(rooms,rows,{'missing':{}})

    def test_relation_editor_auto_completes_all_pairs_with_none(self):
        targets=[
            dict(id='living',name='客厅',role='residual',initial_rect=[0,0,2,2],target_area=10,area_range=[8,12]),
            dict(id='a',name='主卧',initial_rect=[0,0,2,2],target_area=10,area_range=[8,12]),
            dict(id='b',name='餐厅',initial_rect=[0,0,2,2],target_area=10,area_range=[8,12]),
        ]
        rows=page._relation_editor_rows(targets,[{'from':'b','to':'living','type':'adjacent','min_shared_length':1.2}])
        self.assertEqual([(row['from'],row['to']) for row in rows],[('living','a'),('living','b'),('a','b')])
        self.assertEqual(rows[0]['type'],'none')
        self.assertEqual(rows[1]['type'],'adjacent')
        self.assertAlmostEqual(rows[1]['min_shared_length'],1.2)
        self.assertEqual(rows[2]['type'],'none')

    def test_target_role_and_none_relations_survive_conversion_and_build_problem(self):
        targets=[
            dict(id='living',name='客厅',role='residual',initial_rect=[1,1,3,3],target_area=20,area_range=[10,30]),
            dict(id='a',name='卧室A',role='agent',initial_rect=[0,0,4,4],target_area=16,area_range=[14,18]),
            dict(id='b',name='卧室B',role='agent',initial_rect=[5,0,9,4],target_area=16,area_range=[14,18]),
        ]
        restored=page._records_to_targets(pd.DataFrame(page._target_records(targets)),targets)
        self.assertEqual(restored[0]['role'],'residual')
        relations=page._records_to_relations(pd.DataFrame(page._relation_editor_rows(targets,[
            {'from':'living','to':'a','type':'none'},
            {'from':'a','to':'b','type':'separate','min_distance':2.0},
        ])),targets)
        self.assertEqual(len(relations),3)
        config=dict(
            ProjectType='adaptive_reuse',
            ExistingBuilding=dict(boundary=[[0,0],[12,0],[12,8],[0,8]],fixed_objects=[],door_positions=[[5,0],[6,0]]),
            AdaptiveReuseEnvironment=dict(grid_size=.5),
            SeedGrowth=dict(enabled=True),
            TargetSpaces=restored,
            FunctionalRelations=relations,
        )
        problem=build_problem(config)
        self.assertEqual(len(problem.relations),1)
        self.assertEqual(problem.relations[0].kind,'separate')

    def test_launch_switch_is_opt_in_and_resume_uses_archived_config(self):
        c=dict(ProjectType='adaptive_reuse',Training={'episodes':500})
        self.assertTrue(training_command(c,'retrofit')[2].endswith('train_adaptive_reuse.py'))
        c['SeedGrowth']={'enabled':True}
        self.assertTrue(training_command(c,'retrofit')[2].endswith('train_seed_layout.py'))
        c['SeedGrowth']['resume_run']='test run'
        command=training_command(c,'retrofit')
        self.assertIn('--resume',command); self.assertIn('test run',command)
        self.assertNotIn('--config-id',command)
        self.assertEqual(command[-1],'500')

    def test_streamlit_save_preserves_seeds_and_relation_thresholds(self):
        state=copy.deepcopy(load_config('retrofit'))
        # This checks UI field preservation, independent of the user's live annotations.
        state['ExistingBuilding']['fixed_objects']=[]
        state['SeedGrowth']={'enabled':True}
        state['TargetSpaces'][0].update(seed=[15,14.5],min_width=1.2,shape_policy='limited_recess',shape_limits={'max_reflex':2})
        state['FunctionalRelations'][0].update(min_shared_length=1.5,min_clear_width=.9)
        source="""
from gui.adaptive_reuse_page import render_adaptive_reuse_config_page
from gui.config_store import load_config
render_adaptive_reuse_config_page(load_config('retrofit'),'retrofit')
"""
        loader=lambda _:copy.deepcopy(state)
        with patch('gui.config_store.load_config',loader),patch.object(page,'load_config',loader),\
             patch.object(page,'save_config',lambda c,i:state.update(copy.deepcopy(c))),\
             patch.object(page,'_render_plan_constraint_editor',lambda *args:None),\
             patch('core.seed_growth.geometry.decode_layout',side_effect=AssertionError('preview must not decode')):
            app=AppTest.from_string(source,default_timeout=40).run()
            self.assertFalse(app.exception,app.exception)
            self.assertTrue(next(c for c in app.checkbox if c.label=='在原矩形基础上启用图种子加速优化').value)
            for label in page.REWARD_LABELS.values():
                self.assertFalse(next(s for s in app.slider if s.label==label).disabled)
            self.assertTrue(any('原矩形基础布局' in m.value for m in app.markdown))
            next(b for b in app.button if b.label=='保存既有建筑环境').click().run()
            self.assertFalse(app.exception,app.exception)
            self.assertFalse(app.error,[e.value for e in app.error])
        self.assertEqual(state['TargetSpaces'][0]['seed'],[15.,14.5])
        self.assertEqual(state['TargetSpaces'][0]['shape_limits'],{'max_reflex':2})
        self.assertEqual(state['FunctionalRelations'][0]['min_shared_length'],1.5)
        self.assertEqual(state['FunctionalRelations'][0]['min_clear_width'],.9)

    def test_saved_result_is_read_with_its_own_node_snapshot(self):
        root=Path(__file__).resolve().parents[1]/'results2/seed_experimental'
        runs=[p for p in root.iterdir() if (p/'latest_precise.json').exists()]
        self.assertTrue(runs)
        config,result=read_precise(runs[0])
        ids={r['id'] for r in config['TargetSpaces']}
        self.assertEqual(set(result['snapshot']['seeds']),ids)
        self.assertEqual(set(result['polygons']),ids)


if __name__=='__main__': unittest.main()
