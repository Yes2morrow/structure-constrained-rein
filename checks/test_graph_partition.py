import copy
import unittest
import tempfile
import io
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch
import numpy as np
import matplotlib
matplotlib.use('Agg')
from shapely.geometry import box

from core.seed_growth.contracts import build_problem
from core.seed_growth.environment import SeedLayoutEnv
from core.seed_growth.geometry import decode_layout
from core.seed_growth.graph_partition import complete_partition,fit_graph_faces
from core.seed_growth.validation import validate_layout
from gui.seed_page import seed_rows,apply_seed_rows
from gui.environment_canvas import controls,scene,parse_scene
from gui.training_service import training_command


def fixture():
    return dict(ProjectType='adaptive_reuse', ExistingBuilding=dict(
        boundary=[[0,0],[12,0],[12,8],[0,8]],fixed_objects=[],door_positions=[[5,0],[6,0]]),
        AdaptiveReuseEnvironment=dict(grid_size=.5),SeedGrowth=dict(enabled=True),
        Training=dict(max_steps=3,batch_size=4,seed=42,episodes=1,lr=.0003),
        TargetSpaces=[dict(id='living',role='residual',target_area=32,area_range=[10,40],
                           min_width=.8,initial_rect=[1,1,2,2]),
                      dict(id='a',initial_rect=[1,2,5,6],target_area=16,area_range=[14,18],aspect_range=[1,2]),
                      dict(id='b',initial_rect=[6,2,10,6],target_area=16,area_range=[14,18],aspect_range=[1,2])],
        FunctionalRelations=[{'from':'living','to':'a','type':'adjacent'},
                             {'from':'a','to':'b','type':'adjacent','min_shared_length':2}])


class GraphPartitionTest(unittest.TestCase):
    def test_public_parameters_survive_actual_target_editor(self):
        from streamlit.testing.v1 import AppTest
        import gui.adaptive_reuse_page as page
        state=fixture()
        source="""
from gui.config_store import load_config
from gui.adaptive_reuse_page import _render_target_editor
_render_target_editor(load_config('test'),'test')
"""
        loader=lambda _:copy.deepcopy(state)
        def save(c,i): state.update(copy.deepcopy(c)); return 'mock.yaml'
        with patch('gui.config_store.load_config',loader),patch.object(page,'load_config',loader),patch.object(page,'save_config',save):
            app=AppTest.from_string(source,default_timeout=30).run()
            self.assertFalse(app.exception,app.exception)
            self.assertFalse(app.error,[e.value for e in app.error])
            next(n for n in app.number_input if n.label=='客厅公共空间最小面积（㎡）').set_value(12.).run()
            self.assertFalse(app.exception,app.exception)
            self.assertFalse(app.error,[e.value for e in app.error])
        self.assertEqual(state['TargetSpaces'][0]['role'],'residual')
        self.assertEqual(state['TargetSpaces'][0]['residual_min_area'],12.)
        # The editor now stores every node pair, including unconstrained pairs.
        relations={(r['from'],r['to']):r for r in state['FunctionalRelations']}
        self.assertEqual(len(state['FunctionalRelations']),3)
        self.assertEqual(relations[('living','a')]['type'],'adjacent')
        self.assertEqual(relations[('a','b')]['type'],'adjacent')
        self.assertEqual(relations[('a','b')]['min_shared_length'],2)
        self.assertEqual(relations[('living','b')]['type'],'none')

    def test_real_mappo_update_and_checkpoint_with_public_node(self):
        import torch
        from common.training import build_mapppo_agent
        from core.seed_growth.artifacts import RunArtifacts
        from core.seed_growth.training import run_training,restore_checkpoint
        c=fixture(); c['AlgorithmAdvanced']=dict(hidden_size=16,ppo_epochs=1)
        env=SeedLayoutEnv(c)
        with patch('torch.cuda.is_available',return_value=False):
            agent=build_mapppo_agent(env,c,reset_seed=42)
        before=[p.detach().clone() for p in agent.actors[0].parameters()]
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            artifacts=RunArtifacts(Path(folder),c,env.problem,env.proxy)
            result=run_training(env,agent,artifacts,2)
            self.assertEqual(result['snapshot'].completed_episodes,2)
            self.assertEqual(len(agent.actors),2)
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,agent.actors[0].parameters())))
            self.assertTrue(all(torch.isfinite(p).all() for p in agent.actors[0].parameters()))
            restored,reward=restore_checkpoint(folder,env,agent)
            self.assertEqual(restored,result['snapshot'])
            self.assertEqual(set(dict(restored.seeds)),{'a','b'})

    def test_two_policies_three_nodes_and_no_public_seed_or_canvas_rectangle(self):
        c=fixture(); before=copy.deepcopy(c)
        with patch('core.seed_growth.geometry.decode_layout',side_effect=AssertionError('training decoded')):
            env=SeedLayoutEnv(c)
            self.assertEqual(env.ids,('a','b'))
            self.assertEqual(env.proxy.ids,('living','a','b'))
            self.assertEqual(env.get_state().shape,(2,32))
            self.assertNotIn('living',dict(env.snapshot(0).seeds))
            self.assertEqual(env.calculate_metrics()['entrance_reserved'],True)
            _,reward,_,_=env.step([4,4]); self.assertEqual(len(reward),2)
            self.assertTrue(np.isfinite(reward).all())
        self.assertEqual(c,before)
        self.assertFalse(any(k[0] in ('seed','agent') and k[1]=='living' for k in controls(c)))
        self.assertEqual(parse_scene(scene(c,760,480,{}),c,760,480)['TargetSpaces'],c['TargetSpaces'])
        rows=seed_rows(c['TargetSpaces']);self.assertEqual(len(rows),2)
        self.assertEqual(apply_seed_rows(c['TargetSpaces'],rows,{})[0],c['TargetSpaces'][0])
        c['SeedGrowth']['enabled']=False
        self.assertTrue(training_command(c,'test')[2].endswith('train_seed_layout.py'))

    def test_exact_target_rooms_can_be_fitted_without_inflation(self):
        env=SeedLayoutEnv(fixture())
        initial={'a':box(1,2,5,6),'b':box(6,2,10,6)}
        output,records=fit_graph_faces(env.problem,env.seeds,initial)
        report=validate_layout(env.problem,env.seeds,output)
        self.assertTrue(records)
        self.assertTrue(report['geometry_valid'],report['errors'])
        self.assertTrue(report['relations_satisfied'])
        self.assertAlmostEqual(output['a'].area,16)
        self.assertAlmostEqual(output['b'].area,16)
        self.assertAlmostEqual(report['unassigned_area'],0)
        self.assertTrue(report['entrance_in_public_space'])
        self.assertGreater(report['relations'][0]['shared_length'],0)  # Includes hole rings.

    def test_decoder_partitions_every_square_metre_and_preserves_graph(self):
        env=SeedLayoutEnv(fixture()); result=decode_layout(env.problem,env.snapshot(0))
        self.assertEqual(set(result['polygons']),{'living','a','b'})
        self.assertEqual(result['status'],'valid',result['validation'])
        self.assertTrue(result['validation']['entrance_in_public_space'])
        self.assertAlmostEqual(sum(p.area for p in result['polygons'].values()),env.problem.free_space.area)

    def test_disconnected_remainder_is_rejected_without_discarding_fragments(self):
        env=SeedLayoutEnv(fixture())
        polygons=complete_partition(env.problem,{'a':box(4,0,7,8),'b':box(8,2,10,6)})
        report=validate_layout(env.problem,env.seeds,polygons)
        self.assertIn('living:public_space_disconnected',report['errors'])
        self.assertEqual(polygons['living'].geom_type,'MultiPolygon')
        self.assertFalse(report['entrance_in_public_space'])

    def test_door_is_full_segment_on_boundary_and_not_a_free_point(self):
        c=fixture(); c['ExistingBuilding']['door_positions']=[[5,1],[6,1]]
        with self.assertRaisesRegex(ValueError,'建筑边界'): build_problem(c)
        c=fixture(); c['ExistingBuilding']['fixed_objects']=[dict(id='wall',type='fixed',rect=[5,0,6,1])]
        with self.assertRaisesRegex(ValueError,'固定结构'): build_problem(c)

    def test_seed_motion_cannot_take_reserved_entry(self):
        env=SeedLayoutEnv(fixture())
        with self.assertRaises(ValueError): env.proxy.cell((5.25,.25))
        with self.assertRaises(ValueError): env.problem.validate_seeds({'a':(5.25,.25),'b':(8,4)})

    def test_aspect_violation_cannot_be_hidden_by_contact(self):
        env=SeedLayoutEnv(fixture())
        p=complete_partition(env.problem,{'a':box(1,3,9,5),'b':box(9,2,11,6)})
        self.assertIn('a:aspect_out_of_range',validate_layout(env.problem,env.seeds,p)['errors'])


if __name__=='__main__': unittest.main()
