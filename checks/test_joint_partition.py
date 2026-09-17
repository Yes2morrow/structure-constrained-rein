"""Behavioural checks for dynamic, bounded, joint layout improvement."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from shapely.geometry import box, LineString
from shapely.ops import unary_union

from core.floor_partition.contracts import build_floor_partition_problem, target_areas_for_area
from core.floor_partition.circulation import CirculationLayout
from core.floor_partition.growth import PartitionResult
from core.floor_partition.structured import unit_doors
from core.floor_partition.quality import validate_partition
from core.floor_partition.joint import JointPartitionEnv, Action, objective
from core.floor_partition.training import train_partition_policy, GraphPolicy, _observation


def case():
    config = {'ExistingBuilding': {'boundary': [[0,0],[12,0],[12,10],[0,10]],
        'door_positions': [[5,8],[7,8]],
        'fixed_objects': [{'id':'core','type':'traffic_core','rect':[4,8,8,10]}]},
        'FloorPartition': {'grid_size': .5, 'residential': {'unit_count':2,
            'corridor_width':1.5,'door_width':.9,'min_door_spacing':1.,'opening_side':'south'},
            'quality': {'area_tolerance':.45,'min_structure_alignment':.1},
            'joint_search': {'radii':[20], 'max_cells':500,'max_area':120.,
                'max_actions':12,'proposals':64,'seconds_per_action':10.,'cell_size':1.5},
            'rl': {'steps_per_episode':3,'ppo_epochs':2,'seed':4}}}
    problem = build_floor_partition_problem(config)
    corridor = box(3,5,9,8)
    free = problem.boundary.difference(problem.fixed_union).difference(corridor)
    units = {'unit_01': free.intersection(box(0,0,6,10)),
             'unit_02': free.intersection(box(6,0,12,10))}
    opening = LineString([(5,8),(7,8)])
    doors = unit_doors(problem, CirculationLayout(corridor,opening,'south'),units)
    targets, scale = target_areas_for_area(problem, free.area)
    result = PartitionResult(corridor,opening,'south',free,doors,units,
        dict(zip(units,targets)),dict.fromkeys(units,0.),scale)
    return config, problem, result


class JointPartitionTests(unittest.TestCase):
    def test_exact_cells_and_real_adjacency(self):
        _,p,r=case(); env=JointPartitionEnv(p,r)
        self.assertLess(unary_union(env.cells).symmetric_difference(env.free).area,1e-7)
        for a,b in env.edges:
            self.assertGreater(env.cells[a].boundary.intersection(env.cells[b].boundary).length,1e-7)

    def test_neighbourhood_budgets_and_outside_preservation(self):
        _,p,r=case()
        p=replace(p,search_settings={**p.search_settings,'max_cells':20,'max_area':10.})
        env=JointPartitionEnv(p,r); action=env.actions()[0]
        ids,w=env.neighbourhood(action)
        self.assertLessEqual(len(ids),20); self.assertLessEqual(w.area,10.+1e-7)
        result,_,_=env.step(action)
        for uid in r.unit_polygons:
            self.assertLess(r.unit_polygons[uid].difference(w).symmetric_difference(result.unit_polygons[uid].difference(w)).area,1e-7)

    def test_corridor_can_change_with_units_and_doors(self):
        _,p,r=case(); env=JointPartitionEnv(p,r)
        before=objective(env.report)
        for action in env.actions():
            env.step(action)
            if env.current.corridor.symmetric_difference(r.corridor).area>1e-6: break
        self.assertTrue(validate_partition(p,env.current)['valid'])
        self.assertLess(env.current.corridor.area,r.corridor.area)
        self.assertGreater(objective(env.report),before)
        self.assertTrue(any(env.current.unit_polygons[u].symmetric_difference(r.unit_polygons[u]).area>1e-6 for u in r.unit_polygons))
        self.assertNotEqual(env.current.doors,r.doors)

    def test_false_free_space_is_rejected(self):
        _,p,r=case()
        bad=replace(r,allocatable_space=r.allocatable_space.intersection(box(0,0,6,10)))
        self.assertIn('allocatable_space_mismatch',validate_partition(p,bad)['errors'])

    def test_impossible_neighbourhood_is_masked(self):
        _,p,r=case()
        p=replace(p,search_settings={**p.search_settings,'max_area':.00001})
        self.assertEqual(JointPartitionEnv(p,r).actions(),[])

    def test_corridor_narrow_spur_is_rejected(self):
        _,p,r=case()
        corridor=r.corridor.union(box(5.8,3,6.2,5))
        changed=replace(r,corridor=corridor)
        self.assertIn('corridor_thin_appendage',validate_partition(p,changed)['errors'])

    def test_daylight_depth_is_checked_during_acceptance(self):
        _,p,r=case()
        strict=replace(p,settings={**p.settings,'daylight_depth':.01,'min_daylight_coverage':.99})
        self.assertTrue(any('daylight_depth_deficit' in e for e in validate_partition(strict,r)['errors']))

    def test_ppo_multistep_checkpoint_and_reload(self):
        torch.set_num_threads(1)
        c,p,r=case()
        with tempfile.TemporaryDirectory() as td, patch('core.floor_partition.training.candidate_partitions',return_value=[(r,validate_partition(p,r))]):
            result,report,summary=train_partition_policy(p,c,td,episodes=2)
            self.assertTrue(report['valid'])
            self.assertEqual(summary['algorithm'],'graph_neighbourhood_PPO')
            self.assertGreater(summary['steps_completed'],2)
            self.assertGreater(summary['parameter_delta'],0.)
            saved=torch.load(Path(td)/'partition_policy.pt',weights_only=False)
            model=GraphPolicy(); model.load_state_dict(saved['state_dict'])
            env=JointPartitionEnv(p,result); actions=env.actions(); x,edges=_observation(env,actions)
            logits,value=model(x,edges,actions)
            self.assertEqual(len(logits),len(actions)+1)
            self.assertTrue(torch.isfinite(logits).all())

    def test_stop_has_no_training_updates(self):
        c,p,r=case()
        with tempfile.TemporaryDirectory() as td, patch('core.floor_partition.training.candidate_partitions',return_value=[(r,validate_partition(p,r))]):
            stop=Path(td)/'stop'; stop.touch()
            _,_,s=train_partition_policy(p,c,td,episodes=2,stop_file=stop)
            self.assertTrue(s['stopped']); self.assertEqual(s['parameter_delta'],0.)


if __name__=='__main__': unittest.main()
