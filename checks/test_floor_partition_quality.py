import copy
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import yaml
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.affinity import scale
from core.envs.structure_geometry import fixed_polygon
from core.floor_partition import build_floor_partition_problem, run_residential_floor_partition, export_unit_configs
from core.floor_partition.circulation import circulation_candidates
from core.floor_partition.quality import validate_partition
from core.floor_partition.structured import candidate_partitions
from core.floor_partition.training import train_partition_policy
from checks.test_floor_partition import fixture


class FloorQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=yaml.safe_load((Path(__file__).resolve().parents[1]/'config/a1/config.yaml').read_text(encoding='utf-8'))
        cls.problem=build_floor_partition_problem(cls.config)
        cls.pool=candidate_partitions(cls.problem)
        cls.result,cls.report=cls.pool[0]

    def test_modular_symmetric_geometry(self):
        p=self.problem
        self.assertTrue(p.boundary.equals(scale(p.boundary,xfact=-1,yfact=1,origin=(15,0))))
        self.assertTrue(p.traffic_core.equals(scale(p.traffic_core,xfact=-1,yfact=1,origin=(15,0))))
        columns=[i for i in p.fixed_objects if i['type']=='column']
        self.assertEqual(len(columns),16)
        for c in columns:
            self.assertEqual(c['size'],[.6,.8])
            self.assertTrue(all(v==int(v) for v in c['center']))
            self.assertLess(fixed_polygon(c).difference(p.boundary).area,1e-7)
        self.assertFalse(any(i['type']=='retained_circulation' for i in p.fixed_objects))

    def test_candidate_pool_hard_acceptance(self):
        for result,report in self.pool:
            self.assertTrue(validate_partition(self.problem,result)['valid'])
            self.assertAlmostEqual(report['structure_alignment_ratio'],1.)
            for metric in report['units'].values():
                self.assertGreaterEqual(metric['facade_length']+1e-6,metric['required_facade_length'])
        self.assertLessEqual(self.result.corridor.area,31.5)

    def test_infeasible_daylight_rejected(self):
        c=copy.deepcopy(self.config); c['FloorPartition']['quality']['facade_per_area']=10
        with self.assertRaisesRegex(ValueError,'未找到'):
            candidate_partitions(build_floor_partition_problem(c))

    def test_off_structure_cut_rejected(self):
        units=dict(self.result.unit_polygons); names=list(units)
        merged=unary_union([units[names[0]],units[names[1]]])
        left=merged.intersection(box(-1,-1,8.2,20)).buffer(0)
        units[names[0]]=left; units[names[1]]=merged.difference(left).buffer(0)
        report=validate_partition(self.problem,replace(self.result,unit_polygons=units))
        self.assertIn('partition_edge_off_structure_grid',report['errors'])

    def test_disconnected_result_rejected_before_export(self):
        units=dict(self.result.unit_polygons); uid=next(iter(units))
        units[uid]=unary_union([box(0,0,1,1),box(2,2,3,3)])
        bad=replace(self.result,unit_polygons=units)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError): export_unit_configs(self.config,bad,folder)

    def test_export_preserves_column_holes_and_net_area(self):
        with tempfile.TemporaryDirectory() as folder:
            paths=export_unit_configs(self.config,self.result,folder)
            for path in paths:
                c=yaml.safe_load(path.read_text(encoding='utf-8'))
                uid=c['FloorPartitionResult']['unit_id']; b=c['ExistingBuilding']
                net=Polygon(b['boundary']).difference(unary_union([fixed_polygon(i) for i in b['fixed_objects']]))
                self.assertLess(net.symmetric_difference(self.result.unit_polygons[uid]).area,1e-6)
                self.assertFalse(c['FloorPartition']['enabled'])
                self.assertEqual(c['Training']['training_stage'],'room_training')

    def test_rl_parameters_update_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('core.floor_partition.training.candidate_partitions',return_value=self.pool):
                _,report,summary=train_partition_policy(self.problem,self.config,folder,64)
            self.assertGreater(summary['parameter_delta'],0)
            self.assertGreater(summary['final_expected_reward'],summary['initial_expected_reward'])
            self.assertEqual(summary['episodes_completed'],64)
            self.assertTrue((Path(folder)/'partition_policy.pt').is_file())
            self.assertTrue(report['valid'])

    def test_stop_request_does_not_claim_training(self):
        with tempfile.TemporaryDirectory() as folder:
            stop=Path(folder)/'stop'; stop.touch()
            with patch('core.floor_partition.training.candidate_partitions',return_value=self.pool):
                _,_,summary=train_partition_policy(self.problem,self.config,folder,8,stop)
            self.assertTrue(summary['stopped'])
            self.assertEqual(summary['episodes_completed'],0)
            self.assertEqual(summary['parameter_delta'],0)

    def test_north_corridor_has_positive_area(self):
        c=fixture(); c['FloorPartition']['residential']['opening_side']='north'
        options=list(circulation_candidates(build_floor_partition_problem(c)))
        self.assertTrue(options)
        self.assertTrue(all(option.corridor.area>0 for option in options))

    def test_nonfinite_and_negative_targets_rejected(self):
        for value in [float('nan'),float('inf'),-1,0]:
            c=fixture(); c['FloorPartition']['residential']['target_areas'][0]=value
            with self.assertRaises(ValueError): build_floor_partition_problem(c)


if __name__=='__main__': unittest.main()
