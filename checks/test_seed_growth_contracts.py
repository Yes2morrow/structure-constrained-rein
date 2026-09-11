"""CP1 checks: run with python -m unittest checks.test_seed_growth_contracts."""
import copy
import unittest
import yaml
from shapely.geometry import Point
from core.seed_growth.contracts import build_problem, SeedSnapshot, PreciseSchedule


def fixture():
    return {
        'ExistingBuilding':{'boundary':[[0,0],[20,0],[20,20],[0,20]],'fixed_objects':[
            {'id':'wall','type':'shear_wall','start':[10,1],'end':[10,19],
             'left_thickness':.3,'right_thickness':.1}]},
        'AdaptiveReuseEnvironment':{'grid_size':.5},
        'TargetSpaces':[
            {'id':'kitchen','seed':[4,5],'target_area':12,'area_range':[10,15],'shape_policy':'limited_recess'},
            {'id':'living','initial_rect':[12,3,18,9],'target_area':30,'area_range':[25,35]}],
        'FunctionalRelations':[{'from':'kitchen','to':'living','type':'adjacent'}]}


class ContractsTest(unittest.TestCase):
    def setUp(self):
        self.config=fixture()
        self.problem=build_problem(self.config)
        self.seeds={r.id:r.seed for r in self.problem.rooms}

    def test_fixed_geometry_includes_asymmetric_thickness(self):
        self.assertTrue(self.problem.fixed.contains(Point(9.8,5)))
        self.assertFalse(self.problem.fixed.contains(Point(10.2,5)))
        self.assertAlmostEqual(self.problem.fixed.area,18*.4)
        self.assertAlmostEqual(self.problem.free_space.area,400-18*.4)

    def test_input_is_not_mutated_and_rect_only_initializes_seed(self):
        before=copy.deepcopy(self.config)
        p=build_problem(self.config)
        self.assertEqual(self.config,before)
        self.assertEqual(p.rooms[1].seed,(15,6))
        self.assertEqual(p.rooms[0].shape_policy,'limited_recess')
        self.assertEqual(p.relations[0].kind,'adjacent')

    def test_invalid_seeds_fail_without_silent_relocation(self):
        for point in ([9.8,5],[0,5],[21,5],[float('nan'),5],self.seeds['living']):
            with self.subTest(point=point),self.assertRaises(ValueError):
                self.problem.validate_seeds(dict(self.seeds,kitchen=point))

    def test_bad_graph_and_insufficient_area_are_rejected(self):
        self.config['FunctionalRelations'][0]['to']='missing'
        with self.assertRaises(ValueError): build_problem(self.config)
        self.config=fixture()
        self.config['TargetSpaces'][0].update(target_area=450,area_range=[440,460])
        with self.assertRaises(ValueError): build_problem(self.config)

    def test_periodic_only_at_completed_250_episodes(self):
        schedule=PreciseSchedule()
        for episode in (0,1,20,249,251,499):
            self.assertFalse(schedule.due(SeedSnapshot.capture(self.problem,self.seeds,episode)))
        for episode in (250,500,750):
            self.assertTrue(schedule.due(SeedSnapshot.capture(self.problem,self.seeds,episode)))
        self.assertFalse(schedule.due(SeedSnapshot.capture(self.problem,self.seeds,250,step=1)))

    def test_finish_stop_dedup_and_recovery(self):
        schedule=PreciseSchedule()
        snapshot=SeedSnapshot.capture(self.problem,self.seeds,250)
        self.assertTrue(schedule.due(snapshot))
        self.assertTrue(schedule.due(snapshot,'stop_requested'))
        schedule.mark_completed(snapshot)
        restored=PreciseSchedule(yaml.safe_load(yaml.safe_dump(schedule.to_dict()))['completed_keys'])
        self.assertFalse(restored.due(snapshot,'finished'))
        self.assertFalse(restored.due(snapshot,'stop_requested'))
        later=SeedSnapshot.capture(self.problem,self.seeds,251)
        self.assertTrue(restored.due(later,'finished'))
        self.assertTrue(restored.due(later,'interrupted'))

    def test_snapshot_copies_seeds_and_serializes(self):
        seeds={k:list(v) for k,v in self.seeds.items()}
        snap=SeedSnapshot.capture(self.problem,seeds,20)
        before=snap.key
        seeds['kitchen'][0]=6
        self.assertEqual(snap.key,before)
        self.assertEqual(yaml.safe_load(yaml.safe_dump(snap.to_dict())),snap.to_dict())


if __name__=='__main__':
    unittest.main()
