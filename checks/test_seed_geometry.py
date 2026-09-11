import copy
import unittest
from shapely.geometry import box

from core.seed_growth.contracts import build_problem, SeedSnapshot
from core.seed_growth.geometry import decode_layout, _retreat_candidates, _score
from core.seed_growth.validation import validate_layout


def fixture():
    return dict(ExistingBuilding=dict(boundary=[[0,0],[10,0],[10,10],[0,10]], fixed_objects=[]),
                AdaptiveReuseEnvironment=dict(grid_size=.5),
                TargetSpaces=[dict(id='a',seed=[2,5],target_area=20,area_range=[18,22]),
                              dict(id='b',seed=[8,5],target_area=20,area_range=[18,22])])


def decode(config, **kwargs):
    p = build_problem(config)
    seeds = {r.id:r.seed for r in p.rooms}
    return p, seeds, decode_layout(p, SeedSnapshot.capture(p,seeds,250), **kwargs)


class GeometryTest(unittest.TestCase):
    def test_targets_seed_containment_no_overlap(self):
        p, seeds, result = decode(fixture())
        self.assertTrue(result['validation']['geometry_valid'], result['validation'])
        self.assertEqual(result['status'],'valid')
        for r in p.rooms:
            self.assertAlmostEqual(result['polygons'][r.id].area,r.target_area,places=6)
            self.assertTrue(result['polygons'][r.id].equals(result['polygons'][r.id].envelope))

    def test_input_order_does_not_allocate_priority(self):
        c = fixture()
        _, _, first = decode(c)
        c['TargetSpaces'].reverse()
        _, _, second = decode(c)
        for k in first['polygons']:
            self.assertTrue(first['polygons'][k].equals_exact(second['polygons'][k],1e-10))

    def test_real_asymmetric_wall_faces(self):
        c = fixture()
        c['ExistingBuilding']['fixed_objects'] = [dict(id='w',type='shear_wall',
            start=[5,0],end=[5,10],left_thickness=.37,right_thickness=.13)]
        p, seeds, result = decode(c)
        for poly in result['polygons'].values():
            self.assertEqual(poly.intersection(p.fixed).area,0)
        self.assertLessEqual(result['polygons']['a'].bounds[2],4.63+1e-9)
        self.assertGreaterEqual(result['polygons']['b'].bounds[0],5.13-1e-9)

    def test_whole_edge_stops_at_middle_column(self):
        c = fixture()
        c['TargetSpaces'] = [dict(id='a',seed=[5,2],target_area=70,area_range=[65,75])]
        c['ExistingBuilding']['fixed_objects'] = [dict(id='c',type='column',rect=[4,4,6,6])]
        p, _, result = decode(c)
        poly = result['polygons']['a']
        self.assertEqual(poly.intersection(p.fixed).area,0)
        self.assertTrue(poly.equals(poly.envelope))
        self.assertEqual(result['status'],'unresolved')
        self.assertIn('a:area_below_minimum',result['validation']['errors'])

    def test_concave_boundary_no_outside_growth(self):
        c = fixture()
        c['ExistingBuilding']['boundary'] = [[0,0],[10,0],[10,4],[4,4],[4,10],[0,10]]
        c['TargetSpaces'] = [dict(id='a',seed=[2,2],target_area=35,area_range=[30,40])]
        p, _, result = decode(c)
        self.assertEqual(result['polygons']['a'].difference(p.boundary).area,0)

    def test_validator_rejects_unsafe_external_candidate(self):
        p = build_problem(fixture()); seeds={r.id:r.seed for r in p.rooms}
        report = validate_layout(p,seeds,dict(a=box(-1,0,6,10),b=box(5,0,10,10)))
        self.assertFalse(report['geometry_valid'])
        for error in ('a:outside_boundary','a/b:room_overlap','a:area_above_maximum'):
            self.assertIn(error,report['errors'])

    def test_relation_contact_is_not_door_proof(self):
        c = fixture()
        c['FunctionalRelations'] = [{'from':'a','to':'b','type':'connected'}]
        p = build_problem(c); seeds={r.id:r.seed for r in p.rooms}
        report=validate_layout(p,seeds,dict(a=box(0,3,5,7),b=box(5,3,10,7)))
        self.assertTrue(report['geometry_valid'])
        self.assertEqual(report['relations'][0]['shared_length'],4)
        self.assertIsNone(report['relations'][0]['satisfied'])

    def test_corner_contact_is_not_adjacency(self):
        c=fixture(); c['FunctionalRelations']=[{'from':'a','to':'b','type':'adjacent'}]
        p=build_problem(c); seeds={r.id:r.seed for r in p.rooms}
        report=validate_layout(p,seeds,dict(a=box(0,0,5,5),b=box(5,5,10,10)))
        self.assertEqual(report['relations'][0]['shared_length'],0)
        self.assertFalse(report['relations'][0]['satisfied'])

    def test_blocked_neighbor_does_not_freeze_outward_growth(self):
        c=fixture()
        c['TargetSpaces'][0].update(target_area=48,area_range=[40,50],seed=[4,4])
        c['TargetSpaces'][1].update(target_area=48,area_range=[40,50],seed=[6,6])
        _,_,result=decode(c,retreat_passes=0)
        self.assertTrue(result['validation']['geometry_valid'],result['validation'])

    def test_numerical_contact_has_shared_length(self):
        c=fixture(); c['FunctionalRelations']=[{'from':'a','to':'b','type':'adjacent','min_shared_length':3}]
        p=build_problem(c); seeds={r.id:r.seed for r in p.rooms}
        report=validate_layout(p,seeds,dict(a=box(0,3,5,7),b=box(5+1e-10,3,10,7)))
        self.assertAlmostEqual(report['relations'][0]['shared_length'],4)
        self.assertTrue(report['relations'][0]['satisfied'])

    def test_snapshot_and_config_unchanged_and_budget_reported(self):
        c=fixture(); before=copy.deepcopy(c)
        _, _, result=decode(c,max_rounds=4)
        self.assertEqual(c,before)
        self.assertEqual(result['rounds'],4)
        self.assertEqual(result['status'],'unresolved')

    def test_explicit_width_is_hard_requirement(self):
        c=fixture(); c['TargetSpaces'][0]['min_width']=3
        p=build_problem(c); seeds={r.id:r.seed for r in p.rooms}
        report=validate_layout(p,seeds,dict(a=box(0,4,10,6),b=box(6,6,10,11)))
        self.assertIn('a:width_below_minimum',report['errors'])
        c['TargetSpaces'][0]['min_width']=-1
        with self.assertRaises(ValueError): build_problem(c)

    def test_retreat_preserves_donor_seed_and_minimum_area(self):
        p=build_problem(fixture()); seeds={r.id:r.seed for r in p.rooms}
        polygons=dict(a=box(0,3,5,7),b=box(5,3,10,7))
        candidates=list(_retreat_candidates(p,seeds,polygons,.5))
        self.assertTrue(candidates)
        for initial,caps in candidates:
            report=validate_layout(p,seeds,{k:box(*v) for k,v in initial.items()})
            self.assertTrue(report['geometry_valid'],report)
            self.assertEqual(len(caps),1)

    def test_retreat_never_commits_a_worse_layout(self):
        c=fixture()
        c['TargetSpaces'][0].update(target_area=48,area_range=[40,50],seed=[4,4])
        c['TargetSpaces'][1].update(target_area=48,area_range=[40,50],seed=[6,6])
        c['FunctionalRelations']=[{'from':'a','to':'b','type':'separate','min_distance':.5}]
        p,_,base=decode(c,retreat_passes=0)
        _,_,repaired=decode(c,retreat_passes=2)
        self.assertGreater(repaired['retreat_attempts'],0)
        self.assertLessEqual(_score(p,repaired['polygons'],repaired['validation']),
                             _score(p,base['polygons'],base['validation']))


if __name__ == '__main__':
    unittest.main()
