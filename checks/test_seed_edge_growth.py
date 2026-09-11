import unittest
from dataclasses import replace
from shapely.geometry import Polygon, box, Point
from shapely.affinity import rotate, scale
from shapely.ops import unary_union

from core.envs.structure_geometry import fixed_polygon, normalize_structures
from core.seed_growth.contracts import RoomSpec, Problem, SeedSnapshot, build_problem, Relation
from core.seed_growth.edge_growth import exposed_edges, slant_candidates, split_candidate, refine_layout
from core.seed_growth.geometry import decode_layout
from core.seed_growth.shape_rules import clearance_width_ok, shape_metrics, check_shape


def wall(reverse=False):
    w=dict(id='w',type='shear_wall',start=[0,2.5],end=[3,4],left_thickness=.3,right_thickness=.1)
    if reverse:
        w.update(start=[3,4],end=[0,2.5],left_thickness=.1,right_thickness=.3)
    return fixed_polygon(normalize_structures([w])[0])


def north(p):
    return next(e for e in exposed_edges(p) if e.normal == (0,1))


def problem_for(base,fixed,room):
    boundary=box(-20,-20,20,20)
    return Problem((room,),(),boundary,fixed,boundary.difference(fixed),.5)


class EdgeGrowthTest(unittest.TestCase):
    def test_four_direction_a_b_use_real_wall_face(self):
        base=box(.5,0,4,2)
        for angle in (0,90,180,270):
            with self.subTest(angle=angle):
                p=rotate(base,angle,origin=(0,0)); obstacle=rotate(wall(),angle,origin=(0,0))
                candidates=[]
                for edge in exposed_edges(p):
                    candidates.extend((kind,q) for kind,q in slant_candidates(p,edge,obstacle)
                                      if q.intersection(obstacle).area < 1e-8)
                for kind in ('slant_A','slant_B'):
                    matches=[rotate(q,-angle,origin=(0,0)) for k,q in candidates if k==kind]
                    self.assertTrue(matches,(angle,kind))
                    # Right/lower wall face has normal offset (+.04472,-.08944).
                    intercept=2.5-.1*(1.25**.5)
                    expected_y=intercept+.5*4 if kind=='slant_A' else 4-.1/(1.25**.5)
                    self.assertTrue(any(abs(q.bounds[3]-expected_y)<1e-7 for q in matches))

    def test_wall_reversal_swaps_thickness_without_changing_growth(self):
        p=box(.5,0,4,2); edge=north(p)
        a=[(k,q) for k,q in slant_candidates(p,edge,wall()) if q.intersection(wall()).area<1e-8]
        b=[(k,q) for k,q in slant_candidates(p,edge,wall(True)) if q.intersection(wall(True)).area<1e-8]
        self.assertEqual(len(a),len(b))
        for kind,q in a:
            self.assertTrue(any(k==kind and q.symmetric_difference(r).area<1e-8 for k,r in b))

    def test_negative_wall_slope_mirrors_both_candidates(self):
        base=box(.5,0,4,2)
        mirrored=scale(base,xfact=-1,yfact=1,origin=(0,0))
        obstacle=scale(wall(),xfact=-1,yfact=1,origin=(0,0))
        original=[(k,q) for k,q in slant_candidates(base,north(base),wall()) if q.intersection(wall()).area<1e-8]
        reversed_slope=[(k,q) for k,q in slant_candidates(mirrored,north(mirrored),obstacle) if q.intersection(obstacle).area<1e-8]
        for kind,q in original:
            expected=scale(q,xfact=-1,yfact=1,origin=(0,0))
            self.assertTrue(any(k==kind and expected.symmetric_difference(p).area<1e-8 for k,p in reversed_slope))

    def test_b_requires_endpoint_beyond_finite_wall(self):
        p=box(.5,0,2,2)
        kinds=[k for k,q in slant_candidates(p,north(p),wall()) if q.intersection(wall()).area<1e-8]
        self.assertIn('slant_A',kinds)
        self.assertNotIn('slant_B',kinds)

    def test_b_is_selected_when_a_extension_is_outside(self):
        base=box(.5,0,4,2); room=RoomSpec('r',(1,1),12.4,(11,13),(1,3))
        p=problem_for(base,wall(),room); boundary=box(.5,0,4,4)
        p=replace(p,boundary=boundary,free_space=boundary.difference(wall()))
        result,records=refine_layout(p,{'r':room.seed},{'r':base})
        self.assertEqual(records[0]['accepted']['r'],'slant_B')
        self.assertEqual(result['r'].difference(boundary).area,0)
        self.assertLess(result['r'].intersection(wall()).area,1e-8)

    def test_a_extension_is_rejected_if_it_exits_boundary(self):
        base=box(.5,0,4,2); room=RoomSpec('r',(1,1),14,(6,18),(1,5))
        p=problem_for(base,wall(),room)
        p=replace(p,boundary=box(0,0,4,4),free_space=box(0,0,4,4).difference(wall()))
        polygons,records=refine_layout(p,{'r':room.seed},{'r':base})
        self.assertEqual(polygons['r'].difference(p.boundary).area,0)
        self.assertLess(polygons['r'].intersection(p.fixed).area,1e-8)

    def test_single_column_requires_recess_policy(self):
        base=box(0,0,10,4); fixed=box(4,4,6,5)
        room=RoomSpec('r',(5,2),48,(40,50),(1,4))
        self.assertIsNone(split_candidate(base,north(base),fixed,1,room))
        room=replace(room,shape_policy='limited_recess')
        result=split_candidate(base,north(base),fixed,1,room)
        self.assertAlmostEqual(result.area,48)
        self.assertEqual(len(result.interiors),0)
        self.assertEqual(result.intersection(fixed).area,0)
        errors,metrics=check_shape(result,room,fixed)
        self.assertEqual(errors,[])
        self.assertEqual(metrics['reflex_count'],2)

    def test_long_edge_two_columns_can_split_regular_room(self):
        base=box(0,0,10,4); fixed=unary_union([box(2,4,3,5),box(7,4,8,5)])
        room=RoomSpec('r',(5,2),44,(40,50),(1,4))
        result=split_candidate(base,north(base),fixed,.5,room)
        self.assertIsNotNone(result)
        self.assertEqual(check_shape(result,room,fixed)[0],[])

    def test_width_rejects_neck_and_thin_arm_not_regular_corners(self):
        lobes=[box(0,0,3,3),box(7,0,10,3)]
        narrow=unary_union(lobes+[box(3,1.2,7,1.8)])
        wide=unary_union(lobes+[box(3,.8,7,2.2)])
        arm=unary_union([box(0,0,3,3),box(3,1.3,6,1.7)])
        self.assertFalse(clearance_width_ok(narrow,1))
        self.assertFalse(clearance_width_ok(arm,1))
        self.assertTrue(clearance_width_ok(wide,1))
        self.assertTrue(clearance_width_ok(Polygon([(0,0),(4,0),(4,3),(0,2)]),1))
        self.assertTrue(clearance_width_ok(box(0,0,4,1),1))

    def test_metrics_are_scale_normalized(self):
        p=Polygon([(0,0),(6,0),(6,4),(4,4),(4,3),(2,3),(2,4),(0,4)])
        first,second=shape_metrics(p),shape_metrics(scale(p,xfact=10,yfact=10))
        for key in first:
            self.assertAlmostEqual(first[key],second[key])

    def test_deep_notch_is_rejected(self):
        p=Polygon([(0,0),(10,0),(10,10),(6,10),(6,2),(4,2),(4,10),(0,10)])
        room=RoomSpec('r',(2,2),84,(80,90),(1,3),'limited_recess')
        self.assertTrue(check_shape(p,room,Polygon())[0])

    def test_joint_refinement_respects_target_and_unchanged_seed(self):
        base=box(0,0,10,4); fixed=box(4,4,6,5)
        room=RoomSpec('r',(5,2),45,(40,50),(1,4),'limited_recess')
        problem=problem_for(base,fixed,room)
        boundary=box(0,0,10,5)
        problem=replace(problem,boundary=boundary,free_space=boundary.difference(fixed))
        result,records=refine_layout(problem,{'r':room.seed},{'r':base})
        self.assertAlmostEqual(result['r'].area,45,places=6)
        self.assertTrue(result['r'].covers(Point(room.seed)))
        self.assertTrue(records[0]['accepted'])
        self.assertEqual(result['r'].intersection(fixed).area,0)
        self.assertFalse(result['r'].equals(result['r'].envelope))
        self.assertIn('split',str(records[0]['accepted']))

    def test_full_decoder_reaches_area_with_slanted_face(self):
        room=RoomSpec('r',(1,1),10.5,(10,11),(1,3),min_width=2)
        problem=problem_for(None,wall(),room)
        boundary=box(.5,0,4,5)
        problem=replace(problem,boundary=boundary,free_space=boundary.difference(wall()))
        snapshot=SeedSnapshot.capture(problem,{'r':room.seed},250)
        before=decode_layout(problem,snapshot,refine=False)
        after=decode_layout(problem,snapshot)
        self.assertEqual(before['status'],'unresolved')
        self.assertEqual(after['status'],'valid',after['validation'])
        self.assertAlmostEqual(after['polygons']['r'].area,10.5,places=6)
        self.assertFalse(after['polygons']['r'].equals(after['polygons']['r'].envelope))
        self.assertTrue(any('slant_' in str(r['accepted']) for r in after['refinement']))

    def test_joint_conflicting_claims_are_both_rejected(self):
        a=RoomSpec('a',(2,2),24,(20,26),(1,4))
        b=RoomSpec('b',(8,2),24,(20,26),(1,4))
        boundary=box(0,0,10,4)
        problem=Problem((a,b),(),boundary,Polygon(),boundary,.5)
        initial=dict(a=box(0,0,4,4),b=box(6,0,10,4))
        seeds=dict(a=a.seed,b=b.seed)
        result,records=refine_layout(problem,seeds,initial)
        self.assertEqual(records[0]['conflict_rejected'],['a','b'])
        for key in initial:
            self.assertTrue(result[key].equals(initial[key]))
        reverse,other=refine_layout(replace(problem,rooms=(b,a)),seeds,dict(reversed(list(initial.items()))))
        self.assertEqual(records,other)
        self.assertEqual(result['a'].intersection(result['b']).area,0)

    def test_yaml_shape_limits_are_validated_without_mutation(self):
        from checks.test_seed_growth_contracts import fixture
        import copy
        import yaml
        c=fixture(); c['TargetSpaces'][0]['shape_limits']={'max_reflex':2,'allow_long_edge_split':False}
        before=copy.deepcopy(c)
        problem=build_problem(yaml.safe_load(yaml.safe_dump(c)))
        self.assertEqual(problem.rooms[0].shape_limits.max_reflex,2)
        self.assertFalse(problem.rooms[0].shape_limits.allow_long_edge_split)
        self.assertEqual(c,before)
        for invalid in ({'max_reflex':2.5},{'min_width_ratio':float('nan')},{'typo':1}):
            c['TargetSpaces'][0]['shape_limits']=invalid
            with self.assertRaises(ValueError): build_problem(c)

    def test_growth_cannot_destroy_proven_separation(self):
        a=RoomSpec('a',(2,2),18,(16,20),(1,4))
        b=RoomSpec('b',(8,2),16,(16,20),(1,4))
        boundary=box(0,0,10,4)
        relation=Relation('a','b','separate',min_distance=2)
        p=Problem((a,b),(relation,),boundary,Polygon(),boundary,.5)
        initial=dict(a=box(0,0,4,4),b=box(6,0,10,4))
        result,records=refine_layout(p,dict(a=a.seed,b=b.seed),initial)
        self.assertTrue(result['a'].equals(initial['a']))
        self.assertEqual(records[0]['validation_rollback'],['a/b:separate'])


if __name__ == '__main__':
    unittest.main()
