"""Physical wall accounting and consistent alignment, independent of training."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import yaml
from shapely.geometry import box, LineString, Polygon, Point
from shapely.affinity import rotate
from shapely.ops import unary_union

from checks.test_joint_partition import case
from core.floor_partition.contracts import build_floor_partition_problem
from core.floor_partition.walls import wall_geometry, update_net_targets, structural_reference_axes
from core.floor_partition.quality import validate_partition, structural_axes
from core.floor_partition.export import export_unit_configs
from core.floor_partition.lobby import lobby_candidates, lobby_reassignments


class PartitionWallTests(unittest.TestCase):
    def test_subnanometre_edge_noise_does_not_erase_wall_or_inner_door(self):
        from core.floor_partition.contracts import Door
        from core.floor_partition.walls import unit_face_door
        _,p,r=case()
        units={'unit_01':box(0,0,5+1e-14,8),'unit_02':box(7,0,12,8)}
        door=Door('unit_01',((5.,2.),(5.,2.9)),'vertical',0.)
        r=replace(r,corridor=box(5,0,7,8),unit_polygons=units,doors=(door,))
        net=wall_geometry(p,r.corridor,units)
        self.assertAlmostEqual(net.units['unit_01'].bounds[2],4.9)
        face=LineString(unit_face_door(p,r,door))
        self.assertAlmostEqual(face.bounds[0],4.9)
        self.assertLess(face.difference(net.units['unit_01'].boundary.buffer(1e-7)).length,1e-7)

    def test_default_half_thickness_and_no_exterior_offset(self):
        _,p,r=case()
        self.assertEqual(p.profile.wall_thickness,.2)
        g=wall_geometry(p,r.corridor,r.unit_polygons,r.doors)
        self.assertAlmostEqual(g.units['unit_01'].intersection(LineString([(0,2),(12,2)])).length,5.9)
        self.assertEqual(g.units['unit_01'].bounds[0],0.)
        self.assertAlmostEqual(g.units['unit_02'].bounds[0],6.1)
        self.assertLess(g.reservation.intersection(p.fixed_union).area,1e-8)

    def test_area_conservation_and_thresholds(self):
        _,p,r=case()
        r=update_net_targets(p,r)
        g=wall_geometry(p,r.corridor,r.unit_polygons,r.doors)
        self.assertAlmostEqual(sum(v.area for v in g.units.values())+g.corridor.area+g.solid.area+g.thresholds.area,
                               p.boundary.difference(p.fixed_union).area)
        self.assertAlmostEqual(g.thresholds.area,2*.9*.2,places=6)
        self.assertAlmostEqual(g.solid.intersection(g.thresholds).area,0.)
        self.assertTrue(validate_partition(p,r)['valid'])

    def test_wall_turn_has_a_closed_mitre(self):
        _,p,_=case()
        a=box(0,0,6,5).union(box(0,5,3,8))
        b=box(0,0,12,8).difference(a)
        g=wall_geometry(p,Polygon(),{'a':a,'b':b})
        self.assertTrue(g.reservation.covers(Point(6.05,5.05)))

    def test_zero_and_custom_thickness(self):
        _,p,r=case()
        zero=replace(p,profile=replace(p.profile,wall_thickness=0.))
        g=wall_geometry(zero,r.corridor,r.unit_polygons,r.doors)
        self.assertTrue(g.reservation.is_empty)
        self.assertTrue(g.corridor.equals(r.corridor))
        thick=replace(p,profile=replace(p.profile,wall_thickness=.4))
        h=wall_geometry(thick,r.corridor,r.unit_polygons,r.doors)
        self.assertAlmostEqual(h.units['unit_02'].bounds[0],6.2)
        self.assertLess(h.corridor.area,g.corridor.area)

    def test_invalid_thickness_rejected(self):
        c,_,_=case()
        for value in (-.1,float('nan'),float('inf')):
            c['FloorPartition']['residential']['wall_thickness']=value
            with self.assertRaisesRegex(ValueError,'wall_thickness'):
                build_floor_partition_problem(c)

    def test_clear_width_uses_finished_faces(self):
        _,p,r=case()
        # Territory depth 1.5 m cannot supply 1.5 m clear after a wall deduction.
        corridor=box(3,6.5,9,8)
        free=p.boundary.difference(p.fixed_union).difference(corridor)
        units={'unit_01':free.intersection(box(0,0,6,10)), 'unit_02':free.intersection(box(6,0,12,10))}
        bad=replace(r,corridor=corridor,unit_polygons=units,allocatable_space=free)
        self.assertIn('corridor_width',validate_partition(p,bad)['errors'])

    def test_export_entrance_is_on_inner_wall_face(self):
        c,p,r=case(); c['Training']={}
        r=update_net_targets(p,r)
        with tempfile.TemporaryDirectory() as td:
            for path in export_unit_configs(c,r,td):
                data=yaml.safe_load(path.read_text(encoding='utf8'))
                b=data['ExistingBuilding']; meta=data['FloorPartitionResult']
                face=LineString(b['door_positions'])
                center=LineString(meta['door_centerline_positions'])
                self.assertAlmostEqual(face.distance(center),.1)
                self.assertLess(face.difference(Polygon(b['boundary']).boundary.buffer(1e-6)).length,1e-6)

    def test_flush_faces_have_half_wall_offset(self):
        c,_,_=case()
        c['ExistingBuilding']['fixed_objects'].append({'id':'col','type':'column','rect':[5.7,1,6.3,1.8]})
        p=build_floor_partition_problem(c); axes=structural_reference_axes(p)
        self.assertAlmostEqual(axes['low_inner'][0][0]-.1,5.7)
        self.assertAlmostEqual(axes['high_inner'][0][0]+.1,6.3)
        self.assertAlmostEqual(axes['center'][0][0],6.)
        self.assertIn(5.8,structural_axes(p)[0])

    def test_generated_walls_are_not_structural_references(self):
        from core.floor_partition.walls import structural_center_axes
        c,p,r=case()
        before=p.fixed_union.wkb
        wall_geometry(p,r.corridor,r.unit_polygons,r.doors)
        self.assertEqual(before,p.fixed_union.wkb)
        self.assertEqual(structural_center_axes(p),[[],[]])
        c['ExistingBuilding']['fixed_objects'].append(
            {'id':'bearing','type':'load_bearing_wall','rect':[1,1,1.4,4]})
        pp=build_floor_partition_problem(c)
        self.assertAlmostEqual(structural_center_axes(pp)[0][0],1.2)

    def test_interior_agent_rejects_partition_wall_thickness(self):
        from core.envs.adaptive_reuse_env import AdaptiveReuseEnv
        c,p,r=case()
        c['Training']={}
        c['AdaptiveReuseEnvironment']={'grid_size':.25,'randomize_initial':False}
        for thickness in (.2,.4):
            c['FloorPartition']['residential']['wall_thickness']=thickness
            pp=build_floor_partition_problem(c)
            rr=update_net_targets(pp,r)
            with tempfile.TemporaryDirectory() as td:
                path=export_unit_configs(c,rr,td)[0]
                data=yaml.safe_load(path.read_text(encoding='utf8'))
                self.assertEqual(data['FloorPartitionResult']['partition_wall_role'],
                                 'generated_non_load_bearing_partition')
                self.assertTrue(all(o['id']=='core' for o in data['ExistingBuilding']['fixed_objects']))
                data['TargetSpaces']=[dict(id='room',name='room',initial_rect=[1,1,3,3],
                                          target_area=4,area_range=[2,8],aspect_range=[1,3])]
                env=AdaptiveReuseEnv(data);env.reset(seed=1)
                room=env.agent_spaces[0]
                inside=room.copy(x1=4.,y1=1.,x2=6-thickness/2,y2=3.)
                wall=room.copy(x1=4.,y1=1.,x2=6-thickness/4,y2=3.)
                self.assertTrue(env._is_hard_valid(inside,0))
                self.assertFalse(env._is_hard_valid(wall,0))
                self.assertTrue(r.unit_polygons['unit_01'].covers(wall.polygon()))

    def test_lobby_depths_include_intermediate_grid_steps(self):
        _,p,r=case()
        depths=sorted({8-g.bounds[1] for g in lobby_candidates(p,r.opening,'south')})
        self.assertGreater(len(depths),3)
        self.assertTrue(all(b-a<=p.profile.grid_size+1e-6 for a,b in zip(depths,depths[1:])))

    def test_collinear_contact_nodes_do_not_block_door(self):
        from core.floor_partition.structured import unit_doors
        from core.floor_partition.circulation import CirculationLayout
        _,p,r=case()
        dense={u:g.segmentize(.2) for u,g in r.unit_polygons.items()}
        doors=unit_doors(p,CirculationLayout(r.corridor,r.opening,r.opening_side),dense)
        self.assertEqual(len(doors),len(dense))

    def test_mixed_alignment_scores_below_one_reference(self):
        c,_,_=case()
        c['ExistingBuilding']['fixed_objects'] += [
            {'id':'col1','type':'column','rect':[5.7,.6,6.3,1.4]},
            {'id':'col2','type':'column','rect':[5.7,3.6,6.3,4.4]}]
        p=build_floor_partition_problem(c)
        _,_,r=case(); free=p.boundary.difference(p.fixed_union).difference(r.corridor)
        def metric(cut):
            left=free.intersection(cut)
            rr=update_net_targets(p,replace(r,allocatable_space=free,
                unit_polygons={'unit_01':left,'unit_02':free.difference(left)}))
            return validate_partition(p,rr)['consistent_reference_alignment_ratio']
        center=metric(box(0,0,6,10))
        flush=metric(box(0,0,5.8,10))
        mixed=metric(box(0,0,6,2.5).union(box(0,2.5,5.8,10)))
        self.assertAlmostEqual(center,1.)
        self.assertAlmostEqual(flush,1.)
        self.assertLess(mixed,min(center,flush))

    def test_lobby_reassignment_preserves_all_space(self):
        _,p,r=case()
        for lobby in lobby_candidates(p,r.opening,r.opening_side):
            for pieces in lobby_reassignments(r,lobby):
                self.assertLess(unary_union(pieces).symmetric_difference(p.boundary.difference(p.fixed_union)).area,1e-7)
                self.assertAlmostEqual(sum(g.area for g in pieces),unary_union(pieces).area)

    def test_lobby_generator_rotates_with_the_core(self):
        c,p,r=case()
        from core.envs.structure_geometry import fixed_polygon
        turned=copy.deepcopy(c)
        turned['ExistingBuilding']['boundary']=list(rotate(p.boundary,90,origin=(0,0)).exterior.coords)
        opening=rotate(r.opening,90,origin=(0,0))
        turned['ExistingBuilding']['door_positions']=list(opening.coords)
        for obj in turned['ExistingBuilding']['fixed_objects']:
            obj['rect']=list(rotate(fixed_polygon(obj),90,origin=(0,0)).bounds)
        turned['FloorPartition']['residential']['opening_side']='east'
        pp=build_floor_partition_problem(turned)
        original=lobby_candidates(p,r.opening,'south')
        rotated=lobby_candidates(pp,opening,'east')
        self.assertTrue(original)
        self.assertEqual(len(original),len(rotated))
        for a,b in zip(original,rotated):
            self.assertLess(rotate(a,90,origin=(0,0)).symmetric_difference(b).area,1e-7)


if __name__=='__main__': unittest.main()
