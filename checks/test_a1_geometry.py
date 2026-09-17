"""A1 photo-derived invariants; no learned layout or screenshot matching."""
from pathlib import Path
import unittest

import yaml
from shapely.affinity import scale
from shapely.ops import unary_union

from core.envs.structure_geometry import fixed_polygon
from core.floor_partition.contracts import build_floor_partition_problem


class A1GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load((Path(__file__).resolve().parents[1] /
                                    'config/a1/config.yaml').read_text(encoding='utf8'))
        cls.problem = build_floor_partition_problem(cls.config)

    def test_complete_column_grid_and_perimeter_faces(self):
        p = self.problem
        columns = [fixed_polygon(o) for o in p.fixed_objects if o['type'] == 'column']
        self.assertEqual(len(columns), 18)
        xs = sorted({round(g.centroid.x, 6) for g in columns})
        ys = sorted({round(g.centroid.y, 6) for g in columns})
        self.assertEqual((len(xs), len(ys)), (6, 3))
        self.assertAlmostEqual(ys[1]-ys[0], ys[2]-ys[1])
        for g in columns:
            self.assertLess(g.difference(p.boundary).area, 1e-8)
            if round(g.centroid.x, 6) in (xs[0], xs[-1]) or round(g.centroid.y, 6) == ys[0]:
                self.assertGreater(g.boundary.intersection(p.boundary.boundary).length, .5)
        top = sorted((g for g in columns if round(g.centroid.y, 6) == ys[-1]), key=lambda g:g.centroid.x)
        for g in top:
            self.assertAlmostEqual(g.bounds[3], 10.)
        # Core-adjacent columns remain explicit and protrude into the occupied floor.
        for g in top[2:4]:
            self.assertGreater(g.difference(p.traffic_core).area, .4)

    def test_scale_symmetry_and_uniform_sections(self):
        p = self.problem
        measured = self.config['SurveyReference']['metres_per_pixel'] * (894-170)
        self.assertLess(abs(p.boundary.bounds[2]-measured)/measured, .03)
        axis = p.boundary.centroid.x
        for g in (p.boundary, p.traffic_core, unary_union([
                fixed_polygon(o) for o in p.fixed_objects if o['type']=='column'])):
            self.assertLess(g.symmetric_difference(scale(g,xfact=-1,yfact=1,origin=(axis,0))).area,1e-7)
        for o in p.fixed_objects:
            if o['type']=='column': self.assertEqual(o['size'], [.6,.8])

    def test_lobby_finished_faces_are_flush_with_columns(self):
        from dataclasses import replace
        from shapely.geometry import LineString
        from core.floor_partition.lobby import lobby_candidates
        from core.floor_partition.walls import wall_geometry
        for thickness in (.2,.4):
            p=replace(self.problem,profile=replace(self.problem.profile,wall_thickness=thickness))
            lobby=lobby_candidates(p,p.entrance,'south')[-1]
            units={'shell':p.boundary.difference(p.fixed_union).difference(lobby)}
            net=wall_geometry(p,lobby,units).corridor
            cut=LineString([(0,8),(28,8)])
            # Column inner faces are x=10.3 and x=17.7 regardless of wall width.
            self.assertAlmostEqual(net.intersection(cut).bounds[0],10.3)
            self.assertAlmostEqual(net.intersection(cut).bounds[2],17.7)
            self.assertAlmostEqual(lobby.intersection(cut).bounds[0]+thickness/2,10.3)


if __name__ == '__main__':
    unittest.main()
