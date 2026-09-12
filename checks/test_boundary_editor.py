import copy
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from gui import boundary_editor as editor
from gui.structure_canvas import viewport
from gui.config_store import load_config


class BoundaryTest(unittest.TestCase):
    def test_canvas_drag_updates_saved_boundary_and_numeric_field(self):
        from types import SimpleNamespace
        import gui.adaptive_reuse_page as page
        state=copy.deepcopy(load_config('retrofit'))
        original=copy.deepcopy(state)
        drag=[False]
        def canvas(**kwargs):
            objects=copy.deepcopy(kwargs['initial_drawing']['objects'])
            if drag[0]:
                handles=[o for o in objects if o.get('type')=='circle']
                handles[1]['left']+=10
                drag[0]=False
            return SimpleNamespace(json_data={'objects':objects})
        source="""
from gui.adaptive_reuse_page import _render_plan_constraint_editor
from gui.config_store import load_config
_render_plan_constraint_editor(load_config('retrofit'),'retrofit')
"""
        loader=lambda _:copy.deepcopy(state)
        with patch('gui.config_store.load_config',loader),patch.object(editor,'load_config',loader),\
             patch.object(editor,'save_config',lambda c,i:state.update(copy.deepcopy(c))),patch.object(page,'st_canvas',canvas),\
             patch.object(page,'load_config',loader),patch.object(page,'save_config',lambda c,i:state.update(copy.deepcopy(c))):
            app=AppTest.from_string(source,default_timeout=30).run()
            next(r for r in app.radio if r.label=='画布操作').set_value('调整建筑边界').run()
            drag[0]=True
            app.run()
            self.assertFalse(app.exception)
            self.assertGreater(state['ExistingBuilding']['boundary'][1][0],original['ExistingBuilding']['boundary'][1][0])
            table=next(d.value for d in app.dataframe if 'vertex' in d.value.columns)
            self.assertEqual(table[['x','y']].values.tolist(),state['ExistingBuilding']['boundary'])
            self.assertEqual(state['ExistingBuilding']['fixed_objects'],original['ExistingBuilding']['fixed_objects'])
            self.assertEqual(state['Training'],original['Training'])

    def test_drag_and_hydration_keep_world_coordinates(self):
        boundary=[[-4,-2],[17,-2],[17,12],[-4,12]]
        raw=editor.boundary_handles(boundary,900,600)
        rounded=copy.deepcopy(raw)
        for obj in rounded:
            for k in ('left','top'): obj[k]=round(obj[k],2)
        self.assertEqual(editor.parse_boundary_handles(rounded,boundary,900,600),boundary)
        scale,_,_=viewport(boundary,900,600)
        raw[1]['left']+=2*scale
        moved=editor.parse_boundary_handles(raw,boundary,900,600)
        self.assertAlmostEqual(moved[1][0],19)
        self.assertEqual(moved[0],boundary[0])
        with self.assertRaises(ValueError): editor.parse_boundary_handles([],boundary,900,600)

    def test_invalid_shapes(self):
        for points in ([[0,0],[1,1],[2,2]],[[0,0],[2,2],[0,2],[2,0]],[[0,0],[1,0],[float('nan'),1]]):
            with self.assertRaises(ValueError): editor.validate_boundary(points)

    def test_numeric_save_and_canvas_save_share_yaml_without_other_changes(self):
        state=copy.deepcopy(load_config('retrofit')); original=copy.deepcopy(state)
        source="""
from gui.boundary_editor import render_boundary_editor
from gui.config_store import load_config
render_boundary_editor(load_config('retrofit'),'retrofit')
"""
        loader=lambda _:copy.deepcopy(state)
        saver=lambda c,i:state.update(copy.deepcopy(c))
        with patch('gui.config_store.load_config',loader),patch.object(editor,'load_config',loader),patch.object(editor,'save_config',saver):
            app=AppTest.from_string(source).run()
            app.session_state['retrofit_boundary_table_0']=dict(edited_rows={1:{'x':32.},2:{'x':32.,'y':20.},3:{'y':20.}},added_rows=[],deleted_rows=[])
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(state['ExistingBuilding']['boundary'][2],[32,20])
            app.session_state['retrofit_boundary_table_1']=dict(edited_rows={0:{'x':0.,'y':0.},1:{'x':1.,'y':1.},2:{'x':2.,'y':2.}},added_rows=[],deleted_rows=[3])
            app.run()
            self.assertTrue(app.error)
            self.assertEqual(state['ExistingBuilding']['boundary'][2],[32,20])
            points=[[0,0],[33,0],[32,20],[0,20]]
            editor.save_boundary(copy.deepcopy(state),'retrofit',points)
            self.assertEqual(state['ExistingBuilding']['boundary'],points)
        expected=copy.deepcopy(original); expected['ExistingBuilding']['boundary']=points
        self.assertEqual(state,expected)


if __name__=='__main__': unittest.main()
