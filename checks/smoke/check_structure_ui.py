"""Save precise parameters through Streamlit without touching project config."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from copy import deepcopy
from unittest.mock import patch
from types import SimpleNamespace
from streamlit.testing.v1 import AppTest
from gui.config_store import load_config
import gui.structure_editor as editor
import gui.adaptive_reuse_page as page
from core.envs.structure_geometry import column


def test_save():
    state = load_config('retrofit')
    before = deepcopy(state)
    source = """
from gui.structure_editor import render_structure_editor
from gui.config_store import load_config
render_structure_editor(load_config('retrofit'), 'retrofit')
"""
    loader = lambda _: deepcopy(state)
    with patch('gui.config_store.load_config', loader), patch.object(editor, 'load_config', loader), patch.object(editor, 'save_config', lambda c,i: state.update(deepcopy(c))):
        app = AppTest.from_string(source).run()
        app.session_state['retrofit_columns_0'] = dict(edited_rows={}, deleted_rows=[], added_rows=[dict(id='c1',cx=10.,cy=8.,width=.4,depth=.6)])
        app.session_state['retrofit_walls_0'] = dict(edited_rows={}, deleted_rows=[], added_rows=[dict(id='w1',type='shear_wall',sx=10.,sy=5.,ex=15.,ey=8.,left=.15,right=.1)])
        app.run()
        assert not app.exception
        assert not app.error
        objects = state['ExistingBuilding']['fixed_objects']
        assert objects[0]['size'] == [.4,.6]
        assert objects[1]['start'] == [10.,5.]
        assert objects[1]['left_thickness'] == .15
        assert objects[1]['right_thickness'] == .1
        assert state['Training'] == before['Training']
        app.run()
        assert not app.exception
        assert app.dataframe[0].value.iloc[0]['id'] == 'c1'
    print('PASS: form submission saves exact column/wall parameters; reload retains data; training unchanged')


def test_canvas_initialization():
    state=load_config('retrofit')
    state['ExistingBuilding']['fixed_objects']=[column('stable',5,5,.4,.6)]
    saves=[]
    hydrated=[False]
    def canvas(**kwargs):
        return SimpleNamespace(json_data=deepcopy(kwargs['initial_drawing']) if hydrated[0] else {'objects':[]})
    source="""
from gui.config_store import load_config
from gui.adaptive_reuse_page import _render_plan_constraint_editor
_render_plan_constraint_editor(load_config('retrofit'),'retrofit')
"""
    with patch('gui.config_store.load_config',lambda _:deepcopy(state)), patch.object(editor,'load_config',lambda _:deepcopy(state)), patch.object(editor,'save_config',lambda c,i:saves.append(c)), patch.object(page,'st_canvas',canvas):
        app=AppTest.from_string(source,default_timeout=30).run()
        assert not app.exception
        assert not saves, 'empty initialization frame must not erase YAML'
        hydrated[0]=True
        app.run()
        assert not app.exception
        assert not saves, 'loading existing geometry is not an edit'
    print('PASS: empty canvas initialization cannot erase persisted structures')


if __name__ == '__main__':
    test_save()
    test_canvas_initialization()
