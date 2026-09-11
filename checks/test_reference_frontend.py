import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from gui import config_store
from gui import coordinate_editor as coords


class ReferenceFrontendTest(unittest.TestCase):
    def test_legacy_selection_does_not_switch_structural_project(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'config').mkdir()
            (root/'.selected_config_id').write_text('1')
            (root/'config/.selected_config_id').write_text('retrofit')
            with patch.object(config_store,'PROJECT_ROOT',root),patch.object(config_store,'SELECTED_CONFIG_FILE',root/'.selected_config_id'):
                self.assertEqual(config_store.load_persisted_config_id(),'retrofit')
                config_store.persist_selected_config_id('b')
                self.assertEqual((root/'.selected_config_id').read_text(),'b')

    def test_reference_canvas_json_without_custom_ids(self):
        groups=[dict(name='建筑边界',points=[[0,0],[20,0],[20,12],[0,12]],color='#2563eb',closed=True,min_points=3)]
        bounds=coords._view_bounds(groups)
        objects=coords._initial_objects(groups,bounds,670,490)
        self.assertIsNone(coords._parse_objects([],groups,bounds,670,490))
        for obj in objects:
            for key in ('pointIndex','groupName','role'): obj.pop(key,None)
        self.assertEqual(coords._parse_objects(objects,groups,bounds,670,490)['建筑边界'],groups[0]['points'])
        circle=next(o for o in objects if o['type']=='circle' and o.get('fill')=='#2563eb')
        circle['left']+=15
        moved=coords._parse_objects(objects,groups,bounds,670,490)['建筑边界']
        self.assertGreater(moved[0][0],0)
        self.assertEqual(moved[1:],groups[0]['points'][1:])



if __name__=='__main__': unittest.main()
