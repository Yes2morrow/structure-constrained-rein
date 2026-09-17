from pathlib import Path
import copy
import unittest
from unittest.mock import patch
import yaml
from streamlit.testing.v1 import AppTest
from gui.training_service import training_command


class FloorPartitionUITest(unittest.TestCase):
    def test_actual_algorithm_controls_and_saved_learning_rate(self):
        config=yaml.safe_load((Path(__file__).resolve().parents[1]/'config/a1/config.yaml').read_text(encoding='utf-8'))
        source='''
from gui.config_store import load_config
from gui.adaptive_reuse_page import render_adaptive_reuse_config_page
render_adaptive_reuse_config_page(load_config('a1'),'a1',include_environment=False)
'''
        saved=[]
        with patch('gui.config_store.load_config',lambda _:copy.deepcopy(config)),\
             patch('gui.adaptive_reuse_page.save_config',lambda c,i:saved.append(copy.deepcopy(c)) or 'test-only'):
            app=AppTest.from_string(source,default_timeout=30).run()
            self.assertFalse(app.exception,app.exception)
            self.assertTrue(any('PPO' in item.value for item in app.text))
            rate=next(item for item in app.number_input if item.label=='分户策略学习率')
            rate.set_value(.02).run()
            next(button for button in app.button if button.label=='保存参数配置').click().run()
            self.assertFalse(app.exception,app.exception)
            self.assertEqual(saved[-1]['FloorPartition']['rl']['learning_rate'],.02)
            self.assertTrue(training_command(saved[-1],'a1')[2].endswith('train_floor_partition.py'))

    def test_wall_thickness_control_updates_config(self):
        import json
        config=yaml.safe_load((Path(__file__).resolve().parents[1]/'config/a1/config.yaml').read_text(encoding='utf-8'))
        config['FloorPartition']['enabled']=False
        config['Training']['training_stage']='room_training'
        source='''
import streamlit as st
from gui.config_store import load_config
from gui.adaptive_reuse_page import _render_floor_partition_section
config=load_config('a1')
_render_floor_partition_section(config,'wall_test')
st.json(config['FloorPartition']['residential'])
'''
        with patch('gui.config_store.load_config',lambda _:copy.deepcopy(config)):
            app=AppTest.from_string(source,default_timeout=30).run()
            thickness=next(item for item in app.number_input if item.label=='分户隔墙总厚度（m）')
            self.assertEqual(thickness.value,.2)
            thickness.set_value(.25).run()
            self.assertFalse(app.exception,app.exception)
            self.assertEqual(json.loads(app.json[-1].value)['wall_thickness'],.25)


if __name__=='__main__': unittest.main()
