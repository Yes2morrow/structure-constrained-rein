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
            self.assertTrue(any('REINFORCE' in item.value for item in app.text))
            rate=next(item for item in app.number_input if item.label=='分户策略学习率')
            rate.set_value(.02).run()
            next(button for button in app.button if button.label=='保存参数配置').click().run()
            self.assertFalse(app.exception,app.exception)
            self.assertEqual(saved[-1]['FloorPartition']['rl']['learning_rate'],.02)
            self.assertTrue(training_command(saved[-1],'a1')[2].endswith('train_floor_partition.py'))


if __name__=='__main__': unittest.main()
