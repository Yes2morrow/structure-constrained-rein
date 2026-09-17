from pathlib import Path
import copy
import unittest
from unittest.mock import patch
import yaml
from streamlit.testing.v1 import AppTest
from gui.training_service import training_command


class FloorPartitionUITest(unittest.TestCase):
    def test_concurrent_reruns_share_one_preview_calculation(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        import time
        from core.floor_partition import run_residential_floor_partition, _preview_cached
        from checks.test_joint_partition import case
        c,_,r=case()
        c['FloorPartition']['joint_search']['enabled']=False
        barrier=Barrier(2)
        def solve(_):
            time.sleep(.1)
            return [(r,{})]
        def request():
            barrier.wait()
            return run_residential_floor_partition(c)
        _preview_cached.cache_clear()
        with patch('core.floor_partition.structured.candidate_partitions',side_effect=solve) as solver:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(request) for _ in range(2)]
                outputs=[future.result(timeout=10) for future in futures]
            self.assertEqual(solver.call_count,1)
            self.assertIsNot(outputs[0][1],outputs[1][1])
        _preview_cached.cache_clear()

    def test_actual_a1_preview_includes_valid_net_download(self):
        config=yaml.safe_load((Path(__file__).resolve().parents[1]/'config/a1/config.yaml').read_text(encoding='utf8'))
        source='''
from gui.config_store import load_config
from gui.adaptive_reuse_page import _render_floor_partition_section
_render_floor_partition_section(load_config('a1'),'a1_live')
'''
        with patch('gui.config_store.load_config',lambda _:copy.deepcopy(config)):
            app=AppTest.from_string(source,default_timeout=120).run()
            self.assertFalse(app.exception,app.exception)
            self.assertFalse(app.error,[e.value for e in app.error])
            self.assertTrue(any('预校验通过' in s.value for s in app.success))
            self.assertEqual(len(app.get('download_button')),2)

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

    def test_constraint_controls_roundtrip_and_validation(self):
        import json
        from core.floor_partition import build_floor_partition_problem
        from core.floor_partition.quality import settings
        from core.floor_partition.joint import options
        config=yaml.safe_load((Path(__file__).resolve().parents[1]/'config/a1/config.yaml').read_text(encoding='utf8'))
        source='''
import streamlit as st
from gui.config_store import load_config, save_config
from gui.partition_settings import render_partition_constraints
config=load_config('a1')
valid=render_partition_constraints(config['FloorPartition'],'ui_test')
if st.button('保存测试参数', disabled=not valid):
    save_config(config,'a1')
st.json(config)
'''
        saved=[]
        with patch('gui.config_store.load_config',lambda _:copy.deepcopy(config)), \
             patch('gui.config_store.save_config',lambda c,i:saved.append(yaml.safe_load(yaml.safe_dump(c)))):
            app=AppTest.from_string(source).run()
            edits={'门前净空宽度（m）':1.4,'门前净空深度（m）':1.3,
                   '门内净空深度（m）':1.1,'面积均衡免罚范围':.025,
                   '每户最多转角数':18,'初始划分搜索宽度':40,
                   '预览局部修改次数':0,'窄小附属区域面积比例上限':.15}
            for label,value in edits.items():
                next(n for n in app.number_input if n.label==label).set_value(value)
            app.checkbox[0].set_value(False)
            app.button[0].click().run()
            self.assertFalse(app.exception,app.exception)
            p=build_floor_partition_problem(saved[-1]); q=settings(p)
            self.assertEqual(q['door_clearance_width'],1.4)
            self.assertEqual(q['door_clearance_depth'],1.3)
            self.assertEqual(q['entrance_depth'],1.1)
            self.assertEqual(q['area_balance_deadband'],.025)
            self.assertEqual(q['max_corners'],18)
            self.assertEqual(q['beam_width'],40)
            self.assertEqual(q['max_unusable_ratio'],.15)
            self.assertFalse(options(p)['compact_lobby'])
            self.assertEqual(options(p)['preview_steps'],0)
            next(n for n in app.number_input if n.label=='面积均衡免罚范围').set_value(.5).run()
            self.assertTrue(app.error)
            self.assertTrue(app.button[0].disabled)

    def test_preview_download_uses_net_geometry_and_inner_door_face(self):
        from io import BytesIO
        from zipfile import ZipFile
        from shapely.geometry import Polygon, LineString
        from core.floor_partition import run_residential_floor_partition
        from core.floor_partition.walls import wall_geometry, unit_face_door
        from gui.partition_settings import net_unit_archive
        from checks.test_joint_partition import case
        config,_,_=case()
        config['Training']={}
        config['FloorPartition']['joint_search']['preview_steps']=0
        config['FloorPartition']['residential']['wall_thickness']=.4
        problem,result=run_residential_floor_partition(config)
        net=wall_geometry(problem,result.corridor,result.unit_polygons,result.doors)
        with ZipFile(BytesIO(net_unit_archive(config,result))) as archive:
            self.assertIn('summary.yaml',archive.namelist())
            for door in result.doors:
                data=yaml.safe_load(archive.read(f'{door.unit_id}/config.yaml'))
                boundary=Polygon(data['ExistingBuilding']['boundary'])
                # Fixture has no column holes, so its exported shell is net space.
                self.assertLess(boundary.symmetric_difference(net.units[door.unit_id]).area,1e-6)
                self.assertLess(LineString(data['ExistingBuilding']['door_positions']).hausdorff_distance(
                    LineString(unit_face_door(problem,result,door))),1e-6)
                self.assertEqual(data['Training']['training_stage'],'room_training')
                self.assertEqual(data['FloorPartitionResult']['partition_wall_role'],
                                 'generated_non_load_bearing_partition')


if __name__=='__main__': unittest.main()
