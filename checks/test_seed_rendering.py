import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import yaml
from core.seed_growth.rendering import render_result
from core.seed_growth.environment import SeedLayoutEnv
from core.seed_growth.training import model_schema, restore_checkpoint


class RenderingTest(unittest.TestCase):
    def test_export_stays_in_run_and_advances_with_latest_snapshot(self):
        from checks.test_seed_proxy import config
        from core.seed_growth.artifacts import RunArtifacts
        c=config(); c['FunctionalRelations']=[]
        env=SeedLayoutEnv(c)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            artifacts=RunArtifacts(root,c,env.problem,env.proxy,export_root=root)
            first=env.snapshot(250)
            artifacts.emit(first,'periodic')
            self.assertTrue((root/'layout.png').exists())
            second=env.snapshot(500)
            artifacts.emit(second,'periodic')
            self.assertEqual(json.loads((root/'result.json').read_text(encoding='utf-8'))['snapshot_key'],second.key)
            artifacts.emit(first,'interrupted')
            self.assertEqual(json.loads((root/'result.json').read_text(encoding='utf-8'))['snapshot_key'],second.key)
            self.assertFalse((root/'seed_growth').exists())

    def test_saved_geometry_export_without_decode_or_mutation(self):
        root=Path(__file__).resolve().parents[1]/'results2/seed_experimental/cp6_adjacency_baseline_20260911'
        pointer=json.loads((root/'latest_precise.json').read_text(encoding='utf-8'))
        result=json.loads((root/pointer['path']).read_text(encoding='utf-8'))
        config=yaml.safe_load((root/'config.yaml').read_text(encoding='utf-8'))
        before=copy.deepcopy(result)
        with tempfile.TemporaryDirectory() as folder, patch('core.seed_growth.geometry.decode_layout',side_effect=AssertionError('must not decode')):
            path=render_result(config,result,folder)
            self.assertEqual(path.read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
            self.assertTrue(path.with_suffix('.svg').exists())
            self.assertEqual(json.loads((Path(folder)/'result.json').read_text(encoding='utf-8')),before)
        self.assertEqual(result,before)

    def test_v2_model_rejected_before_loading_weights(self):
        from checks.test_seed_proxy import config
        c=config(); c['FunctionalRelations']=[]
        env=SeedLayoutEnv(c)
        schema=model_schema(env); schema['mode']='seed_proxy_v2'
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'checkpoint').mkdir()
            (root/'latest_checkpoint.json').write_text(json.dumps({'path':'checkpoint'}))
            (root/'checkpoint/checkpoint.json').write_text(json.dumps({'schema':schema}))
            with self.assertRaisesRegex(ValueError,'不兼容'):
                restore_checkpoint(root,env,None)


if __name__=='__main__': unittest.main()
