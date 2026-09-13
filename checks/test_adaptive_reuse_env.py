"""既有建筑适应性转换环境回归测试。"""

from pathlib import Path
import unittest

import numpy as np

from core.envs import AdaptiveReuseEnv, summarize_target_area_budget, redistribute_target_max_areas
from gui.config_store import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "retrofit" / "config.yaml"


class AdaptiveReuseEnvTest(unittest.TestCase):
    def setUp(self):
        import yaml
        config=yaml.safe_load((PROJECT_ROOT/'checks/fixtures/retrofit.yaml').read_text(encoding='utf-8'))
        # Stable smoke fixture: user-drawn obstacles may intentionally overlap a
        # starting room while the environment is being edited.
        config['ExistingBuilding']['fixed_objects']=[dict(id='fixture_column',type='column',rect=[.1,.1,.5,.5])]
        self.env=AdaptiveReuseEnv(config)

    def test_reset_exposes_multi_agent_state_and_grid(self):
        state, info = self.env.reset(seed=42)
        self.assertEqual(state.shape, self.env.observation_space.shape)
        self.assertEqual(state.shape[0], len(self.env.space_specs))
        self.assertTrue(np.isfinite(state).all())
        self.assertEqual(info["grid_matrix"].ndim, 2)
        self.assertTrue(all(4 in actions for actions in info["allow_actions"]))
        self.assertEqual(info["metrics"]["hard_conflicts"], 0.0)

    def test_stay_preserves_existing_layout(self):
        self.env.randomize_initial = False
        state, _ = self.env.reset(seed=1)
        next_state, rewards, done, info = self.env.step([4] * self.env.num_agents)
        np.testing.assert_allclose(state, next_state)
        self.assertEqual(len(rewards), self.env.num_agents)
        self.assertFalse(done)
        self.assertEqual(info["metrics"]["original_reuse"], 1.0)
        self.assertEqual(info["metrics"]["hard_conflicts"], 0.0)

    def test_every_allowed_action_obeys_hard_constraints(self):
        self.env.randomize_initial = False
        self.env.reset(seed=2)
        for index, actions in enumerate(self.env._get_allow_actions()):
            for action in actions:
                candidate = self.env._candidate_for_action(self.env.agent_spaces[index], action)
                self.assertTrue(self.env._is_hard_valid(candidate, index))

    def test_forbidden_action_is_rejected_and_penalized(self):
        self.env.randomize_initial = False
        self.env.reset(seed=3)
        # activity 向西移动会越过其可用边缘；连续执行直至动作被边界屏蔽。
        while 2 in self.env._get_allow_actions()[0]:
            self.env.step([2] + [4] * (self.env.num_agents - 1))
        before = self.env.agent_spaces[0]
        _, rewards, _, info = self.env.step([2] + [4] * (self.env.num_agents - 1))
        after = self.env.agent_spaces[0]
        self.assertEqual(before, after)
        self.assertTrue(info["reward_components"][0]["invalid_action"] < 0)
        self.assertTrue(np.isfinite(rewards).all())

    def test_reset_auto_repairs_invalid_initial_rect(self):
        import yaml
        config = yaml.safe_load((PROJECT_ROOT / 'checks/fixtures/retrofit.yaml').read_text(encoding='utf-8'))
        config['AdaptiveReuseEnvironment']['randomize_initial'] = False
        config['ExistingBuilding']['fixed_objects'] = [
            dict(id='fixture_column', type='column', rect=[11.85, 11.85, 12.15, 12.15]),
        ]
        env = AdaptiveReuseEnv(config)

        _, info = env.reset(seed=4)

        self.assertEqual(info["metrics"]["hard_conflicts"], 0.0)
        self.assertTrue(info["initial_repairs"])
        self.assertEqual(info["initial_repairs"][0]["space_id"], "living")
        self.assertTrue(env._is_hard_valid(env.agent_spaces[0], 0))

    def test_area_budget_only_subtracts_space_like_constraints(self):
        import yaml
        config = yaml.safe_load((PROJECT_ROOT / 'checks/fixtures/retrofit.yaml').read_text(encoding='utf-8'))
        config['ExistingBuilding']['fixed_objects'] = [
            dict(id='traffic_1', type='traffic_core', rect=[0.0, 0.0, 5.0, 4.0]),
            dict(id='cir_1', type='retained_circulation', rect=[10.0, 0.0, 12.0, 3.0]),
            dict(id='column_1', type='column', rect=[20.0, 1.0, 20.5, 1.5]),
            dict(id='wall_1', type='load_bearing_wall', rect=[25.0, 1.0, 29.0, 1.4]),
        ]

        budget = summarize_target_area_budget(config)

        self.assertAlmostEqual(budget["boundary_area"], 540.0)
        self.assertAlmostEqual(budget["constraint_area"], 26.0)
        self.assertAlmostEqual(budget["allocatable_area"], 514.0)

    def test_auto_redistribute_max_area_matches_allocatable_area(self):
        import yaml
        config = yaml.safe_load((PROJECT_ROOT / 'checks/fixtures/retrofit.yaml').read_text(encoding='utf-8'))
        config['ExistingBuilding']['fixed_objects'] = [
            dict(id='traffic_1', type='traffic_core', rect=[0.0, 0.0, 6.0, 5.0]),
        ]
        updated = redistribute_target_max_areas(config)
        budget = summarize_target_area_budget({**config, 'TargetSpaces': updated})

        self.assertAlmostEqual(budget["allocatable_area"], budget["max_area_sum"])
        for item in updated:
            self.assertGreaterEqual(item["area_range"][1], item["target_area"])
            self.assertGreaterEqual(item["area_range"][1], item["area_range"][0])

    def test_reset_auto_repairs_large_translation_around_traffic_core(self):
        import yaml
        config = yaml.safe_load((PROJECT_ROOT / 'checks/fixtures/retrofit.yaml').read_text(encoding='utf-8'))
        config['AdaptiveReuseEnvironment']['randomize_initial'] = False
        config['ExistingBuilding']['fixed_objects'] = [
            dict(id='traffic_1', type='traffic_core', rect=[20.0, 0.0, 30.0, 18.0]),
        ]
        config['TargetSpaces'][0]['initial_rect'] = [8.5, 12.5, 21.584858490566038, 17.09896226415094]
        config['TargetSpaces'][1]['initial_rect'] = [0.0, 1.0, 5.5, 9.5]
        config['TargetSpaces'][2]['initial_rect'] = [22.5, 12.5, 29.0, 18.0]
        config['TargetSpaces'][3]['initial_rect'] = [22.5, 1.0, 29.0, 5.0]
        config['TargetSpaces'][4]['initial_rect'] = [1.0, 12.5, 7.5, 18.0]
        config['TargetSpaces'][5]['initial_rect'] = [22.5, 6.5, 29.0, 10.5]
        env = AdaptiveReuseEnv(config)

        _, info = env.reset(seed=42)

        self.assertEqual(info["metrics"]["hard_conflicts"], 0.0)
        repaired_ids = {item["space_id"] for item in info["initial_repairs"]}
        self.assertTrue({"living", "second_bedroom", "kitchen", "dining"} <= repaired_ids)


if __name__ == "__main__":
    unittest.main()
