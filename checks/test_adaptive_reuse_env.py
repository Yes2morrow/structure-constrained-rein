"""既有建筑适应性转换环境回归测试。"""

from pathlib import Path
import unittest

import numpy as np

from core.envs import make_adaptive_reuse_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "retrofit" / "config.yaml"


class AdaptiveReuseEnvTest(unittest.TestCase):
    def setUp(self):
        self.env, _ = make_adaptive_reuse_env(CONFIG_PATH)

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


if __name__ == "__main__":
    unittest.main()
