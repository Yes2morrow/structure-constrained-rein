"""住区布局环境的基础回归测试。"""

from pathlib import Path
import unittest

import numpy as np

from core.envs import make_residential_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "residential" / "config.yaml"


class ResidentialLayoutEnvTest(unittest.TestCase):
    def setUp(self):
        self.env, _ = make_residential_env(CONFIG_PATH)

    def test_reset_matches_declared_spaces(self):
        state, info = self.env.reset(seed=42)
        self.assertEqual(state.shape, self.env.observation_space.shape)
        self.assertEqual(len(info["allow_actions"]), self.env.num_agents)
        self.assertTrue(all(4 in actions for actions in info["allow_actions"]))
        self.assertTrue(np.isfinite(state).all())

    def test_joint_stay_keeps_valid_layout(self):
        state, _ = self.env.reset(seed=42)
        next_state, rewards, done, info = self.env.step([4] * self.env.num_agents)
        np.testing.assert_allclose(state, next_state)
        self.assertEqual(len(rewards), self.env.num_agents)
        self.assertFalse(done)
        self.assertEqual(info["metrics"]["overlap_ratio"], 0.0)
        self.assertEqual(info["metrics"]["spacing_compliance"], 1.0)

    def test_allowed_actions_preserve_buildable_boundary(self):
        self.env.reset(seed=7)
        for agent_index, allowed in enumerate(self.env._get_allow_actions()):
            original = self.env.buildings[agent_index]
            for action in allowed:
                candidate = self.env._candidate_for_action(original, action)
                self.assertTrue(self.env._is_geometrically_valid(candidate))


if __name__ == "__main__":
    unittest.main()

