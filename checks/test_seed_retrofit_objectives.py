import copy
import unittest
from unittest.mock import patch
import numpy as np

from core.seed_growth.environment import SeedLayoutEnv
from core.seed_growth.objectives import DEFAULT_WEIGHTS, precise_retrofit_metrics
from shapely.geometry import box


def config():
    return dict(ExistingBuilding=dict(boundary=[[0,0],[8,0],[8,4],[0,4]],
                    fixed_objects=[], original_spaces=[dict(id='a',rect=[0,0,4,4])]),
                AdaptiveReuseEnvironment=dict(grid_size=1), Training=dict(max_steps=3),
                TargetSpaces=[dict(id='a',initial_rect=[0,0,4,4],target_area=16)],
                RewardWeights={k:0. for k in DEFAULT_WEIGHTS})


class RetrofitObjectiveTest(unittest.TestCase):
    def test_precise_report_uses_original_geometry(self):
        report=precise_retrofit_metrics(config(),{'a':box(0,0,4,4)})
        self.assertFalse(report['estimated'])
        self.assertEqual(report['original_reuse'],1.)
        self.assertEqual(report['intervention_ratio'],0.)
        moved=precise_retrofit_metrics(config(),{'a':box(4,0,8,4)})
        self.assertEqual(moved['original_reuse'],0.)
        self.assertEqual(moved['intervention_ratio'],2.)

    def test_weights_control_training_and_do_not_decode(self):
        c=config()
        with patch('core.seed_growth.geometry.decode_layout',side_effect=AssertionError('expensive decoder')):
            env=SeedLayoutEnv(c)
            self.assertEqual(env.seeds['a'],(2.,2.))
            _,reward,_,_=env.step([4])
            self.assertEqual(reward,[0.])
            c['RewardWeights']['original_reuse']=2.
            weighted=SeedLayoutEnv(c)
            _,reward,_,_=weighted.step([4])
            self.assertAlmostEqual(reward[0],2*weighted.calculate_metrics()['original_reuse'])

    def test_original_rectangle_is_fixed_and_intervention_penalizes_motion(self):
        c=config(); c['RewardWeights']['intervention']=1.
        original=copy.deepcopy(c)
        env=SeedLayoutEnv(c)
        before=env.calculate_metrics()
        env.step([3]); _,reward,_,_=env.step([3])
        after=env.calculate_metrics()
        self.assertGreater(after['intervention_ratio'],before['intervention_ratio'])
        self.assertLess(after['original_reuse'],before['original_reuse'])
        self.assertAlmostEqual(reward[0],-after['intervention_ratio'])
        self.assertEqual(c,original)

    def test_fallback_reference_is_initial_rectangle_and_reset_is_stable(self):
        c=config(); c['ExistingBuilding']['original_spaces']=[]
        env=SeedLayoutEnv(c)
        self.assertEqual(env.objectives.references[0].bounds,(0.,0.,4.,4.))
        first=env.get_state().copy()
        env.step([3]); env.reset()
        np.testing.assert_array_equal(first,env.get_state())


if __name__=='__main__': unittest.main()
