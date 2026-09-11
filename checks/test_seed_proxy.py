import copy
import time
import unittest
import numpy as np
from core.seed_growth.contracts import build_problem
from core.seed_growth.proxy import LayoutProxy
from core.seed_growth.environment import SeedLayoutEnv


def config():
    return dict(ExistingBuilding=dict(boundary=[[0,0],[10,0],[10,10],[0,10]],fixed_objects=[]),
                AdaptiveReuseEnvironment=dict(grid_size=.5),Training=dict(max_steps=4),
                TargetSpaces=[dict(id='a',seed=[2.25,5.25],target_area=15),dict(id='b',seed=[7.25,5.25],target_area=15)],
                FunctionalRelations=[dict(from_='a')])


class ProxyTest(unittest.TestCase):
    def setUp(self):
        self.c=config(); self.c['FunctionalRelations']=[{'from':'a','to':'b','type':'adjacent'}]

    def test_thin_wall_blocks_distance_and_adjacency(self):
        self.c['ExistingBuilding']['fixed_objects']=[dict(id='wall',type='shear_wall',rect=[4.99,0,5.01,10])]
        p=build_problem(self.c); proxy=LayoutProxy(p)
        seeds={r.id:r.seed for r in p.rooms}
        result=proxy.evaluate(seeds)
        self.assertTrue(np.isinf(proxy.distances(proxy.cell(seeds['a']))[proxy.cell(seeds['b'])]))
        self.assertEqual(result['shared'][0,1],0)
        self.assertEqual(result['relations'][0]['estimated'],0)

    def test_narrow_gap_is_conservatively_closed(self):
        self.c['ExistingBuilding']['fixed_objects']=[dict(id='w1',type='fixed',rect=[4.9,0,5.1,4.9]),dict(id='w2',type='fixed',rect=[4.9,5.1,5.1,10])]
        p=build_problem(self.c); proxy=LayoutProxy(p)
        self.assertGreaterEqual(proxy.component_count,2)

    def test_deterministic_capacity_and_cache(self):
        p=build_problem(self.c); proxy=LayoutProxy(p); seeds={r.id:r.seed for r in p.rooms}
        a=proxy.evaluate(seeds); b=proxy.evaluate(seeds)
        self.assertIs(a,b)
        self.assertLessEqual(sum(a['areas']),sum(r.target_area for r in p.rooms))
        self.assertTrue(a['estimated'])
        self.assertFalse(a['owners'].flags.writeable)

    def test_same_cell_is_not_silently_relocated(self):
        p=build_problem(self.c); proxy=LayoutProxy(p)
        with self.assertRaises(ValueError): proxy.evaluate({'a':[2.1,5.1],'b':[2.2,5.2]})

    def test_connectivity_without_openings_is_unknown(self):
        self.c['FunctionalRelations'][0]['type']='connected'
        env=SeedLayoutEnv(self.c)
        self.assertIsNone(env.estimate['relations'][0]['estimated'])
        self.assertTrue(env.estimate['uncertain'])

    def test_environment_stay_timeout_and_mask(self):
        env=SeedLayoutEnv(self.c); state,info=env.reset()
        self.assertEqual(state.shape,env.observation_space.shape)
        for i in range(4):
            next_state,rewards,done,info=env.step([4,4])
            np.testing.assert_allclose(state,next_state)
        self.assertTrue(done); self.assertEqual(info['done_list'],[True,True])
        self.assertTrue(np.isfinite(rewards).all())

    def test_joint_conflict_rolls_back_both(self):
        self.c['TargetSpaces'][1]['seed']=[3.25,5.25]
        env=SeedLayoutEnv(self.c); before=copy.deepcopy(env.seeds)
        env.step([3,2])
        self.assertEqual(env.seeds,before)

    def test_timing_baseline(self):
        start=time.perf_counter(); env=SeedLayoutEnv(self.c); init=time.perf_counter()-start
        start=time.perf_counter()
        for _ in range(50): env.step([4,4])
        ms=(time.perf_counter()-start)*1000/50
        print(f'proxy benchmark 10x10m, 0.5m grid: init={init:.4f}s, cached full step={ms:.3f}ms')

    def test_all_graph_relations_survive_motion_and_edge_order(self):
        self.c['FunctionalRelations']=[
            {'from':'a','to':'b','type':'adjacent','min_shared_length':2},
            {'from':'a','to':'b','type':'connected','min_clear_width':.9},
            {'from':'b','to':'a','type':'via_circulation'},
            {'from':'b','to':'a','type':'separate','min_distance':3}]
        env=SeedLayoutEnv(self.c)
        graph=env.get_state()[:,11:18].copy()
        np.testing.assert_allclose(graph,[[1,1,1,1,.2,.09,.3]]*2)
        env.step([0,1])
        np.testing.assert_array_equal(env.get_state()[:,11:18],graph)
        self.c['FunctionalRelations'].reverse()
        other=SeedLayoutEnv(self.c)
        np.testing.assert_array_equal(other.get_state()[:,11:18],graph)
        self.assertEqual(env.observation_space.shape,(2,19))


if __name__=='__main__': unittest.main()
