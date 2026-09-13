import copy
import io
import json
import tempfile
import os
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch,MagicMock
import torch
from shapely.geometry import box

from checks.test_seed_proxy import config
from core.seed_growth.contracts import build_problem,SeedSnapshot
from core.seed_growth.environment import SeedLayoutEnv
from core.seed_growth.artifacts import RunArtifacts,atomic_json,atomic_text,compare_proxy
from core.seed_growth.training import run_training,restore_checkpoint,stop_requested
from core.seed_growth.geometry import decode_layout
from common.training import build_mapppo_agent


def fixture():
    c=config()
    c['FunctionalRelations']=[{'from':'a','to':'b','type':'adjacent'}]
    c['Training'].update(lr=.0003,batch_size=4,seed=42)
    c['AlgorithmAdvanced']={'hidden_size':16,'ppo_epochs':1}
    return c


def valid_decoder(problem,snapshot):
    return dict(polygons={'a':box(0,3,5,6),'b':box(5,3,10,6)},decoder_version='test',snapshot=snapshot.to_dict())


class FakeAgent:
    def __init__(self,n):
        self.n=n; self.memory={'rewards':[]}
        self.reward_normalizer=self
    def normalize(self,r): return r
    def act(self,*args,**kwargs): return [4]*self.n,[0]*self.n,[0]*self.n
    def remember(self,*args): self.memory['rewards'].append(args[-2])
    def update(self): pass
    def save(self,path): Path(path).mkdir(parents=True,exist_ok=True)


class ArtifactTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.c=fixture(); self.p=build_problem(self.c)
        self.seeds={r.id:r.seed for r in self.p.rooms}
    def snap(self,episode=250,step=0,seeds=None):
        return SeedSnapshot.capture(self.p,seeds or self.seeds,episode,step)

    def test_periodic_finish_and_restart_reuse_one_artifact(self):
        with patch('core.seed_growth.artifacts.decode_layout',wraps=decode_layout):
            calls=[]
            def decoder(p,s): calls.append(s.key); return valid_decoder(p,s)
            a=RunArtifacts(self.root,self.c,self.p,decoder=decoder)
            self.assertIsNone(a.emit(self.snap(249),'periodic'))
            self.assertEqual(a.emit(self.snap(),'periodic')['status'],'valid')
            a.emit(self.snap(),'finished')
            restored=RunArtifacts(self.root,self.c,self.p,decoder=decoder)
            restored.emit(self.snap(),'interrupted')
            self.assertEqual(len(calls),1)
            self.assertEqual(len(list((self.root/'precise').glob('*/result.json'))),1)

    def test_failed_decode_can_retry_and_does_not_mark_done(self):
        a=RunArtifacts(self.root,self.c,self.p,decoder=lambda *x:(_ for _ in ()).throw(RuntimeError('injected')))
        result=a.emit(self.snap(),'finished')
        self.assertEqual(result['status'],'decode_failed')
        self.assertNotIn(self.snap().key,a.schedule.completed_keys)
        self.assertTrue((self.root/'latest_snapshot.json').exists())
        a.decoder=valid_decoder
        self.assertEqual(a.emit(self.snap(),'finished')['status'],'valid')

    def test_invalid_new_layout_cannot_overwrite_last_valid(self):
        a=RunArtifacts(self.root,self.c,self.p,decoder=valid_decoder)
        a.emit(self.snap(),'finished')
        before=(self.root/'last_valid.json').read_text()
        moved=dict(self.seeds,a=(8.25,5.25))
        self.assertEqual(a.emit(self.snap(251,1,moved),'stop_requested')['status'],'unresolved')
        self.assertEqual((self.root/'last_valid.json').read_text(),before)

    def test_commit_survives_pointer_write_failure(self):
        calls=[]
        def decoder(p,s): calls.append(1); return valid_decoder(p,s)
        a=RunArtifacts(self.root,self.c,self.p,decoder=decoder)
        original=atomic_json
        def fail(path,value):
            if Path(path).name=='latest_precise.json': raise OSError('injected disk failure')
            original(path,value)
        with patch('core.seed_growth.artifacts.atomic_json',side_effect=fail):
            self.assertEqual(a.emit(self.snap(),'finished')['status'],'decode_failed')
        a.emit(self.snap(),'finished')
        self.assertEqual(len(calls),1)
        self.assertTrue((self.root/'latest_precise.json').exists())

    def test_changed_node_config_rejected(self):
        RunArtifacts(self.root,self.c,self.p)
        changed=copy.deepcopy(self.c); changed['TargetSpaces'].reverse()
        with self.assertRaises(ValueError): RunArtifacts(self.root,changed,build_problem(changed))

    def test_replaying_old_result_does_not_move_latest_pointers_back(self):
        a=RunArtifacts(self.root,self.c,self.p,decoder=valid_decoder)
        a.emit(self.snap(250),'finished')
        a.emit(self.snap(251),'finished')
        a.emit(self.snap(250),'interrupted')
        for name in ('latest_precise.json','last_valid.json','latest_snapshot.json'):
            self.assertEqual(json.loads((self.root/name).read_text())['completed_episodes'],251)

    @unittest.skipUnless(os.name=='nt','Windows transient sharing behavior')
    def test_atomic_replace_retries_without_truncating_destination(self):
        path=self.root/'value.json'; atomic_text(path,'old')
        original=os.replace; attempts=[]
        def busy(source,destination):
            attempts.append(1)
            if len(attempts)==1:
                self.assertEqual(path.read_text(),'old')
                raise PermissionError('temporary reader')
            original(source,destination)
        with patch('core.seed_growth.artifacts.os.replace',side_effect=busy): atomic_text(path,'new')
        self.assertEqual(path.read_text(),'new')
        self.assertEqual(len(list(self.root.glob('*.tmp'))),0)

    def test_far_node_movement_gets_progress_without_fake_contact(self):
        c=fixture()
        for r in c['TargetSpaces']: r.update(target_area=2,area_range=[1,3])
        env=SeedLayoutEnv(c)
        before=env.estimate['relations'][0]
        graph=env.problem.relations
        env.step([3,4])
        after=env.estimate['relations'][0]
        self.assertEqual(before['estimated'],0); self.assertEqual(after['estimated'],0)
        self.assertGreater(after['proximity'],before['proximity'])
        self.assertFalse(after['verified']); self.assertTrue(env.estimate['uncertain'])
        self.assertEqual(env.problem.relations,graph)
        self.assertEqual((after['source'],after['target']),('a','b'))

    def test_all_relation_types_remain_bound_to_node_ids(self):
        for kind in ('adjacent','separate','connected','via_circulation'):
            c=fixture(); c['FunctionalRelations'][0]['type']=kind
            env=SeedLayoutEnv(c); original=env.problem.relations
            env.step([3,4])
            self.assertEqual(env.problem.relations,original)
            self.assertEqual((env.estimate['relations'][0]['source'],env.estimate['relations'][0]['target']),('a','b'))

    def test_boundary_grid_cell_does_not_authorize_boundary_seed(self):
        for seed,action in (([.5,5.25],2),([2.25,.5],1)):
            c=fixture(); c['TargetSpaces'][0]['seed']=seed
            env=SeedLayoutEnv(c); before=dict(env.seeds)
            self.assertNotIn(action,env._allowed()[0])
            env.step([action,4])
            self.assertEqual(env.seeds,before)
            env.snapshot(0,1)

    def test_stop_accepts_venv_launcher_pid_but_not_other_run(self):
        flag=self.root/'flag'
        flag.write_text(f'pid={os.getppid()}\n')
        self.assertTrue(stop_requested(flag))
        flag.write_text('pid=-999\n')
        self.assertFalse(stop_requested(flag))

    def test_actual_case_graph_fit_removes_the_previous_false_adjacency(self):
        baseline=Path(__file__).parent/'evidence/seed_geometry_baseline.json'
        c=json.loads(baseline.read_text(encoding='utf-8'))['config_snapshot']
        env=SeedLayoutEnv(c); result=decode_layout(env.problem,env.snapshot(0))
        report=compare_proxy(env.problem,env.estimate,result['validation'])
        dining=next(r for r in report['relations'] if r['source']=='dining' and r['target']=='kitchen')
        self.assertTrue(dining['precise_satisfied'])
        self.assertFalse(dining['false_positive'])
        # A failed precise check must still remain visibly different from a
        # successful proxy estimate, regardless of decoder improvements.
        failed=copy.deepcopy(result['validation'])
        edge=next(r for r in failed['relations'] if r['source']=='dining' and r['target']=='kitchen')
        edge['satisfied']=False; edge['shared_length']=0.
        comparison=compare_proxy(env.problem,env.estimate,failed)
        self.assertTrue(next(r for r in comparison['relations'] if r['source']=='dining' and r['target']=='kitchen')['false_positive'])


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.c=fixture()

    def test_250_boundary_finish_does_not_decode_each_episode(self):
        self.c['Training']['max_steps']=1
        env=SeedLayoutEnv(self.c); calls=[]
        def decoder(p,s): calls.append(s.completed_episodes); return valid_decoder(p,s)
        artifacts=RunArtifacts(self.root,self.c,env.problem,env.proxy,decoder)
        with redirect_stdout(io.StringIO()):
            result=run_training(env,FakeAgent(2),artifacts,250)
        self.assertEqual(calls,[250])
        self.assertEqual(result['snapshot'].completed_episodes,250)
        self.assertEqual(result['precise']['trigger'],'periodic')

    def test_resume_repairs_missed_250_episode_output(self):
        env=SeedLayoutEnv(self.c); calls=[]
        def decoder(p,s): calls.append(s.completed_episodes); return valid_decoder(p,s)
        artifacts=RunArtifacts(self.root,self.c,env.problem,env.proxy,decoder)
        with redirect_stdout(io.StringIO()):
            run_training(env,FakeAgent(2),artifacts,250,resume=(env.snapshot(250),0.))
        self.assertEqual(calls,[250])

    def test_mid_episode_stop_and_resume_match_uninterrupted_training(self):
        torch.manual_seed(123)
        env=SeedLayoutEnv(self.c); agent=build_mapppo_agent(env,self.c)
        artifacts=RunArtifacts(self.root/'split',self.c,env.problem,env.proxy,valid_decoder)
        flag=self.root/'stop.flag'
        def hook(s):
            if s.step==2: flag.write_text('')
        result=run_training(env,agent,artifacts,1,stop_check=lambda:'stop_requested' if stop_requested(flag) else None,on_step=hook)
        self.assertEqual((result['snapshot'].completed_episodes,result['snapshot'].step),(0,2))
        self.assertEqual(len(agent.memory['rewards']),2)
        new_env=SeedLayoutEnv(self.c); new_agent=build_mapppo_agent(new_env,self.c)
        resumed=restore_checkpoint(self.root/'split',new_env,new_agent)
        self.assertEqual(resumed[0].key,result['snapshot'].key)
        with redirect_stdout(io.StringIO()):
            final=run_training(new_env,new_agent,artifacts,1,resume=resumed)
        torch.manual_seed(123)
        whole_env=SeedLayoutEnv(self.c); whole_agent=build_mapppo_agent(whole_env,self.c)
        whole_artifacts=RunArtifacts(self.root/'whole',self.c,whole_env.problem,whole_env.proxy,valid_decoder)
        with redirect_stdout(io.StringIO()):
            whole=run_training(whole_env,whole_agent,whole_artifacts,1)
        self.assertEqual(final['snapshot'].key,whole['snapshot'].key)
        for a,b in zip(new_agent.actors,whole_agent.actors):
            for x,y in zip(a.parameters(),b.parameters()): torch.testing.assert_close(x,y)
        self.assertTrue(new_agent.actor_optimizers[0].state)

    def test_short_episodes_accumulate_for_network_update(self):
        self.c['Training']['max_steps']=2
        env=SeedLayoutEnv(self.c); agent=build_mapppo_agent(env,self.c)
        artifacts=RunArtifacts(self.root,self.c,env.problem,env.proxy,valid_decoder)
        with redirect_stdout(io.StringIO()): run_training(env,agent,artifacts,2)
        self.assertTrue(agent.actor_optimizers[0].state)
        self.assertEqual(len(agent.memory['rewards']),0)

    def test_exception_preserves_last_completed_step_not_partial_step(self):
        env=SeedLayoutEnv(self.c); agent=FakeAgent(2)
        artifacts=RunArtifacts(self.root,self.c,env.problem,env.proxy,valid_decoder)
        original=env.step; calls=[]
        def fail(actions):
            calls.append(1)
            result=original(actions)
            if len(calls)==2: raise RuntimeError('partial step')
            return result
        with patch.object(env,'step',side_effect=fail),self.assertRaises(RuntimeError):
            run_training(env,agent,artifacts,2)
        last=json.loads((self.root/'latest_snapshot.json').read_text())
        self.assertEqual(last['step'],1)
        pointer=json.loads((self.root/'latest_checkpoint.json').read_text())
        coherent=json.loads((self.root/pointer['path']/'checkpoint.json').read_text())
        self.assertEqual(coherent['snapshot']['step'],0)

    def test_old_rectangle_entry_honors_stop_at_episode_boundary(self):
        from argparse import Namespace
        from scripts import train_adaptive_reuse as old
        import yaml
        source=self.root/'config.yaml'; source.write_text(yaml.safe_dump(self.c))
        env=MagicMock(); env.num_agents=2
        env.calculate_metrics.return_value={k:1. for k in ('area_compliance','shape_compliance','adjacency_score',
            'original_reuse','intervention_ratio','hard_conflicts')}
        agent=MagicMock(); agent.learn.return_value={'episode_steps_reward':[1.,1.],'total_step':3}
        args=Namespace(config=source,config_id='test',episodes=3,max_steps=2,stop_file=self.root/'flag')
        with patch.object(old,'parse_args',return_value=args),patch.object(old,'PROJECT_ROOT',self.root),\
             patch.object(old,'make_adaptive_reuse_env',return_value=(env,self.c)),\
             patch.object(old,'build_mapppo_agent',return_value=agent),\
             patch.object(old,'stop_requested',side_effect=[False,True]),redirect_stdout(io.StringIO()):
            old.main()
        self.assertEqual(agent.learn.call_count,1)
        saved=list(self.root.glob('results2/adaptive_reuse/*/run_status.json'))
        self.assertEqual(json.loads(saved[0].read_text())['status'],'stop_requested')


if __name__=='__main__': unittest.main()
