"""Step-boundary seed training, durable checkpoints and offline output."""
import json
import os
from pathlib import Path
from uuid import uuid4
from .artifacts import atomic_json, atomic_text, read_snapshot, config_key


def stop_requested(path):
    path=Path(path)
    if not path.exists(): return False
    text=path.read_text(encoding='utf-8')
    pids=[line.split('=',1)[1] for line in text.splitlines() if line.startswith('pid=')]
    # Windows venv python.exe may be a launcher whose child runs Python code.
    # GUI/Popen tracks that launcher PID, so accept our direct parent as well.
    return not pids or pids[0] in {str(os.getpid()),str(os.getppid())}


def model_schema(env):
    return dict(mode='seed_proxy_v3',objective_version='graph_partition_v2',actions=5,room_ids=list(env.ids),graph_node_ids=list(env.proxy.ids),
                observation=list(env.observation_space.shape),max_steps=env.max_steps,
                config_key=config_key(env.config))


def save_checkpoint(root, env, agent, snapshot, episode_reward):
    import torch
    import numpy as np
    import random
    folder=Path(root)/'checkpoints'/f'{snapshot.completed_episodes}_{snapshot.step}_{uuid4().hex}'
    folder.mkdir(parents=True)
    agent.save(str(folder/'model'))
    # Preserve unconsumed rollout: short episodes must not discard the batch.
    torch.save(dict(memory=agent.memory,torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    numpy_rng=np.random.get_state(),python_rng=random.getstate()),folder/'runtime.pt')
    atomic_json(folder/'checkpoint.json',dict(schema=model_schema(env),snapshot=snapshot.to_dict(),episode_reward=episode_reward))
    atomic_json(Path(root)/'latest_checkpoint.json',dict(path=str(folder.relative_to(root))))
    return folder


def restore_checkpoint(root, env, agent):
    import torch
    import numpy as np
    import random
    root=Path(root).resolve()
    pointer=json.loads((root/'latest_checkpoint.json').read_text(encoding='utf-8'))
    folder=(root/pointer['path']).resolve()
    if root not in folder.parents: raise ValueError('续训检查点不在本次运行目录内')
    record=json.loads((folder/'checkpoint.json').read_text(encoding='utf-8'))
    if record['schema']!=model_schema(env):
        raise ValueError('模型/节点顺序/动作/配置/奖励版本不兼容；旧矩形、CP2 或固定实验奖励检查点不能续训共同目标版本，请开始新训练；已有成图仍可查看')
    snapshot=read_snapshot(env.problem,record['snapshot'])
    for i,actor in enumerate(agent.actors):
        actor.load_state_dict(torch.load(folder/'model'/f'actor_{i}.pth',map_location=agent.device,weights_only=True))
    agent.critic.load_state_dict(torch.load(folder/'model/critic.pth',map_location=agent.device,weights_only=True))
    # Locally generated optimizer and NumPy RNG state, not downloaded weights.
    agent.load_training_state(torch.load(folder/'model/training_state.pth',map_location=agent.device,weights_only=False))
    runtime=torch.load(folder/'runtime.pt',map_location='cpu',weights_only=False)
    agent.memory=runtime['memory']
    torch.set_rng_state(runtime['torch_rng'])
    if runtime['cuda_rng'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(runtime['cuda_rng'])
    np.random.set_state(runtime['numpy_rng']); random.setstate(runtime['python_rng'])
    env.seeds=dict(snapshot.seeds); env.total_steps=snapshot.step
    env.estimate=env.proxy.evaluate(env.seeds)
    return snapshot,float(record['episode_reward'])


def run_training(env, agent, artifacts, total_episodes, stop_check=lambda:None,
                 resume=None, snapshot_every=10, save_interval=20, on_step=None):
    """Signals become flags in the CLI; stop only after a complete update.

    Unexpected exceptions retain the previous coherent model checkpoint and
    attempt offline output from the last completed step. Pending rollout is
    retained across episode boundaries and checkpoints.
    """
    if total_episodes<1 or min(snapshot_every,save_interval)<1: raise ValueError('轮数与保存间隔必须为正')
    if resume is None:
        snapshot=env.snapshot(0); episode_reward=0.
        save_checkpoint(artifacts.root,env,agent,snapshot,episode_reward)
    else:
        snapshot,episode_reward=resume
    if snapshot.completed_episodes>total_episodes: raise ValueError('目标累计轮数小于已完成轮数')
    if resume is not None:
        metrics=artifacts.root/'estimated_metrics.jsonl'
        if metrics.exists():
            original=metrics.read_text(encoding='utf-8'); retained={}
            lines=original.splitlines()
            for i,line in enumerate(lines):
                try: record=json.loads(line)
                except json.JSONDecodeError:
                    if i==len(lines)-1: break
                    raise
                if record['episode']<=snapshot.completed_episodes:
                    retained[record['episode']]=line
            restored=''.join(retained[k]+'\n' for k in sorted(retained))
            if restored!=original:
                atomic_text(artifacts.root/'metrics_history'/f'{uuid4().hex}.jsonl',original)
                atomic_text(metrics,restored)
    artifacts.save_snapshot(snapshot,force=True)
    completed=snapshot.completed_episodes
    in_episode=snapshot.step>0
    state,info=env.get_state(),env._info()
    reason='finished'
    try:
        artifacts.emit(snapshot,'periodic')
        while completed<total_episodes:
            requested=stop_check()
            if requested:
                reason=requested; break
            if not in_episode:
                state,info=env.reset(seed=int(env.config.get('Training',{}).get('seed',42))+completed)
                episode_reward=0.; in_episode=True
            global_state=state.flatten()
            actions,log_probs,values=agent.act(state,global_state,training=True,info=info)
            next_state,rewards,done,info=env.step(actions)
            normalized=agent.reward_normalizer.normalize(rewards)
            agent.remember(state,global_state,actions,log_probs,values,normalized,done)
            agent.update()
            state=next_state
            episode_reward+=sum(rewards)
            if done:
                completed+=1; in_episode=False
                snapshot=env.snapshot(completed)
                with (artifacts.root/'estimated_metrics.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(episode=completed,reward=episode_reward,
                        **env.calculate_metrics(),snapshot=snapshot.to_dict()))+'\n')
                print(f'种子训练 {completed}/{total_episodes}，代理评价（未经精确证明）',flush=True)
                episode_reward=0.
            else:
                snapshot=env.snapshot(completed,env.total_steps)
            if done or env.total_steps%snapshot_every==0:
                artifacts.save_snapshot(snapshot)
            if done and (completed%save_interval==0 or completed%250==0):
                save_checkpoint(artifacts.root,env,agent,snapshot,episode_reward)
            if done:
                artifacts.emit(snapshot,'periodic')
            if on_step is not None: on_step(snapshot)
    except BaseException:
        artifacts.save_snapshot(snapshot)
        artifacts.emit(snapshot,'interrupted')
        atomic_json(artifacts.root/'run_status.json',dict(status='interrupted',snapshot=snapshot.to_dict(),
            note='异常时不保存可能处于更新中间的模型；模型续训用 latest_checkpoint，最新种子可离线补图'))
        raise
    artifacts.save_snapshot(snapshot)
    save_checkpoint(artifacts.root,env,agent,snapshot,episode_reward)
    output=artifacts.emit(snapshot,reason)
    atomic_json(artifacts.root/'run_status.json',dict(status=reason,snapshot=snapshot.to_dict(),
        precise_status=None if output is None else output['status']))
    return dict(reason=reason,snapshot=snapshot,precise=output)
