"""Multi-step graph PPO selecting interfaces and bounded modification ranges."""
import csv
import json
import math
import os
from pathlib import Path

import torch
from torch import nn

from .joint import JointPartitionEnv, objective
from .structured import candidate_partitions
from .quality import validate_partition


class GraphPolicy(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.input = nn.Linear(26, hidden)
        self.messages = nn.ModuleList([nn.Linear(hidden * 2, hidden) for _ in range(2)])
        self.actor = nn.Sequential(nn.Linear(hidden * 3 + 3, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.value = nn.Linear(hidden, 1)
        self.stop = nn.Linear(hidden, 1)

    def forward(self, x, edges, actions):
        h = torch.tanh(self.input(x))
        for layer in self.messages:
            aggregate = torch.zeros_like(h)
            counts = torch.zeros((len(h), 1), dtype=h.dtype)
            if edges.numel():
                src, dst = edges.T
                aggregate.index_add_(0, dst, h[src])
                counts.index_add_(0, dst, torch.ones((len(dst), 1)))
            h = torch.tanh(layer(torch.cat((h, aggregate / counts.clamp_min(1)), -1)))
        pooled = h.mean(0)
        if actions:
            a = torch.tensor([v.a for v in actions]); b = torch.tensor([v.b for v in actions])
            radius = torch.tensor([[v.radius / 8., float(v.kind=='lobby'), v.variant/4.] for v in actions])
            z = torch.cat(((h[a]+h[b])/2, (h[a]-h[b]).abs(), pooled.expand(len(actions), -1), radius), -1)
            logits = self.actor(z).flatten()
        else:
            logits = torch.empty(0)
        return torch.cat((logits, self.stop(pooled))), self.value(pooled).squeeze()


def _observation(env, actions):
    x, edges = env.observation(actions)
    directed = edges + [(b, a) for a, b in edges]
    return torch.tensor(x), torch.tensor(directed, dtype=torch.long).reshape(-1, 2)


def policy_rollout(problem, model, initial=None, steps=8):
    """Inference on a fresh graph; keep the best independently validated state."""
    env = JointPartitionEnv(problem, initial)
    used = set()
    for _ in range(steps):
        actions = [a for a in env.actions() if a not in used]
        x, edges = _observation(env, actions)
        with torch.no_grad():
            logits, _ = model(x, edges, actions)
        at = int(logits.argmax())
        if at == len(actions): break
        _, _, info = env.step(actions[at])
        used.add(actions[at])
        if info['changed']: used.clear()
    return env.best, env.best_report, env.history


def train_partition_policy(problem, config, output_dir, episodes=None, stop_file=None):
    opt = config.get('FloorPartition', {}).get('rl', {})
    if opt.get('algorithm','graph_ppo') not in ('graph_ppo','legacy_bandit'):
        raise ValueError('Unknown floor partition training algorithm')
    if opt.get('algorithm') == 'legacy_bandit':
        from .bandit import train_partition_policy as legacy
        return legacy(problem, config, output_dir, episodes, stop_file)
    count = int(episodes if episodes is not None else config.get('Training', {}).get('episodes', opt.get('episodes', 128)))
    horizon = int(opt.get('steps_per_episode', 8))
    epochs = int(opt.get('ppo_epochs', 3))
    lr = float(opt.get('learning_rate', .0003))
    entropy = float(opt.get('entropy_coef', .02))
    gamma = float(opt.get('gamma', .95))
    if count < 1 or horizon < 1 or epochs < 1 or not 0 < gamma <= 1:
        raise ValueError('Invalid PPO episodes, horizon, epochs or gamma')
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(entropy) or entropy < 0:
        raise ValueError('Invalid PPO learning_rate or entropy_coef')
    seed = int(opt.get('seed', 42))
    threads=int(opt.get('torch_threads',1))
    if threads<1: raise ValueError('torch_threads must be positive')
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    pool = candidate_partitions(problem)
    env = JointPartitionEnv(problem, pool[0][0])
    starts=int(opt.get('initial_starts',3))
    if starts<1: raise ValueError('initial_starts must be positive')
    environments=[env]
    for result,_ in pool[1:min(starts,len(pool))]:
        environments.append(JointPartitionEnv(problem,result))
    model = GraphPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    initial_weights = torch.cat([p.detach().flatten().clone() for p in model.parameters()])
    baseline = float(objective(env.report))
    best, best_report = env.best, env.best_report
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    metrics = []; transitions = []; stopped = False
    for episode in range(1, count + 1):
        if stop_file and Path(stop_file).exists(): stopped = True; break
        env=environments[(episode-1)%len(environments)]
        env.reset(); used = set(); rollout = []
        for step in range(horizon):
            if stop_file and Path(stop_file).exists(): stopped = True; break
            actions = [a for a in env.actions() if a not in used]
            x, edges = _observation(env, actions)
            with torch.no_grad():
                logits, value = model(x, edges, actions)
                distribution = torch.distributions.Categorical(logits=logits)
                selected = distribution.sample()
                old_logp = distribution.log_prob(selected)
            if int(selected) == len(actions):
                reward = 0.
                info = dict(step=step+1, stopped_by_policy=True, reward=0., changed=False)
            else:
                action = actions[int(selected)]
                _, reward, info = env.step(action, stop_file)
                reward -= .002
                used.add(action)
                if info['changed']: used.clear()
            rollout.append((x, edges, actions, selected, old_logp, value, reward))
            transitions.append(dict(episode=episode, **info))
            if objective(env.best_report) > objective(best_report):
                best, best_report = env.best, env.best_report
            if info.get('stopped'): stopped = True; break
            if int(selected) == len(actions): break
        if not rollout: break
        # The finite horizon is the episode definition, hence zero terminal bootstrap.
        returns = []; total = 0.
        for item in reversed(rollout):
            total = item[-1] + gamma * total; returns.append(total)
        returns = torch.tensor(returns[::-1], dtype=torch.float32)
        advantage = returns - torch.stack([r[5] for r in rollout])
        if len(advantage) > 1 and float(advantage.std(unbiased=False)) > 1e-6:
            advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False)
        for _ in range(epochs):
            losses = []
            for i, (x, edges, actions, selected, old_logp, _, _) in enumerate(rollout):
                logits, value = model(x, edges, actions)
                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(selected) - old_logp).exp()
                policy = -torch.minimum(ratio * advantage[i], ratio.clamp(.8, 1.2) * advantage[i])
                losses.append(policy + .5 * (value-returns[i]).square() - entropy * dist.entropy())
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        metrics.append(dict(episode=episode, steps=len(rollout), reward=float(sum(r[-1] for r in rollout)),
                            loss=float(loss.detach()), best_score=float(objective(best_report))))
        if episode == 1 or episode % 10 == 0 or episode == count:
            print(f'PID: {os.getpid()}, Episode[{episode}/{count}], Total Step: {len(transitions)}, Rwd(mean): {metrics[-1]["reward"]:.4f}', flush=True)
        if stopped: break
    final_weights = torch.cat([p.detach().flatten() for p in model.parameters()])
    summary = dict(algorithm='graph_neighbourhood_PPO', episodes_requested=count,
        episodes_completed=len(metrics), steps_completed=len(transitions), stopped=stopped,
        seed=seed, parameter_delta=float(torch.linalg.vector_norm(final_weights-initial_weights)),
        initial_score=baseline, exported_score=float(objective(best_report)),
        graph_cells=len(env.cells), graph_edges=len(env.edges),initialization_count=len(environments),
        changed_steps=sum(bool(t.get('changed')) for t in transitions),
        incumbent_source='best_validated_layout_visited_during_multistep_rollouts',
        scope='single-building training; heuristic local repair; no generalisation or optimality claim')
    torch.save(dict(schema='floor-partition-joint-ppo-v3', state_dict=model.state_dict(),
                    optimizer=optimizer.state_dict(), config=config, summary=summary), destination/'partition_policy.pt')
    with (destination/'training_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['episode','steps','reward','loss','best_score'])
        writer.writeheader(); writer.writerows(metrics)
    (destination/'rl_training.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (destination/'search_history.json').write_text(json.dumps(transitions, indent=2), encoding='utf-8')
    if not validate_partition(problem, best)['valid']: raise RuntimeError('Invalid training incumbent')
    return best, best_report, summary
