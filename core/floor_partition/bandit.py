"""Single-step constrained policy-gradient RL over structural layout actions.

The action set is a bounded beam-search pool of independently verified plans.
This is a contextual bandit, not MAPPO or a claim of multi-building generalisation.
The best verified search result is retained as a safety incumbent.
"""
from pathlib import Path
import csv
import json
import os
import numpy as np
import torch
from .structured import candidate_partitions, score_report
from .quality import validate_partition


def features(report):
    units=list(report['units'].values())
    errors=[m['area_error_ratio'] for m in units]
    ratios=[m['facade_length']/m['required_facade_length'] for m in units]
    return [np.mean(errors),max(errors),np.mean([m['corners'] for m in units]),
            report['corridor_area'],min(ratios),np.std(ratios),report['structure_alignment_ratio']]


def train_partition_policy(problem, config, output_dir, episodes=None, stop_file=None):
    options=config.get('FloorPartition',{}).get('rl',{})
    count=int(episodes if episodes is not None else config.get('Training',{}).get('episodes',options.get('episodes',128)))
    if count<=0: raise ValueError('RL episodes 必须大于0')
    seed=int(options.get('seed',config.get('Training',{}).get('seed',42)))
    torch.manual_seed(seed)
    pairs=candidate_partitions(problem)
    raw=torch.tensor([features(report) for _,report in pairs],dtype=torch.float32)
    mean=raw.mean(0); std=raw.std(0,unbiased=False).clamp_min(1e-6)
    x=(raw-mean)/std
    rewards=torch.tensor([score_report(report) for _,report in pairs],dtype=torch.float32)
    model=torch.nn.Linear(x.shape[1],1)
    torch.nn.init.zeros_(model.weight); torch.nn.init.zeros_(model.bias)
    optimizer=torch.optim.Adam(model.parameters(),lr=float(options.get('learning_rate',.01)))
    entropy_coef=float(options.get('entropy_coef',.02))
    if not np.isfinite(entropy_coef) or entropy_coef<0: raise ValueError('entropy_coef 必须为非负有限数')
    initial=model.weight.detach().clone(); baseline=float(rewards.mean()); history=[]
    initial_expected=float(rewards.mean()); stopped=False
    destination=Path(output_dir); destination.mkdir(parents=True,exist_ok=True)
    for episode in range(1,count+1):
        if stop_file and Path(stop_file).exists(): stopped=True; break
        distribution=torch.distributions.Categorical(logits=model(x).squeeze(-1))
        action=distribution.sample(); reward=rewards[action]
        advantage=(reward-baseline).detach()
        loss=-distribution.log_prob(action)*advantage-entropy_coef*distribution.entropy()
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step(); baseline=.9*baseline+.1*float(reward)
        with torch.no_grad():
            expected=float((torch.softmax(model(x).squeeze(-1),0)*rewards).sum())
        history.append(dict(episode=episode,action=int(action),reward=float(reward),
                            loss=float(loss.detach()),expected_reward=expected))
        if episode==1 or episode%10==0 or episode==count:
            print(f'PID: {os.getpid()}, Episode[{episode}/{count}], Total Step: {episode}, Rwd(mean): {float(reward):.4f}',flush=True)
    with torch.no_grad():
        probabilities=torch.softmax(model(x).squeeze(-1),0)
        policy_choice=int(probabilities.argmax())
    # RL exploration must not replace an already better validated incumbent.
    best=int(rewards.argmax()); result,report=pairs[best]
    assert validate_partition(problem,result)['valid']
    summary=dict(algorithm='constrained_contextual_bandit_REINFORCE',candidate_count=len(pairs),
        episodes_requested=count,episodes_completed=len(history),stopped=stopped,seed=seed,
        parameter_delta=float(torch.linalg.vector_norm(model.weight.detach()-initial)),
        initial_expected_reward=initial_expected,
        final_expected_reward=float((probabilities*rewards).sum()),
        policy_selected_candidate=policy_choice,exported_candidate=best,
        exported_reward=float(rewards[best]),incumbent_source='best_independently_validated_search_candidate',
        scope='single-building finite action pool; no claim of global optimum or generalisation')
    torch.save(dict(schema='floor-partition-bandit-v1',state_dict=model.state_dict(),
                    optimizer=optimizer.state_dict(),feature_mean=mean,feature_std=std,
                    features=raw,rewards=rewards,seed=seed,config=config),destination/'partition_policy.pt')
    with (destination/'training_metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['episode','action','reward','loss','expected_reward'])
        writer.writeheader(); writer.writerows(history)
    (destination/'rl_training.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return result,report,summary
