"""Experimental seed training: 250 episodes / finish / requested stop output."""

# Allow direct script execution from any working directory.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from datetime import datetime
import yaml
import signal
from core.seed_growth.environment import SeedLayoutEnv
from core.seed_growth.artifacts import RunArtifacts, atomic_json
from core.seed_growth.training import run_training, restore_checkpoint, stop_requested, model_schema
from common.training import build_mapppo_agent
from common.project_paths import STOP_REQUEST_FILE


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path)
    parser.add_argument('--config-id',default='retrofit')
    parser.add_argument('--episodes',type=int,help='目标累计完成轮数（续训也使用累计数）')
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--resume',type=Path,help='本版本运行目录，不是旧矩形模型目录')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--stop-file',type=Path,default=Path(STOP_REQUEST_FILE))
    args=parser.parse_args()
    if args.resume and args.output: parser.error('续训沿用原运行目录，不同时指定 --output')
    if args.resume:
        manifest=args.resume/'manifest.json'
        if (not manifest.exists() or not (args.resume/'latest_checkpoint.json').exists()
                or json.loads(manifest.read_text(encoding='utf-8')).get('mode')!='seed_proxy_v3'):
            parser.error('--resume 只接受 v3 完整运行目录；旧矩形/CP2/v2 观测格式不同，不能直接续训；原成图仍可查看和导出')
    from common.config_manager import get_config_path
    source=args.config or (args.resume/'config.yaml' if args.resume else Path(get_config_path(args.config_id)))
    config=yaml.safe_load(source.read_text(encoding='utf-8'))
    if config.get('Training',{}).get('ckpt_path'): parser.error('请用 --resume 续训种子模式；不能加载旧矩形权重')
    if args.max_steps is not None: config['Training']['max_steps']=args.max_steps
    episodes=args.episodes if args.episodes is not None else int(config['Training'].get('episodes',1000))
    if min(episodes,int(config['Training'].get('max_steps',240)))<1: parser.error('轮次与步数必须大于零')
    if int(config['Training'].get('batch_size',64))<2: parser.error('种子 PPO 批量大小至少为 2，避免单样本优势归一化失效')
    import torch
    import numpy as np
    import random
    seed=int(config['Training'].get('seed',42))
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    env=SeedLayoutEnv(config)
    agent=build_mapppo_agent(env,config,reset_seed=int(config['Training'].get('seed',42)))
    out=args.resume or args.output or Path(__file__).resolve().parents[1]/'results2/seed_experimental'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if not args.resume and out.exists(): parser.error('新训练输出目录必须不存在；已有运行请用 --resume')
    artifacts=RunArtifacts(out,config,env.problem,env.proxy,
                          export_root=out)
    restored=restore_checkpoint(out,env,agent) if args.resume else None
    atomic_json(out/'schema.json',model_schema(env))
    requested=[]
    def handle_signal(signum,frame): requested.append('interrupted')
    signal.signal(signal.SIGINT,handle_signal)
    signal.signal(signal.SIGTERM,handle_signal)
    def check():
        return requested[0] if requested else ('stop_requested' if stop_requested(args.stop_file) else None)
    print(f'运行目录：{out}',flush=True)
    result=run_training(env,agent,artifacts,episodes,stop_check=check,resume=restored,
                        save_interval=max(1,int(config['Training'].get('save_interval',20))))
    print(f'结束原因：{result["reason"]}；已完成 {result["snapshot"].completed_episodes} 轮，回合内 {result["snapshot"].step} 步。精确结果见 {out}/latest_precise.json',flush=True)
    if result['precise'] and result['precise']['status']=='decode_failed': raise SystemExit(1)


if __name__=='__main__': main()
