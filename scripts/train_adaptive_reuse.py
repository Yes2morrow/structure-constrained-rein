"""既有建筑适应性转换环境的 MAPPO 训练入口。"""

from __future__ import annotations

# Allow direct script execution from any working directory.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import csv
from datetime import datetime
import shutil

import matplotlib
import numpy as np

from common.config_manager import get_config_path
from core.envs import make_adaptive_reuse_env
from common.training import build_mapppo_agent
from common.project_paths import STOP_REQUEST_FILE
from core.seed_growth.training import stop_requested
from core.seed_growth.artifacts import atomic_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="既有建筑多智能体适应性转换训练")
    parser.add_argument("--config-id", default="retrofit")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--stop-file", type=Path, default=Path(STOP_REQUEST_FILE))
    return parser.parse_args()


def main():
    args = parse_args()
    matplotlib.use("Agg", force=True)
    config_path = args.config or Path(get_config_path(args.config_id))
    import yaml
    selected=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if str(selected.get('Training',{}).get('training_stage','room_training')).strip() == 'floor_partition':
        from scripts.train_floor_partition import main as floor_main
        return floor_main()
    if any(r.get('role')=='residual' for r in selected.get('TargetSpaces',[])):
        from scripts.train_seed_layout import main as seed_main
        return seed_main()
    env, config = make_adaptive_reuse_env(config_path)
    training = config["Training"]
    episodes = args.episodes if args.episodes is not None else int(training.get("episodes", 1000))
    max_steps = args.max_steps if args.max_steps is not None else int(training.get("max_steps", 240))
    env.max_steps = max_steps
    seed = int(training.get("seed", 42))
    save_interval = max(int(training.get("save_interval", 20)), 1)
    output_dir = PROJECT_ROOT / "results2" / "adaptive_reuse" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    agent = build_mapppo_agent(env, config, reset_seed=seed)

    ckpt_path = str(training.get("ckpt_path", "")).strip()
    resume_mode = str(training.get("resume_mode", "fresh"))
    if ckpt_path and resume_mode in {"resume", "new_from_model"}:
        agent.load(ckpt_path, load_training_state=resume_mode == "resume")
        print(f"已加载模型：{ckpt_path}（模式：{resume_mode}）")

    best_reward = -np.inf
    completed_episodes = 0
    stopped = False
    columns = [
        "episode", "steps", "mean_reward", "area_compliance", "shape_compliance",
        "adjacency_score", "original_reuse", "intervention_ratio", "hard_conflicts",
    ]
    with open(output_dir / "training_metrics.csv", "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for episode in range(episodes):
            if stop_requested(args.stop_file):
                stopped = True
                break
            output = agent.learn(env, max_steps, seed=seed + episode, render=False, num_agents=env.num_agents)
            completed_episodes = episode + 1
            mean_reward = float(np.mean(output["episode_steps_reward"]))
            metrics = env.calculate_metrics()
            writer.writerow({
                "episode": episode + 1, "steps": output["total_step"] - 1, "mean_reward": mean_reward,
                **{key: metrics[key] for key in columns[3:]},
            })
            stream.flush()
            if mean_reward > best_reward:
                best_reward = mean_reward
                agent.save(str(output_dir / "best_model"))
                env.render().savefig(output_dir / "best_layout.png", dpi=180)
            if (episode + 1) % save_interval == 0 or episode + 1 == episodes:
                agent.save(str(output_dir / "final_model"))
                env.render().savefig(output_dir / "latest_layout.png", dpi=180)
            print(
                f"Episode {episode + 1}/{episodes} | reward={mean_reward:.3f} | "
                f"area={metrics['area_compliance']:.0%} | reuse={metrics['original_reuse']:.0%} | "
                f"intervention={metrics['intervention_ratio']:.0%} | conflicts={int(metrics['hard_conflicts'])}"
            )
    if stopped:
        agent.save(str(output_dir / "final_model"))
        env.render().savefig(output_dir / "latest_layout.png", dpi=180)
    atomic_json(output_dir / "run_status.json",dict(status="stop_requested" if stopped else "finished",completed_episodes=completed_episodes))
    print(f"训练结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
