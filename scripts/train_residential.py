"""住区楼栋布局的独立训练入口。"""

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

from core.envs import make_residential_env
from common.training import build_mapppo_agent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "residential" / "config.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="多智能体住区楼栋布局训练")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int, default=None, help="覆盖配置中的训练回合数")
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖配置中的单回合步数")
    return parser.parse_args()


def main():
    args = parse_args()
    matplotlib.use("Agg", force=True)
    env, config = make_residential_env(args.config)
    training = config["Training"]
    episodes = args.episodes if args.episodes is not None else int(training["episodes"])
    max_steps = args.max_steps if args.max_steps is not None else int(training["max_steps"])
    env.max_steps = max_steps
    save_interval = max(int(training.get("save_interval", 25)), 1)
    seed = int(training.get("seed", 42))

    output_dir = PROJECT_ROOT / "results2" / "residential" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "config.yaml")
    agent = build_mapppo_agent(env, config, reset_seed=training.get("seed"))
    best_reward = -np.inf

    with open(output_dir / "training_metrics.csv", "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["episode", "steps", "mean_reward", "median_reward", "far", "density", "overlap_ratio", "spacing_compliance"])
        for episode in range(episodes):
            output = agent.learn(env, max_steps, seed=seed + episode, render=False, num_agents=env.num_agents)
            mean_reward = float(np.mean(output["episode_steps_reward"]))
            median_reward = float(np.median(output["episode_steps_reward"]))
            metrics = env.calculate_metrics()
            writer.writerow([
                episode + 1, output["total_step"] - 1, mean_reward, median_reward,
                metrics["far"], metrics["density"], metrics["overlap_ratio"], metrics["spacing_compliance"],
            ])
            stream.flush()

            if median_reward > best_reward:
                best_reward = median_reward
                agent.save(str(output_dir / "best_model"))
                figure = env.render()
                figure.savefig(output_dir / "best_layout.png", dpi=200)

            if (episode + 1) % save_interval == 0 or episode + 1 == episodes:
                agent.save(str(output_dir / "final_model"))
                figure = env.render()
                figure.savefig(output_dir / "latest_layout.png", dpi=200)
            print(
                f"Episode {episode + 1}/{episodes} | reward={mean_reward:.3f} | "
                f"FAR={metrics['far']:.3f} | density={metrics['density']:.3f} | "
                f"spacing={metrics['spacing_compliance']:.1%}"
            )
    print(f"训练结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
