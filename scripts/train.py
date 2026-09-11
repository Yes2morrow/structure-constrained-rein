"""
主函数代码: 不再支持单智能体的训练和测试
"""

# Allow direct script execution from any working directory.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
from typing import Any, Union
import os
import time
import yaml
import copy 
from datetime import datetime
import numpy as np
import multiprocessing as mp
import matplotlib
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from core import (
    make_env, HouseEnv, MAPPO, QMIX, MADQN,
    save_layout_image, copy_yaml_file
)
from gui.manual_env import InteractiveHouseEnv
from common.config_manager import DEFAULT_CONFIG_ID, get_config_path, normalize_config_id
from common.project_paths import PID_FILE, RESULTS_DIR, STOP_REQUEST_FILE


def parse_args():
    """解析命令行参数，支持按编号加载配置。"""
    parser = argparse.ArgumentParser(description="房屋布局强化学习训练入口")
    parser.add_argument(
        "--config-id",
        type=str,
        default=DEFAULT_CONFIG_ID,
        help="读取的配置编号，对应 config/<编号>/config.yaml",
    )
    return parser.parse_args()


def normalize_resume_mode(resume_mode: str) -> str:
    """标准化续训模式。"""
    valid_modes = {"fresh", "resume", "new_from_model"}
    normalized_mode = str(resume_mode or "fresh").strip().lower()
    if normalized_mode not in valid_modes:
        return "fresh"
    return normalized_mode


BASE_METRIC_COLUMNS = [
    "timestamp",
    "episode",
    "total_episodes",
    "episode_steps",
    "total_env_steps",
    "mean_reward",
    "median_reward",
    "episode_seconds",
    "step_seconds",
    "avg_episode_seconds",
    "avg_step_seconds",
    "steps_per_second",
    "avg_steps_per_second",
]

BASE_REWARD_COMPONENT_COLUMNS = [
    "area_reward",
    "aspect_ratio_reward",
    "edge_reward",
    "shared_wall_reward",
    "door_to_door_reward",
    "corner_reward",
    "door_block_reward",
    "prior_layout_reward",
    "overlap_reward",
    "nesting_penalty_reward",
]

TRAINING_METRICS_COLUMNS = BASE_METRIC_COLUMNS + BASE_REWARD_COMPONENT_COLUMNS


def prepare_metrics_csv(metrics_csv_path: str, append_metrics: bool, expected_columns: list[str]) -> tuple[str, bool]:
    """确保训练指标 CSV 表头与当前版本一致。"""
    if not append_metrics or not os.path.isfile(metrics_csv_path):
        return "w", True

    with open(metrics_csv_path, "r", encoding="utf-8", newline="") as metrics_file:
        reader = csv.DictReader(metrics_file)
        existing_columns = reader.fieldnames or []
        existing_rows = list(reader)

    if existing_columns == expected_columns:
        return "a", False

    # 兼容旧版 CSV：补齐新增列并保留既有训练记录。
    with open(metrics_csv_path, "w", encoding="utf-8", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=expected_columns)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow({column: row.get(column, "") for column in expected_columns})
    return "a", False


def resolve_run_dir_from_checkpoint(ckpt_path: str) -> str:
    """根据模型目录推导所属结果目录。"""
    checkpoint_dir = os.path.abspath(ckpt_path)
    parent_dir = os.path.dirname(checkpoint_dir)
    if os.path.basename(parent_dir).lower() == "checkpoints":
        return os.path.dirname(parent_dir)
    return parent_dir


def is_stop_requested() -> bool:
    """检查是否收到网页端停止请求。"""
    return os.path.isfile(STOP_REQUEST_FILE)


def clear_stop_request() -> None:
    """清理停止请求标记文件。"""
    if os.path.exists(STOP_REQUEST_FILE):
        os.remove(STOP_REQUEST_FILE)


class Trainer(object):
    """ 执行训练的类 """
    def __init__(
        self,
        agent_name:str="DQN",
        muliple_agents:bool=False,
        save_dir:str="results",
    ) -> None:
        # 初始化
        self.agent_name = agent_name
        self.multiple_agents = muliple_agents
        self.save_dir = os.path.join(save_dir, agent_name.lower())
    
    def build_model(
    self, 
    env:HouseEnv, 
    batch_size:int=256, 
    lr:float=1e-3, 
    **kwargs) -> Union[MAPPO]:
        """ 构建模型 """
        algorithm_config = kwargs.get("algorithm_config", {}) or {}
        if self.agent_name.lower() == "dqn":
            self.multiple_agents = True
            state, info = env.reset()
            state_size_per_agent = state.shape[1]
            global_state_size = state.shape[0] * state.shape[1]
            action_size = env.action_space.nvec[0]
            agent = QMIX(
                state_size_per_agent=state_size_per_agent,
                global_state_size=global_state_size,
                action_size=action_size,
                num_agents=env.num_agents,
                hidden_size=int(algorithm_config.get("qmix_hidden_size", 128)),
                mixing_hidden_size=int(algorithm_config.get("qmix_mixing_hidden_size", 32)),
                hyper_hidden_size=int(algorithm_config.get("qmix_hyper_hidden_size", 64)),
                lr=lr,
                gamma=float(algorithm_config.get("qmix_gamma", 0.99)),
                tau=float(algorithm_config.get("qmix_tau", 0.01)),
                batch_size=batch_size,
                buffer_size=int(algorithm_config.get("qmix_buffer_size", 20000)),
                update_frequency=int(algorithm_config.get("qmix_update_frequency", 50)),
                epsilon=float(algorithm_config.get("qmix_epsilon", 0.12)),
            )
        elif self.agent_name.lower() == "mappo":
            self.multiple_agents = True
            state, info = env.reset()
            state_size_per_agent = state.shape[1]  # 每个智能体的状态维度
            global_state_size = state.shape[0] * state.shape[1]  # 原始全局状态维度
            action_size = env.action_space.nvec[0]  # 每个智能体的动作数量
            
            # 注意：现在MAPPO会自动计算扩展的全局状态大小
            agent = MAPPO(
                state_size_per_agent=state_size_per_agent,
                global_state_size=global_state_size,  # 传入原始全局状态大小
                action_size=action_size,
                hidden_size=int(algorithm_config.get("mappo_hidden_size", 256)),
                lr=lr,
                gamma=float(algorithm_config.get("mappo_gamma", 0.99)),
                gae_lambda=float(algorithm_config.get("mappo_gae_lambda", 0.95)),
                clip_epsilon=float(algorithm_config.get("mappo_clip_epsilon", 0.2)),
                value_coef=float(algorithm_config.get("mappo_value_coef", 0.5)),
                entropy_coef=float(algorithm_config.get("mappo_entropy_coef", 0.01)),
                ppo_epochs=int(algorithm_config.get("mappo_ppo_epochs", 5)),
                batch_size=batch_size,
                max_grad_norm=float(algorithm_config.get("mappo_max_grad_norm", 0.5)),
                update_step=int(algorithm_config.get("mappo_update_step", 1024)),
                reward_clip_range=float(algorithm_config.get("mappo_reward_clip_range", 8.0)),
                num_agents=env.num_agents,
            )
        elif self.agent_name.lower() == "madqn":
            self.multiple_agents = True
            state, info = env.reset()
            state_size_per_agent = state.shape[1]  # 每个智能体的状态维度
            global_state_size = state.shape[0] * state.shape[1]  # 原始全局状态维度
            action_size = env.action_space.nvec[0]  # 每个智能体的动作数量
            agent = MADQN(
                state_dim=None,
                local_state_dim=state_size_per_agent,
                action_dim=action_size,
                num_agents=env.num_agents,
                hidden_dim=int(algorithm_config.get("madqn_hidden_dim", 256)),
                lr=float(algorithm_config.get("madqn_lr", 0.0001)),
                gamma=float(algorithm_config.get("madqn_gamma", 0.99)),
                tau=float(algorithm_config.get("madqn_tau", 0.01)),
                epsilon_start=float(algorithm_config.get("madqn_epsilon_start", 1.0)),
                epsilon_end=float(algorithm_config.get("madqn_epsilon_end", 0.01)),
                epsilon_decay=float(algorithm_config.get("madqn_epsilon_decay", 0.9995)),
                buffer_capacity=int(algorithm_config.get("madqn_buffer_capacity", 100000)),
                batch_size=batch_size,
                device=str(algorithm_config.get("madqn_device", "cuda:0"))
            )
        else:
            raise NotImplementedError("NotImplementedError: {} is not implemented.".format(self.agent_name))
        return agent

    def _get_trainer_state_path(self, path_dir: str) -> str:
        """返回训练状态文件路径。"""
        return os.path.join(path_dir, "trainer_state.json")

    def _load_trainer_state(self, pretrain_ckpt_path: str) -> dict[str, Any]:
        """从模型目录或所属结果目录中恢复训练状态。"""
        candidate_dirs = [os.path.abspath(pretrain_ckpt_path), resolve_run_dir_from_checkpoint(pretrain_ckpt_path)]
        for candidate_dir in candidate_dirs:
            trainer_state_path = self._get_trainer_state_path(candidate_dir)
            if not os.path.isfile(trainer_state_path):
                continue
            try:
                with open(trainer_state_path, "r", encoding="utf-8") as file:
                    state = json.load(file)
                if isinstance(state, dict):
                    return state
            except Exception as exc:
                print(f"读取训练状态失败: {trainer_state_path}, error={exc}")
        return {}

    def _write_trainer_state(self, path_dir: str, state: dict[str, Any]) -> None:
        """写入训练状态文件。"""
        os.makedirs(path_dir, exist_ok=True)
        trainer_state_path = self._get_trainer_state_path(path_dir)
        with open(trainer_state_path, "w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False, indent=2)

    def _sync_trainer_state(self, run_dir: str, state: dict[str, Any], extra_dirs: list[str] | None = None) -> None:
        """同步训练状态到结果目录和指定模型目录。"""
        self._write_trainer_state(run_dir, state)
        for extra_dir in extra_dirs or []:
            if extra_dir:
                self._write_trainer_state(extra_dir, state)

    def _copy_runtime_config(self, config_path: str | None, save_dir: str, resume_mode: str) -> None:
        """保存当前用于启动训练的配置快照。"""
        if not config_path:
            return
        if resume_mode == "resume":
            if not os.path.isfile(os.path.join(save_dir, "config.yaml")):
                copy_yaml_file(config_path, os.path.join(save_dir, "config.yaml"))
            copy_yaml_file(config_path, os.path.join(save_dir, "config_resume_latest.yaml"))
            return
        copy_yaml_file(config_path, os.path.join(save_dir, "config.yaml"))
    
    def train(
        self, 
        env:HouseEnv,
        episodes:int=10000,
        max_iters:int=1000,
        batch_size:int=256,
        lr:float=1e-3,
        seed:int=None,
        pretrain_ckpt_path:str=None,
        render:bool=False,
        render_every_steps:int=1,
        save_threshold:float=200,
        final_model_save_interval:int=1,
        latest_layout_save_interval:int=1,
        threshold_save_interval:int=1,
        config_path:dict=None,
        algorithm_config:dict=None,
        resume_mode:str="fresh",
    ) -> None:
        """ 训练Agent """
        resume_mode = normalize_resume_mode(resume_mode)
        if not pretrain_ckpt_path:
            resume_mode = "fresh"
        clear_stop_request()

        PID = os.getpid()
        trainer_state = {}
        if resume_mode == "resume" and pretrain_ckpt_path:
            trainer_state = self._load_trainer_state(pretrain_ckpt_path)
            save_dir = resolve_run_dir_from_checkpoint(pretrain_ckpt_path)
        else:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(self.save_dir, run_name)
            suffix = 2
            while os.path.exists(save_dir):
                save_dir = os.path.join(self.save_dir, f"{run_name}_{suffix}")
                suffix += 1
        os.makedirs(save_dir, exist_ok=True)
        print(f"训练模式: {resume_mode}")
        print(f"结果目录: {save_dir}")
        env.live_render_path = os.path.join(save_dir, "live_render.png")
        env.live_render_interval = float(getattr(env, "live_render_interval", 0.5))
        env.live_render_show_metrics = bool(getattr(env, "live_render_show_metrics", False))
        env.live_render_show_total_area = bool(getattr(env, "live_render_show_total_area", env.live_render_show_metrics))
        env.live_render_show_summary_panel = bool(getattr(env, "live_render_show_summary_panel", False))
        env.saved_image_show_metrics = bool(getattr(env, "saved_image_show_metrics", False))
        env.saved_image_show_total_area = bool(getattr(env, "saved_image_show_total_area", env.saved_image_show_metrics))
        env.saved_image_show_summary_panel = bool(getattr(env, "saved_image_show_summary_panel", False))
        metrics_csv_path = os.path.join(save_dir, "training_metrics.csv")
        print(f"速度指标文件: {metrics_csv_path}")

        # 创建tensorboard
        tb_writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard"))
        
        self._copy_runtime_config(config_path, save_dir, resume_mode)
        
        # 加载模型
        agent = self.build_model(env, batch_size, lr, algorithm_config=algorithm_config or {})
        if pretrain_ckpt_path:
            agent.load(pretrain_ckpt_path, load_training_state=(resume_mode == "resume"))
        
        start_episode = int(trainer_state.get("episode", 0)) if resume_mode == "resume" else 0
        best_reward = float(trainer_state.get("best_reward", -np.inf)) if resume_mode == "resume" else -np.inf
        total_env_steps = int(trainer_state.get("total_env_steps", 0)) if resume_mode == "resume" else 0
        previous_total_elapsed = float(trainer_state.get("total_elapsed_seconds", 0.0)) if resume_mode == "resume" else 0.0
        total_target_episodes = max(int(episodes), start_episode) if resume_mode == "resume" else int(episodes)
        session_train_start_time = time.perf_counter()

        if resume_mode == "resume":
            print(f"恢复训练状态: start_episode={start_episode}, total_env_steps={total_env_steps}, best_reward={best_reward:.3f}")
            if total_target_episodes <= start_episode:
                print(f"当前配置的训练轮数 {total_target_episodes} 不大于已完成轮数 {start_episode}，本次无需继续训练。")
                return

        # 保存先验知识
        prior_ = env.prior_positions
        with open(os.path.join(save_dir, "cpnfig.yaml"), 'w') as file:
            yaml.dump({"Prior_Positions": prior_}, file, allow_unicode=True,
                      default_flow_style=True)

        append_metrics = resume_mode == "resume" and os.path.isfile(metrics_csv_path)
        open_mode, write_metrics_header = prepare_metrics_csv(
            metrics_csv_path,
            append_metrics=append_metrics,
            expected_columns=TRAINING_METRICS_COLUMNS,
        )
        current_state = {
            "version": 1,
            "agent_name": self.agent_name,
            "resume_mode": resume_mode,
            "result_dir": save_dir,
            "episode": start_episode,
            "total_target_episodes": total_target_episodes,
            "best_reward": best_reward,
            "total_env_steps": total_env_steps,
            "total_elapsed_seconds": round(previous_total_elapsed, 6),
            "updated_at": int(time.time()),
        }
        stop_after_episode = False

        try:
            with open(metrics_csv_path, open_mode, encoding="utf-8", newline="") as metrics_file:
                metrics_writer = csv.writer(metrics_file)
                if write_metrics_header:
                    metrics_writer.writerow(TRAINING_METRICS_COLUMNS)

                for episode in range(start_episode, total_target_episodes):
                    episode_start_time = time.perf_counter()
                    # 训练一个回合
                    output = agent.learn(env, max_iters, seed=seed, render=render,
                                         render_every_steps=render_every_steps,
                                         rnd=True if episode % 5 == 0 else False,
                                         num_agents=env.num_agents)
                    current_episode = episode + 1
                    episode_steps = max(int(output['total_step']) - 1, 1)
                    total_env_steps += episode_steps
                    episode_elapsed = time.perf_counter() - episode_start_time
                    total_elapsed = previous_total_elapsed + (time.perf_counter() - session_train_start_time)
                    episode_avg_step_time = episode_elapsed / episode_steps
                    avg_episode_time = total_elapsed / current_episode
                    avg_step_time = total_elapsed / max(total_env_steps, 1)
                    steps_per_second = episode_steps / max(episode_elapsed, 1e-9)
                    avg_steps_per_second = total_env_steps / max(total_elapsed, 1e-9)
                    mean_reward = float(np.mean(output['episode_steps_reward']))
                    median_reward = float(np.median(output['episode_steps_reward']))
                    episode_reward_components = output.get("episode_reward_components", {})

                    metrics_writer.writerow([
                        int(time.time()),
                        current_episode,
                        total_target_episodes,
                        episode_steps,
                        total_env_steps,
                        mean_reward,
                        median_reward,
                        round(episode_elapsed, 6),
                        round(episode_avg_step_time, 6),
                        round(avg_episode_time, 6),
                        round(avg_step_time, 6),
                        round(steps_per_second, 6),
                        round(avg_steps_per_second, 6),
                        round(float(episode_reward_components.get("area_reward", 0.0)), 6),
                        round(float(episode_reward_components.get("aspect_ratio_reward", 0.0)), 6),
                        round(float(episode_reward_components.get("edge_reward", 0.0)), 6),
                        round(float(episode_reward_components.get("shared_wall_reward", 0.0)), 6),
                        round(float(episode_reward_components.get("door_to_door_reward", 0.0)), 6),
                        round(float(episode_reward_components.get("corner_reward", 0.0)), 6),
                        round(float(episode_reward_components.get("door_block_reward", 0.0)), 6),
                        round(float(episode_reward_components.get("prior_layout_reward", 0.0)), 6),
                        round(float(episode_reward_components.get("overlap_reward", 0.0)), 6),
                        round(float(episode_reward_components.get("nesting_penalty_reward", 0.0)), 6),
                    ])
                    metrics_file.flush()

                    # 整理数据
                    tb_writer.add_scalar("episode step", output['total_step'], current_episode)
                    tb_writer.add_scalar("episode reward median", median_reward, current_episode)
                    tb_writer.add_scalar("time/episode_seconds", episode_elapsed, current_episode)
                    tb_writer.add_scalar("time/step_seconds", episode_avg_step_time, current_episode)
                    tb_writer.add_scalar("time/avg_episode_seconds", avg_episode_time, current_episode)
                    tb_writer.add_scalar("time/avg_step_seconds", avg_step_time, current_episode)
                    tb_writer.add_scalar("time/steps_per_second", steps_per_second, current_episode)
                    tb_writer.add_scalar("time/avg_steps_per_second", avg_steps_per_second, current_episode)
                    for i, r in output['episode_agent_reward'].items():
                        tb_writer.add_scalar(f"agent_{i} reward median",
                                             np.median(r), current_episode)

                    current_state = {
                        "version": 1,
                        "agent_name": self.agent_name,
                        "resume_mode": resume_mode,
                        "result_dir": save_dir,
                        "episode": current_episode,
                        "total_target_episodes": total_target_episodes,
                        "best_reward": best_reward,
                        "total_env_steps": total_env_steps,
                        "total_elapsed_seconds": round(total_elapsed, 6),
                        "updated_at": int(time.time()),
                    }

                    # 保存模型
                    if median_reward > best_reward:
                        best_reward = median_reward
                        current_state["best_reward"] = best_reward
                        agent.save(os.path.join(save_dir, "best_model"))
                        self._sync_trainer_state(save_dir, current_state, [os.path.join(save_dir, "best_model")])
                        print(f"保存 best_model: {os.path.join(save_dir, 'best_model')}")

                    # 打印日志
                    print(
                        "PID: {}, Episode[{}/{}], Total Step: {}, Rwd(mean): {:.3f}, Time(ep): {:.2f}s, Time(step): {:.4f}s, Avg(ep): {:.2f}s, Avg(step): {:.4f}s, Speed(ep): {:.2f} step/s, Speed(avg): {:.2f} step/s, ".format(
                            PID,
                            current_episode,
                            total_target_episodes,
                            output['total_step']-1,
                            mean_reward,
                            episode_elapsed,
                            episode_avg_step_time,
                            avg_episode_time,
                            avg_step_time,
                            steps_per_second,
                            avg_steps_per_second,
                        ) + ", ".join(
                            "Rwd(agent_{}): {:.3f}".format(i, np.mean(r))
                            for i, r in output['episode_agent_reward'].items()
                        )
                    )

                    # 按间隔保存最终模型与最新布局图；0 表示关闭自动保存
                    if final_model_save_interval > 0 and current_episode % final_model_save_interval == 0:
                        agent.save(os.path.join(save_dir, "final_model"))
                        self._sync_trainer_state(save_dir, current_state, [os.path.join(save_dir, "final_model")])
                        print(f"更新 final_model: {os.path.join(save_dir, 'final_model')}")

                    # 保存布局达到阈值条件的模型
                    threshold_save_due = (
                        mean_reward >= save_threshold
                        and threshold_save_interval > 0
                        and current_episode % threshold_save_interval == 0
                    )
                    if threshold_save_due:
                        check_point_dir = os.path.join(save_dir, "checkpoints", "epsiode_{:0>6d}_reward_{:.3f}".format(current_episode, mean_reward))
                        os.makedirs(check_point_dir, exist_ok=True)
                        # 保存模型
                        agent.save(check_point_dir)
                        self._sync_trainer_state(save_dir, current_state, [check_point_dir])
                        print(f"保存 checkpoint: {check_point_dir}")
                        # 奖励达标时优先保存奖励对应的布局图
                        image_path = os.path.join(save_dir, "images", "epsiode_{:0>6d}_reward_{:.3f}.png".format(current_episode, mean_reward))
                        os.makedirs(os.path.dirname(image_path), exist_ok=True)
                        save_layout_image(
                            env,
                            image_path,
                            init_livingroom=False,
                            show_room_metrics=env.saved_image_show_metrics,
                            show_total_area=env.saved_image_show_total_area,
                            show_summary_panel=env.saved_image_show_summary_panel,
                        )
                        print(f"保存布局图: {image_path}")
                    elif latest_layout_save_interval > 0 and current_episode % latest_layout_save_interval == 0:
                        # 奖励未达标时，再按轮数间隔更新最新布局图
                        latest_image_path = os.path.join(save_dir, "latest_layout.png")
                        save_layout_image(
                            env,
                            latest_image_path,
                            init_livingroom=False,
                            show_room_metrics=env.saved_image_show_metrics,
                            show_total_area=env.saved_image_show_total_area,
                            show_summary_panel=env.saved_image_show_summary_panel,
                        )
                        print(f"更新最新布局图: {latest_image_path}")

                    self._sync_trainer_state(save_dir, current_state)

                    if is_stop_requested():
                        final_model_dir = os.path.join(save_dir, "final_model")
                        agent.save(final_model_dir)
                        self._sync_trainer_state(save_dir, current_state, [final_model_dir])
                        print(f"检测到停止请求，已补存续训状态: {final_model_dir}")
                        stop_after_episode = True
                        break
        finally:
            tb_writer.close()
            if stop_after_episode:
                clear_stop_request()
            if os.path.exists(PID_FILE):
                try:
                    with open(PID_FILE, "r", encoding="utf-8") as file:
                        running_pid = int(file.read().strip())
                    if running_pid == PID:
                        os.remove(PID_FILE)
                except Exception:
                    pass
            
    def evaluate(
        self, 
        env:HouseEnv,
        max_iters:int=1000,
        seed:int=None,
        pretrain_ckpt_path:str=None,
        render:bool=True,
    ) -> None:
        """ 测验模型 """
        # 加载模型
        agent = self.build_model(env)
        if pretrain_ckpt_path:
            agent.load(pretrain_ckpt_path)
        
        # 测试模型
        env = agent.evaluate(env, max_iters, render=render, seed=seed)
        
        # 测试完成后进行调试
        interactive_env = InteractiveHouseEnv(env)
        interactive_env.render()
        
        print("交互式房屋布局环境已启动!")
        print("使用说明:")
        print("1. 点击选择房间")
        print("2. 使用方向键移动选中的房间")
        print("3. 使用Y/H键调整高度")
        print("4. 使用G/J键调整宽度")
        print("5. 使用T切换增加/减少模式")
        print("6. 按R键重置环境")
        print("7. 按Q键退出")
        plt.show()


if __name__ == "__main__":
    args = parse_args()

    # 创建环境
    config_id = normalize_config_id(args.config_id)
    config_path = get_config_path(config_id)
    print(f"当前读取配置: {config_path}")
    env, conf = make_env(config_path)
    env.live_render_interval = float(conf.get("RenderInterval", 0.5))
    env.live_render_show_metrics = bool(conf.get("MonitorImageAnnotations", False))
    env.live_render_show_total_area = bool(conf.get("MonitorImageAnnotations", False))
    env.live_render_show_summary_panel = False
    env.saved_image_show_metrics = bool(conf.get("SavedImageAnnotations", False))
    env.saved_image_show_total_area = bool(conf.get("SavedImageAnnotations", False))
    env.saved_image_show_summary_panel = False
    
    # 创建训练器
    runner = Trainer(
        agent_name=conf['Training']['agent_name'],
        muliple_agents=True,
        save_dir=RESULTS_DIR)
    
    if conf['Training']['train']:
        # 训练阶段统一切到无界面后端，避免渲染依赖前台窗口、IDE 或终端会话。
        try:
            plt.switch_backend("Agg")
        except Exception:
            matplotlib.use("Agg", force=True)

        resume_mode = normalize_resume_mode(conf['Training'].get('resume_mode', 'fresh'))
        if resume_mode == "resume" and conf['Training']['num_workers'] > 1:
            raise ValueError("真正续训当前仅支持单进程训练，请先将并行训练进程数设为 1。")

        if conf['Training']['num_workers'] > 1:
            # 多进程
            # 创建进程池
            processes = []
            args = (conf['Training']['episodes'], conf['Training']['max_steps'], conf['Training']['batch_size'],
                    conf['Training']['lr'], (None if int(conf['Training'].get('seed', -1)) < 0 else int(conf['Training'].get('seed', -1))), conf['Training']['ckpt_path'], conf['Render'],
                    int(conf.get('RenderEverySteps', 1)), conf['Training']['save_threshold'],
                    int(conf['Training'].get('final_model_save_interval', 1)),
                    int(conf['Training'].get('latest_layout_save_interval', 1)),
                    int(conf['Training'].get('threshold_save_interval', 1)),
                    config_path, conf.get('AlgorithmAdvanced', {}), resume_mode, )

            envs = [copy.deepcopy(env) for _ in range(conf['Training']['num_workers'])]
            runners = [copy.deepcopy(runner) for _ in range(conf['Training']['num_workers'])]
            argses = [args for _ in range(conf['Training']['num_workers'])]

            # 启动进程
            for i in range(conf['Training']['num_workers']):
                p = mp.Process(target=runners[i].train, args=(envs[i], ) + argses[i])
                p.start()
                processes.append(p)
            
            # 等待所有进程完成
            for process in processes:
                process.join()
                
            print("所有任务完成！")
        else:
            runner.train(
                env, 
                episodes=conf['Training']['episodes'], 
                max_iters=conf['Training']['max_steps'], 
                batch_size=conf['Training']['batch_size'],
                lr=conf['Training']['lr'],
                seed=None if int(conf['Training'].get('seed', -1)) < 0 else int(conf['Training'].get('seed', -1)),
                render_every_steps=int(conf.get('RenderEverySteps', 1)),
                save_threshold=conf['Training']['save_threshold'], 
                final_model_save_interval=int(conf['Training'].get('final_model_save_interval', 1)),
                latest_layout_save_interval=int(conf['Training'].get('latest_layout_save_interval', 1)),
                threshold_save_interval=int(conf['Training'].get('threshold_save_interval', 1)),
                render=conf['Render'],
                pretrain_ckpt_path=conf['Training']['ckpt_path'],
                config_path = config_path,
                algorithm_config=conf.get('AlgorithmAdvanced', {}),
                resume_mode=resume_mode,
            )
    else:
        # 测试
        runner.evaluate(env, max_iters=conf['Training']['max_steps'],
                        pretrain_ckpt_path=conf['Training']['ckpt_path'],
                        render=True)
    
