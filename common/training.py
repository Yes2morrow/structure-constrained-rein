"""训练入口共享的构建逻辑。"""

from __future__ import annotations

from core.agents import MAPPO


def build_mapppo_agent(env, config: dict, reset_seed=None) -> MAPPO:
    """按配置构建 MAPPO 智能体；reset_seed 用于探测状态维度时重置环境。"""
    training = config["Training"]
    advanced = config.get("AlgorithmAdvanced", {})
    state, _ = env.reset(seed=reset_seed)
    return MAPPO(
        state_size_per_agent=state.shape[1],
        global_state_size=state.size,
        action_size=int(env.action_space.nvec[0]),
        num_agents=env.num_agents,
        hidden_size=int(advanced.get("hidden_size", 256)),
        lr=float(training.get("lr", 3e-4)),
        gamma=float(advanced.get("gamma", 0.99)),
        gae_lambda=float(advanced.get("gae_lambda", 0.95)),
        clip_epsilon=float(advanced.get("clip_epsilon", 0.2)),
        entropy_coef=float(advanced.get("entropy_coef", 0.01)),
        ppo_epochs=int(advanced.get("ppo_epochs", 5)),
        batch_size=int(training.get("batch_size", 64)),
        max_grad_norm=float(advanced.get("max_grad_norm", 0.5)),
        reward_clip_range=float(advanced.get("reward_clip_range", 8.0)),
    )
