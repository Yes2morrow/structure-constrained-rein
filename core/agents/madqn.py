# madqn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import gym
import os


class DQNNetwork(nn.Module):
    """深度Q网络 - 基于局部观察"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)


class MADQNBuffer:
    """多智能体DQN经验回放缓冲区"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, local_states, actions, rewards, next_state, next_local_states, dones):
        """存储经验"""
        experience = (state, local_states, actions, rewards, next_state, next_local_states, dones)
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        """随机采样"""
        if len(self.buffer) < batch_size:
            return None
            
        batch = random.sample(self.buffer, batch_size)
        state_batch, local_states_batch, actions_batch, rewards_batch, next_state_batch, next_local_states_batch, dones_batch = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(state_batch)),
            torch.FloatTensor(np.array(local_states_batch)),
            torch.LongTensor(np.array(actions_batch)),
            torch.FloatTensor(np.array(rewards_batch)),
            torch.FloatTensor(np.array(next_state_batch)),
            torch.FloatTensor(np.array(next_local_states_batch)),
            torch.BoolTensor(np.array(dones_batch))
        )
        
    def __len__(self):
        return len(self.buffer)


class MADQN:
    """多智能体深度Q网络算法"""
    
    def __init__(self, state_dim, local_state_dim, action_dim, num_agents,
                 hidden_dim=128, lr=0.0001, gamma=0.99, tau=0.01,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_capacity=10000, batch_size=32, device='cpu'):
        
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.action_dim = action_dim
        
        # 探索参数
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 为每个智能体创建DQN网络和目标网络
        self.q_networks = [DQNNetwork(local_state_dim, action_dim, hidden_dim).to(self.device) 
                          for _ in range(num_agents)]
        self.target_networks = [DQNNetwork(local_state_dim, action_dim, hidden_dim).to(self.device) 
                               for _ in range(num_agents)]
        
        # 初始化目标网络参数
        for i in range(num_agents):
            self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())
        
        # 优化器
        self.optimizers = [torch.optim.Adam(self.q_networks[i].parameters(), lr=lr) 
                          for i in range(num_agents)]
        
        # 经验回放缓冲区
        self.buffer = MADQNBuffer(buffer_capacity)
        
    def select_action(self, local_states, training=True):
        """选择动作 - 基于局部观察和ε-greedy策略"""
        actions = []
        
        for i in range(self.num_agents):
            local_state = local_states[i]
            
            if len(local_state.shape) == 1:
                local_state = local_state.reshape(1, -1)
            local_state_tensor = torch.FloatTensor(local_state).to(self.device)
            
            # ε-greedy策略
            if training and random.random() < self.epsilon:
                # 探索：随机选择动作
                action = random.randint(0, self.action_dim - 1)
            else:
                # 利用：选择Q值最大的动作
                with torch.no_grad():
                    q_values = self.q_networks[i](local_state_tensor)
                    action = q_values.argmax().item()
            
            actions.append(action)
            
        return actions
    
    def update(self):
        """更新Q网络"""
        if len(self.buffer) < self.batch_size:
            return
            
        # 从缓冲区采样
        batch = self.buffer.sample(self.batch_size)
        if batch is None:
            return
            
        state_batch, local_states_batch, actions_batch, rewards_batch, next_state_batch, next_local_states_batch, dones_batch = batch
        
        # 移动到设备
        state_batch = state_batch.to(self.device)
        local_states_batch = local_states_batch.to(self.device)
        actions_batch = actions_batch.to(self.device)
        rewards_batch = rewards_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        next_local_states_batch = next_local_states_batch.to(self.device)
        dones_batch = dones_batch.to(self.device)
        
        # 为每个智能体更新网络
        for i in range(self.num_agents):
            # 当前Q值
            current_q_values = self.q_networks[i](local_states_batch[:, i, :])
            current_q_values = current_q_values.gather(1, actions_batch[:, i].unsqueeze(1))
            
            # 目标Q值 - 修复维度不匹配问题
            with torch.no_grad():
                next_q_values = self.target_networks[i](next_local_states_batch[:, i, :])
                max_next_q_values = next_q_values.max(1)[0]
                
                # 确保所有张量维度一致
                agent_dones = dones_batch[:, i]  # 获取第i个智能体的done标志
                agent_rewards = rewards_batch[:, i]  # 获取第i个智能体的奖励
                
                target_q_values = agent_rewards + self.gamma * max_next_q_values * (~agent_dones)
                target_q_values = target_q_values.unsqueeze(1)
            
            # 计算损失
            loss = F.mse_loss(current_q_values, target_q_values)
            
            # 反向传播
            self.optimizers[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_networks[i].parameters(), max_norm=1.0)
            self.optimizers[i].step()
        
        # 衰减探索率
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def soft_update_target_networks(self):
        """软更新目标网络"""
        for i in range(self.num_agents):
            for target_param, param in zip(self.target_networks[i].parameters(), self.q_networks[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, checkpoint_dir):
        """保存模型"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        for i in range(self.num_agents):
            torch.save(self.q_networks[i].state_dict(), f"{checkpoint_dir}/madqn_q_network_{i}.pth")
            torch.save(self.target_networks[i].state_dict(), f"{checkpoint_dir}/madqn_target_network_{i}.pth")
        torch.save(self.get_training_state(), os.path.join(checkpoint_dir, "training_state.pth"))
    
    def load(self, checkpoint_dir, load_training_state: bool = False):
        """加载模型"""
        for i in range(self.num_agents):
            self.q_networks[i].load_state_dict(
                torch.load(f"{checkpoint_dir}/madqn_q_network_{i}.pth", map_location=self.device)
            )
            self.target_networks[i].load_state_dict(
                torch.load(f"{checkpoint_dir}/madqn_target_network_{i}.pth", map_location=self.device)
            )
        if load_training_state:
            training_state_path = os.path.join(checkpoint_dir, "training_state.pth")
            if os.path.isfile(training_state_path):
                training_state = torch.load(training_state_path, map_location=self.device)
                self.load_training_state(training_state)

    def get_training_state(self) -> dict:
        """导出真正续训需要的训练状态。"""
        return {
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "buffer": self.buffer,
            "epsilon": self.epsilon,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
        }

    def load_training_state(self, state: dict) -> None:
        """恢复真正续训需要的优化器、经验池和探索率。"""
        optimizer_states = state.get("optimizers", [])
        for optimizer, optimizer_state in zip(self.optimizers, optimizer_states):
            optimizer.load_state_dict(optimizer_state)
        if "buffer" in state:
            self.buffer = state["buffer"]
        self.epsilon = state.get("epsilon", self.epsilon)
        self.epsilon_end = state.get("epsilon_end", self.epsilon_end)
        self.epsilon_decay = state.get("epsilon_decay", self.epsilon_decay)
            
    def learn(self, env:gym.Env, max_iters:int=-1, **kwargs) -> dict:
        """ 训练一个回合 """
        # 记录内容
        episode_total_reward = 0.0  # 总奖励
        episode_steps_reward = []  # 每一步的奖励
        episode_agent_reward = {r: [] for r in range(kwargs.get("num_agents", 4))}  # 每个智能体的奖励
        episode_reward_components = {}
        reward_component_steps = 0
        # 重置环境
        state, info = env.reset(seed=kwargs.get("seed", None), rnd=kwargs.get("rnd", False))
        done  = False
        total_step = 1
        render_every_steps = max(1, int(kwargs.get("render_every_steps", 1)))
        while True:
            print("inner step: {}, reward: {:.3f}".format(total_step, episode_total_reward/(total_step+1)),
                  end="\r")
            # 获取全局状态
            global_state = state.flatten()
            
            # 获取局部观察
            local_observations = []
            for i in range(self.num_agents):
                local_observations.append(state[i])
            
            # 选择动作
            action = self.select_action(local_observations, training=True)

            # 执行动作
            next_state, reward_list, done, info = env.step(action)

            # 记录奖励
            episode_total_reward += sum(reward_list)
            episode_steps_reward.append(sum(reward_list))
            for agent_id, r in enumerate(reward_list):
                episode_agent_reward[agent_id].append(r)
            step_reward_components = info.get("reward_components", [])
            if step_reward_components:
                reward_component_steps += 1
                for component_dict in step_reward_components:
                    for component_name, component_value in component_dict.items():
                        episode_reward_components[component_name] = (
                            episode_reward_components.get(component_name, 0.0) + float(component_value)
                        )
            
            # 存储经验
            self.buffer.push(
                state,
                state,
                np.array(action),
                np.array(reward_list),
                next_state,
                next_state,
                np.array([done] * self.num_agents),
            )
            state = next_state

            # 训练
            self.update()

            total_step += 1
            
            # 可视化
            if kwargs.get("render", False) and total_step % render_every_steps == 0:
                env.render(render=False)
            
            is_done = done or (total_step > max_iters)
            if is_done: break
            
        averaged_reward_components = {
            f"{component_name}_reward": component_value / max(reward_component_steps, 1)
            for component_name, component_value in episode_reward_components.items()
        }
        return {
            "episode_total_reward": episode_total_reward,
            "episode_steps_reward": episode_steps_reward,
            "episode_agent_reward": episode_agent_reward,
            "total_step": total_step,
            "episode_reward_components": averaged_reward_components,
        }
