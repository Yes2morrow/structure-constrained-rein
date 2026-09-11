import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np


class RewardNormalizer:
    def __init__(self, clip_range=10.0):
        self.clip_range = clip_range
        self.running_mean = 0.0
        self.running_variance = 0.0
        self.count = 0

    def normalize(self, rewards):
        # Update running mean and variance
        for reward in rewards:
            delta = reward - self.running_mean
            self.running_mean += delta / (self.count + 1)
            delta2 = reward - self.running_mean
            self.running_variance += delta * delta2
            self.count += 1
        
        # Calculate standard deviation
        std_dev = np.sqrt(self.running_variance / self.count)

        # Normalize rewards
        normalized_rewards = [(r - self.running_mean) / (std_dev + 1e-8) for r in rewards]

        # Clip the normalized rewards to avoid extreme values
        clipped_rewards = np.clip(normalized_rewards, -self.clip_range, self.clip_range)

        return clipped_rewards
    

class MAPPO:
    """ Multi-Agent PPO with Centralized Critic """
    def __init__(
        self,
        state_size_per_agent,
        global_state_size,
        action_size,
        hidden_size=128,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        ppo_epochs=4,
        batch_size=64,
        max_grad_norm=0.5,
        num_agents=4,
        update_step:int=1024,
        **kwargs
    ):
        self.state_size_per_agent = state_size_per_agent
        self.global_state_size = global_state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.num_agents = num_agents
        self.update_step = update_step
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reward_normalizer = RewardNormalizer(clip_range=float(kwargs.get("reward_clip_range", 8.0)))
        
        # Actor networks (one per agent)
        self.actors = []
        for _ in range(num_agents):
            actor = nn.Sequential(
                nn.Linear(state_size_per_agent, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_size)
            ).to(self.device)
            self.actors.append(actor)
        
        # Centralized critic network
        self.critic = nn.Sequential(
            nn.Linear(global_state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(self.device)

        # Optimizers
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=float(lr)) for actor in self.actors]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=float(lr))
        
        # Memory
        self.memory = {
            'states': [],
            'global_states': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'dones': [],
        }

    def act(self, states, global_state=None, training=True, info:dict=None, **kwargs):
        """Choose actions for all agents"""
        allow_act = info['allow_actions']  # 获取当前可执行的动作
        actions = []
        log_probs = []
        values = []
        
        # Get global value
        global_state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            global_value = self.critic(global_state_tensor).item()
        
        for i, state in enumerate(states):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.actors[i](state_tensor)
                mask = torch.zeros_like(logits, dtype=torch.bool, device=self.device)[0]
                mask[allow_act[i]] = 1
                logits[:, ~mask] = -torch.inf
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
            actions.append(action.item())
            log_probs.append(log_prob.item())
            values.append(global_value)  # All agents share the same global value
            
        return actions, log_probs, values
    
    def remember(self, states, global_state, actions, log_probs, values, rewards, dones):
        """Store experience in memory"""
        self.memory['states'].append(states)
        self.memory['global_states'].append(global_state)
        self.memory['actions'].append(actions)
        self.memory['log_probs'].append(log_probs)
        self.memory['values'].append(values)
        self.memory['rewards'].append(rewards)
        self.memory['dones'].append(dones)
    
    def clear_memory(self):
        """Clear the memory"""
        self.memory = {
            'states': [],
            'global_states': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'dones': [],
        }
    
    def compute_gae(self, next_value):
        """Compute Generalized Advantage Estimation using global rewards and dones"""
        # Convert to tensor
        rewards = torch.tensor(self.memory['rewards']).to(self.device)  # [T, num_agents]
        dones = torch.tensor(self.memory['dones']).to(self.device).float()  # [T, num_agents]
        values = torch.tensor(self.memory['values']).to(self.device)  # [T, num_agents]

        # Use global reward and global done
        global_rewards = rewards.mean(dim=1)  # [T]
        global_dones = dones  # [T]

        # Values are the same for all agents, so take first agent's value
        global_values = values[:, 0]  # [T]
        values = torch.cat([global_values, torch.tensor([next_value[0]]).to(self.device)])  # [T+1]

        advantages = []
        gae = 0
        for t in reversed(range(len(global_rewards))):
            delta = global_rewards[t] + self.gamma * values[t+1] * (1 - global_dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - global_dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages).to(self.device)
        returns = advantages + global_values
        return advantages, returns
    
    def update(self):
        """Update the model using PPO"""
        if len(self.memory['rewards']) < self.batch_size:
            return
            
        # Convert to tensors
        states = [torch.FloatTensor([s[i] for s in self.memory['states']]).to(self.device) 
                 for i in range(self.num_agents)]
        global_states = torch.FloatTensor(self.memory['global_states']).to(self.device)
        actions = [torch.LongTensor([a[i] for a in self.memory['actions']]).to(self.device) 
                  for i in range(self.num_agents)]
        old_log_probs = [torch.FloatTensor([lp[i] for lp in self.memory['log_probs']]).to(self.device) 
                        for i in range(self.num_agents)]
        
        # Compute advantages and returns
        next_value = self.memory['values'][-1]
        advantages, returns = self.compute_gae(next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 只归一化一次
        
        # PPO update
        for _ in range(self.ppo_epochs):
            # Get mini-batch indices
            indices = torch.randperm(len(global_states))
            
            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_global_states = global_states[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Update critic
                values = self.critic(batch_global_states).squeeze().float()
                critic_loss = F.mse_loss(values, batch_returns.float())
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                # Update actors
                for i in range(self.num_agents):
                    batch_states = states[i][batch_indices]
                    batch_actions = actions[i][batch_indices]
                    batch_old_log_probs = old_log_probs[i][batch_indices]
                    
                    logits = self.actors[i](batch_states)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()
                    
                    # Calculate ratios
                    ratios = torch.exp(new_log_probs - batch_old_log_probs)
                    
                    # Calculate surrogate losses
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    
                    # Calculate losses
                    actor_loss = -torch.min(surr1, surr2).mean()
                    loss = actor_loss - self.entropy_coef * entropy
                    
                    # Update actor
                    self.actor_optimizers[i].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.max_grad_norm)
                    self.actor_optimizers[i].step()
        
        # Clear memory
        self.clear_memory()
    
    def save(self, path_dir):
        """Save all models"""
        if os.path.isfile(path_dir):
            path_dir = os.path.dirname(path_dir)
        os.makedirs(path_dir, exist_ok=True)
        for i, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), os.path.join(path_dir, f"actor_{i}.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path_dir, "critic.pth"))
        torch.save(self.get_training_state(), os.path.join(path_dir, "training_state.pth"))
    
    def load(self, path_dir, load_training_state: bool = False):
        """Load all models"""
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(torch.load(os.path.join(path_dir, f"actor_{i}.pth")))
        
        self.critic.load_state_dict(torch.load(os.path.join(path_dir, "critic.pth")))
        if load_training_state:
            training_state_path = os.path.join(path_dir, "training_state.pth")
            if os.path.isfile(training_state_path):
                training_state = torch.load(training_state_path)
                self.load_training_state(training_state)

    def get_training_state(self) -> dict:
        """导出可用于真正续训的训练状态。"""
        return {
            "actor_optimizers": [optimizer.state_dict() for optimizer in self.actor_optimizers],
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "reward_normalizer": {
                "clip_range": self.reward_normalizer.clip_range,
                "running_mean": self.reward_normalizer.running_mean,
                "running_variance": self.reward_normalizer.running_variance,
                "count": self.reward_normalizer.count,
            },
        }

    def load_training_state(self, state: dict) -> None:
        """恢复真正续训需要的优化器和归一化器状态。"""
        actor_optimizer_states = state.get("actor_optimizers", [])
        for optimizer, optimizer_state in zip(self.actor_optimizers, actor_optimizer_states):
            optimizer.load_state_dict(optimizer_state)
        critic_optimizer_state = state.get("critic_optimizer")
        if critic_optimizer_state:
            self.critic_optimizer.load_state_dict(critic_optimizer_state)

        reward_state = state.get("reward_normalizer", {})
        if reward_state:
            self.reward_normalizer.clip_range = reward_state.get("clip_range", self.reward_normalizer.clip_range)
            self.reward_normalizer.running_mean = reward_state.get("running_mean", self.reward_normalizer.running_mean)
            self.reward_normalizer.running_variance = reward_state.get("running_variance", self.reward_normalizer.running_variance)
            self.reward_normalizer.count = reward_state.get("count", self.reward_normalizer.count)
    
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
            
            # 选择动作
            action, logpred, value = self.act(state, global_state, training=True, info=info)
            
            # 执行动作
            next_state, reward_list, done, info = env.step(action)

            # 记录奖励
            episode_total_reward += sum(reward_list)
            episode_steps_reward.append(sum(reward_list))
            r_sum = self.reward_normalizer.normalize(reward_list)
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
            if kwargs.get("split_done", False):
                self.remember(state, global_state, action, logpred, value, r_sum, info['done_list'])
            else:
                self.remember(state, global_state, action, logpred, value, r_sum, done)
            
            # 训练
            self.update()
                
            # 更新状态
            state = next_state
            total_step += 1
            
            # 可视化
            if kwargs.get("render", False) and total_step % render_every_steps == 0:
                env.render(render=False)
            
            is_done = done or (total_step > max_iters)
            if is_done: break
            
        self.clear_memory()
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
        
    def evaluate(self, env, max_iters:int=256, **kwargs):
        """ 测试一个回合 """
        # 记录内容
        episode_total_reward = 0.0  # 总奖励
        # 重置环境
        state, info = env.reset(seed=kwargs.get("seed", None))
        done  = False
        total_step = 1
        while True:
            print("inner step: {}, reward: {:.3f}".format(total_step, episode_total_reward/(total_step+1)),
                  end="\r")
            # 获取全局状态
            global_state = state.flatten()
            
            # 选择动作
            action, logpred, value = self.act(state, global_state, training=False, info=info)
            
            # 执行动作
            next_state, reward_list, done, info = env.step(action)
 
            # 更新状态
            state = next_state
            total_step += 1
            
            # 可视化
            if kwargs.get("render", False):
                env.render(render=False)
            
            is_done = done or (total_step > max_iters)
            if is_done: break
            
        self.clear_memory()
        return env
    
