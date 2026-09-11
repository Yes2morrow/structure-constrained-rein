import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class QNetwork(nn.Module):
    """单个智能体的Q网络"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
    def forward(self, x):
        return self.network(x)


class HyperNetwork(nn.Module):
    """超网络，用于生成混合网络的权重"""
    def __init__(self, num_agents, state_size, hyper_hidden_size, mixing_hidden_size):
        super(HyperNetwork, self).__init__()
        self.num_agents = num_agents
        self.state_size = state_size
        
        # 生成混合网络第一层权重的超网络
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_size, hyper_hidden_size),
            nn.ReLU(),
            nn.Linear(hyper_hidden_size, num_agents * mixing_hidden_size)
        )
        self.hyper_w1_out_features = num_agents * mixing_hidden_size
        
        # 生成混合网络第一层偏置的超网络
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_size, mixing_hidden_size)
        )
        self.hyper_b1_out_feuatures = mixing_hidden_size
        
        # 生成混合网络第二层权重的超网络
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_size, hyper_hidden_size),
            nn.ReLU(),
            nn.Linear(hyper_hidden_size, mixing_hidden_size)
        )
        self.hyper_w2_out_features = mixing_hidden_size
        
        # 生成混合网络第二层偏置的超网络
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_size, mixing_hidden_size),
            nn.ReLU(),
            nn.Linear(mixing_hidden_size, 1)
        )
        
    
    def forward(self, state):
        # 生成混合网络的权重和偏置
        w1 = self.hyper_w1(state).view(-1, self.num_agents, self.hyper_w1_out_features // self.num_agents)
        w1 = torch.abs(w1)  # 确保权重为正
        
        b1 = self.hyper_b1(state).view(-1, 1, self.hyper_b1_out_feuatures)
        
        w2 = torch.abs(self.hyper_w2(state)).view(-1, self.hyper_w2_out_features, 1)
        b2 = self.hyper_b2(state).view(-1, 1, 1)
        
        return w1, b1, w2, b2


class MixingNetwork(nn.Module):
    """混合网络，将个体Q值混合成全局Q值"""
    def __init__(self, num_agents, state_size, hyper_hidden_size=64, mixing_hidden_size=32):
        super(MixingNetwork, self).__init__()
        self.num_agents = num_agents
        self.hyper_net = HyperNetwork(num_agents, state_size, hyper_hidden_size, mixing_hidden_size)
    
    def forward(self, agent_qs, state):
        """
        agent_qs: [batch_size, num_agents] 每个智能体的Q值
        state: [batch_size, state_size] 全局状态
        """
        batch_size = agent_qs.size(0)
        
        # 通过超网络生成权重
        w1, b1, w2, b2 = self.hyper_net(state)
        
        # 第一层混合
        agent_qs = agent_qs.view(batch_size, 1, self.num_agents)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        
        # 第二层混合
        q_total = torch.bmm(hidden, w2) + b2
        return q_total.view(batch_size, 1)


class QMIX:
    """QMIX算法实现"""
    def __init__(
        self,
        state_size_per_agent,
        global_state_size,
        action_size,
        num_agents=4,
        hidden_size=128,
        mixing_hidden_size=32,
        hyper_hidden_size=64,
        lr=1e-4,
        gamma=0.99,
        tau=0.01,
        buffer_size=10000,
        batch_size=32,
        update_frequency=100,
        epsilon=1.0,
        epsilon_decay=0.999,
        min_epsilon=0.01,
        **kwargs
    ):
        self.state_size_per_agent = state_size_per_agent
        self.global_state_size = global_state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建Q网络和目标Q网络
        self.q_networks = []
        self.target_q_networks = []
        
        for i in range(num_agents):
            q_net = QNetwork(state_size_per_agent, action_size, hidden_size).to(self.device)
            target_q_net = QNetwork(state_size_per_agent, action_size, hidden_size).to(self.device)
            
            # 目标网络初始化为与在线网络相同
            target_q_net.load_state_dict(q_net.state_dict())
            
            self.q_networks.append(q_net)
            self.target_q_networks.append(target_q_net)
        
        # 创建混合网络和目标混合网络
        self.mixing_net = MixingNetwork(num_agents, global_state_size, hyper_hidden_size, mixing_hidden_size).to(self.device)
        self.target_mixing_net = MixingNetwork(num_agents, global_state_size, hyper_hidden_size, mixing_hidden_size).to(self.device)
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())
        
        # 优化器
        self.optimizers = []
        for i in range(num_agents):
            optimizer = optim.Adam(self.q_networks[i].parameters(), lr=lr)
            self.optimizers.append(optimizer)
        
        self.mixing_optimizer = optim.Adam(self.mixing_net.parameters(), lr=lr)
        
        # 经验回放缓冲区
        self.buffer = {
            'states': [],
            'global_states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'next_global_states': [],
            'dones': []
        }
        self.buffer_size = buffer_size
        self.step_count = 0
        
        # 正交初始化
        self.orthogonal_init()

    def orthogonal_init(self):
        """正交初始化"""
        for q_net in self.q_networks:
            for m in q_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
                    nn.init.constant_(m.bias, 0.1)
        
        for m in self.mixing_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def act(self, states, global_state=None, training=True, info:dict=None, epsilon=0.1, **kwargs):
        """选择动作"""
        actions = []
        q_values_list = []
        
        allow_act = info.get('allow_actions', [list(range(self.action_size))] * self.num_agents)
        
        for i, state in enumerate(states):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_networks[i](state_tensor)
                
                # 应用动作掩码
                mask = torch.ones_like(q_values) * -1e10
                mask[0, allow_act[i]] = 0
                q_values = q_values + mask
                
                if training and np.random.random() < epsilon:
                    # 探索：从允许的动作中随机选择
                    action = np.random.choice(allow_act[i])
                else:
                    # 利用：选择Q值最大的动作
                    action = q_values.argmax(dim=1).item()
                
            actions.append(action)
            q_values_list.append(q_values.max().item())
        
        return actions, q_values_list, [0] * self.num_agents  # 为了兼容性，返回空log_probs和values

    def remember(self, states, global_state, actions, rewards, next_states, next_global_state, dones):
        """存储经验"""
        if len(self.buffer['states']) >= self.buffer_size:
            # 如果缓冲区已满，移除最旧的经验
            for key in self.buffer:
                self.buffer[key].pop(0)
        
        self.buffer['states'].append(states)
        self.buffer['global_states'].append(global_state)
        self.buffer['actions'].append(actions)
        self.buffer['rewards'].append(rewards)
        self.buffer['next_states'].append(next_states)
        self.buffer['next_global_states'].append(next_global_state)
        self.buffer['dones'].append(dones)

    def clear_memory(self):
        """清空缓冲区（QMIX不需要此方法，但为了兼容性保留）"""
        pass

    def update(self):
        """更新网络"""
        if len(self.buffer['states']) < self.batch_size:
            return
        
        self.step_count += 1
        
        # 从缓冲区采样
        indices = np.random.choice(len(self.buffer['states']), self.batch_size, replace=False)
        
        batch_states = [self.buffer['states'][i] for i in indices]
        batch_global_states = [self.buffer['global_states'][i] for i in indices]
        batch_actions = [self.buffer['actions'][i] for i in indices]
        batch_rewards = [self.buffer['rewards'][i] for i in indices]
        batch_next_states = [self.buffer['next_states'][i] for i in indices]
        batch_next_global_states = [self.buffer['next_global_states'][i] for i in indices]
        batch_dones = [self.buffer['dones'][i] for i in indices]
        
        # 转换为张量
        states_tensor = torch.FloatTensor(batch_states).to(self.device)  # [batch, num_agents, state_size]
        global_states_tensor = torch.FloatTensor(batch_global_states).to(self.device)  # [batch, global_state_size]
        actions_tensor = torch.LongTensor(batch_actions).to(self.device)  # [batch, num_agents]
        rewards_tensor = torch.FloatTensor(batch_rewards).to(self.device)  # [batch]
        next_states_tensor = torch.FloatTensor(batch_next_states).to(self.device)  # [batch, num_agents, state_size]
        next_global_states_tensor = torch.FloatTensor(batch_next_global_states).to(self.device)  # [batch, global_state_size]
        dones_tensor = torch.BoolTensor(batch_dones).to(self.device)  # [batch]
        
        # 计算当前Q值
        current_q_values = []
        for i in range(self.num_agents):
            agent_states = states_tensor[:, i, :]
            agent_actions = actions_tensor[:, i].unsqueeze(1)
            q_values = self.q_networks[i](agent_states)
            current_q = q_values.gather(1, agent_actions)
            current_q_values.append(current_q)
        
        current_q_values = torch.stack(current_q_values, dim=1).squeeze(-1)  # [batch, num_agents]
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = []
            for i in range(self.num_agents):
                agent_next_states = next_states_tensor[:, i, :]
                next_q = self.target_q_networks[i](agent_next_states).max(dim=1)[0]
                next_q_values.append(next_q)
            
            next_q_values = torch.stack(next_q_values, dim=1)  # [batch, num_agents]
            
            # 通过目标混合网络计算全局目标Q值
            target_global_q = self.target_mixing_net(next_q_values, next_global_states_tensor)
            target_q = rewards_tensor.unsqueeze(1) + self.gamma * target_global_q * (~dones_tensor).unsqueeze(1)
        
        # 通过混合网络计算全局当前Q值
        current_global_q = self.mixing_net(current_q_values, global_states_tensor)
        
        # 计算损失
        loss = F.mse_loss(current_global_q, target_q)
        
        # 反向传播
        self.mixing_optimizer.zero_grad()
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.mixing_net.parameters(), 10.0)
        for q_net in self.q_networks:
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
        
        # 更新参数
        self.mixing_optimizer.step()
        for optimizer in self.optimizers:
            optimizer.step()
        
        # 定期更新目标网络
        if self.step_count % self.update_frequency == 0:
            self.update_target_networks()

    def update_target_networks(self):
        """软更新目标网络"""
        for i in range(self.num_agents):
            for target_param, param in zip(self.target_q_networks[i].parameters(), 
                                         self.q_networks[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_mixing_net.parameters(), 
                                     self.mixing_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path_dir):
        """保存模型"""
        if os.path.isfile(path_dir):
            path_dir = os.path.dirname(path_dir)
        os.makedirs(path_dir, exist_ok=True)
        for i, q_net in enumerate(self.q_networks):
            torch.save(q_net.state_dict(), os.path.join(path_dir, f"q_network_{i}.pth"))
        
        torch.save(self.mixing_net.state_dict(), os.path.join(path_dir, "mixing_network.pth"))
        torch.save(self.get_training_state(), os.path.join(path_dir, "training_state.pth"))

    def load(self, path_dir, load_training_state: bool = False):
        """加载模型"""
        for i, q_net in enumerate(self.q_networks):
            q_net.load_state_dict(torch.load(os.path.join(path_dir, f"q_network_{i}.pth")))
        
        self.mixing_net.load_state_dict(torch.load(os.path.join(path_dir, "mixing_network.pth")))
        
        # 同时更新目标网络
        for i in range(self.num_agents):
            self.target_q_networks[i].load_state_dict(self.q_networks[i].state_dict())
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())
        if load_training_state:
            training_state_path = os.path.join(path_dir, "training_state.pth")
            if os.path.isfile(training_state_path):
                training_state = torch.load(training_state_path)
                self.load_training_state(training_state)

    def get_training_state(self) -> dict:
        """导出真正续训所需的训练状态。"""
        return {
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "mixing_optimizer": self.mixing_optimizer.state_dict(),
            "buffer": self.buffer,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
        }

    def load_training_state(self, state: dict) -> None:
        """恢复真正续训需要的优化器、经验池和探索状态。"""
        optimizer_states = state.get("optimizers", [])
        for optimizer, optimizer_state in zip(self.optimizers, optimizer_states):
            optimizer.load_state_dict(optimizer_state)
        mixing_optimizer_state = state.get("mixing_optimizer")
        if mixing_optimizer_state:
            self.mixing_optimizer.load_state_dict(mixing_optimizer_state)
        if "buffer" in state:
            self.buffer = state["buffer"]
        self.initial_epsilon = state.get("initial_epsilon", self.initial_epsilon)
        self.epsilon_decay = state.get("epsilon_decay", self.epsilon_decay)
        self.min_epsilon = state.get("min_epsilon", self.min_epsilon)

    def learn(self, env, max_iters:int=-1, **kwargs) -> dict:
        """训练一个回合"""
        episode_total_reward = 0.0
        episode_steps_reward = []
        episode_agent_reward = {r: [] for r in range(self.num_agents)}
        episode_reward_components = {}
        reward_component_steps = 0
        
        state, info = env.reset(seed=kwargs.get("seed", None))
        done = False
        total_step = 1
        epsilon = kwargs.get("epsilon", self.initial_epsilon)
        epsilon_decay = kwargs.get("epsilon_decay", self.epsilon_decay)
        min_epsilon = kwargs.get("min_epsilon", self.min_epsilon)
        render_every_steps = max(1, int(kwargs.get("render_every_steps", 1)))
        
        while True:
            print("inner step: {}, reward: {:.3f}".format(total_step, episode_total_reward/(total_step+1)),
                  end="\r")
            # 获取全局状态
            global_state = state.flatten()
            
            # 选择动作（带探索）
            action, _, _ = self.act(state, global_state, training=True, 
                                           info=info, epsilon=epsilon)
            
            # 执行动作
            next_state, reward_list, done, info = env.step(action)
            
            # 记录奖励
            r_sum = sum(reward_list)
            episode_total_reward += r_sum
            episode_steps_reward.append(r_sum)
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
            next_global_state = next_state.flatten()
            self.remember(state, global_state, action, r_sum, next_state, next_global_state, done)
            
            # 更新网络
            self.update()
            
            # 衰减探索率
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            
            # 更新状态
            state = next_state
            total_step += 1
            
            # 可视化
            if kwargs.get("render", False) and total_step % render_every_steps == 0:
                env.render(render=False)
            
            is_done = done or (total_step > max_iters)
            if is_done: 
                break
        
        averaged_reward_components = {
            f"{component_name}_reward": component_value / max(reward_component_steps, 1)
            for component_name, component_value in episode_reward_components.items()
        }
        return {
            "episode_total_reward": episode_total_reward,
            "episode_steps_reward": episode_steps_reward,
            "episode_agent_reward": episode_agent_reward,
            "total_step": total_step,
            "epsilon": epsilon,
            "episode_reward_components": averaged_reward_components,
        }

    def evaluate(self, env, max_iters:int=256, **kwargs):
        """测试一个回合"""
        episode_total_reward = 0.0
        
        state, info = env.reset(seed=kwargs.get("seed", None))
        done = False
        total_step = 1
        
        while True:
            global_state = state.flatten()
            
            # 选择动作（无探索）
            action, _, _ = self.act(state, global_state, training=False, 
                                  info=info, epsilon=0.0)
            
            # 执行动作
            next_state, reward_list, done, info = env.step(action)
            
            # 更新状态
            state = next_state
            total_step += 1
            
            # 可视化
            if kwargs.get("render", False):
                env.render(render=False)
            
            is_done = done or (total_step > max_iters)
            if is_done: 
                break
        
        return env
