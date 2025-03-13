import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# 定义策略网络，用于输出动作的概率分布
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # 输入层到隐藏层的全连接层
        self.fc2 = nn.Linear(hidden_dim, action_dim)  # 隐藏层到输出层的全连接层

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 使用ReLU激活函数对隐藏层输出进行激活
        return F.softmax(self.fc2(x), dim=1)  # 使用softmax函数将输出层的值转换为概率分布


# 定义价值网络，用于输出状态的价值
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # 输入层到隐藏层的全连接层
        self.fc2 = nn.Linear(hidden_dim, 1)  # 隐藏层到输出层的全连接层

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 使用ReLU激活函数对隐藏层输出进行激活
        return self.fc2(x)  # 输出状态的价值


# 定义PPO算法类
class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        # 初始化策略网络和价值网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        # 定义优化器，用于更新策略网络和价值网络的参数
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        # 设置超参数
        self.gamma = gamma  # 折扣因子
        self.lmbda = lmbda  # 优势函数的通用优势估计参数
        self.epochs = epochs  # 每次更新时的训练轮数
        self.eps = eps  # PPO算法中的截断范围参数
        self.device = device  # 设备（CPU或GPU）

    # 定义策略网络的采样函数，根据当前状态选择动作
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)  # 将状态转换为张量
        probs = self.actor(state)  # 计算动作的概率分布
        action_dist = torch.distributions.Categorical(probs)  # 定义分类分布
        action = action_dist.sample()  # 从分布中采样动作
        return action.item()  # 返回动作的值

    # 定义PPO算法的更新函数
    def update(self, transition_dict):
        # 将数据转换为张量
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 计算时序差分误差和优势函数
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)  # 计算时序差分目标
        td_delta = td_target - self.critic(states)  # 计算时序差分误差
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)  # 计算优势函数

        # 计算旧策略的概率
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()  # 计算旧策略下动作的概率

        # 更新策略网络和价值网络
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))  # 计算新策略下动作的概率
            ratio = torch.exp(log_probs - old_log_probs)  # 计算概率比值
            surr1 = ratio * advantage  # 计算目标函数的第一个部分
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 计算目标函数的第二个部分
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # 计算策略网络的损失函数
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))  # 计算价值网络的损失函数
            self.actor_optimizer.zero_grad()  # 清空策略网络的梯度
            self.critic_optimizer.zero_grad()  # 清空价值网络的梯度
            actor_loss.backward()  # 计算策略网络的梯度
            critic_loss.backward()  # 计算价值网络的梯度
            self.actor_optimizer.step()  # 更新策略网络的参数
            self.critic_optimizer.step()  # 更新价值网络的参数


# 定义计算优势函数的函数
def compute_advantage(gamma, lmbda, td_delta):
    advantage_list = []  # 存储优势函数的列表
    advantage = 0.0  # 初始化优势函数
    # 从后往前计算优势函数
    for delta in reversed(td_delta):
        advantage = gamma * lmbda * advantage + delta.item()  # 计算优势函数
        advantage_list.append(advantage)  # 将优势函数添加到列表中
    advantage_list.reverse()  # 将列表反转，恢复顺序
    return torch.tensor(advantage_list, dtype=torch.float)  # 返回优势函数的张量


# 设置超参数
actor_lr = 1e-3  # 策略网络的学习率
critic_lr = 1e-2  # 价值网络的学习率
num_episodes = 500  # 训练的轮数
hidden_dim = 128  # 隐藏层的维度
gamma = 0.98  # 折扣因子
lmbda = 0.95  # 优势函数的通用优势估计参数
epochs = 10  # 每次更新时的训练轮数
eps = 0.2  # PPO算法中的截断范围参数
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 设置设备

# 创建环境
env_name = 'CartPole-v1'
env = gym.make(env_name)

# 初始化PPO算法
state_dim = env.observation_space.shape[0]  # 状态的维度
action_dim = env.action_space.n  # 动作的维度
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

# 训练过程
for episode in range(num_episodes):
    state = env.reset()[0]  # 重置环境，获取初始状态
    done = False  # 是否完成标志
    episode_return = 0  # 当前回合的回报
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}  # 存储数据的字典
    while not done:
        action = agent.take_action(state)  # 根据当前策略选择动作
        next_state, reward, terminated, truncated, _ = env.step(action)  # 执行动作，获取下一个状态、奖励等信息
        done = terminated or truncated  # 判断是否完成
        transition_dict['states'].append(state)  # 添加当前状态
        transition_dict['actions'].append(action)  # 添加当前动作
        transition_dict['next_states'].append(next_state)  # 添加下一个状态
        transition_dict['rewards'].append(reward)  # 添加奖励
        transition_dict['dones'].append(done)  # 添加是否完成标志
        state = next_state  # 更新当前状态
        episode_return += reward  # 累加回报
    agent.update(transition_dict)  # 更新策略网络和价值网络
    print(f"Episode: {episode+1}, Return: {episode_return}")  # 打印当前回合的信息
