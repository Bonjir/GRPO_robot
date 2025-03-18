import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 与训练时相同的策略网络定义（连续动作策略网络）
class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc_mu = torch.nn.Linear(128, action_dim)
        self.fc_std = torch.nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std
    
def take_action(state,policy_net, device):
        state = torch.tensor(state, dtype=torch.float).to(device)
        mu, sigma = policy_net(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return [action.item()]

def test_model(model_path, num_episodes=10):
    env = gym.make("Pendulum-v1", render_mode="human")
    state_dim = env.observation_space.shape[0]  # 3
    n_actions = env.action_space.shape[0]       # 1
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 初始化策略网络并加载模型参数
    policy = PolicyNetContinuous(state_dim, n_actions).to(device)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()  # 设置为评估模式

    episode_rewards = []
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            # 将 state 转为 tensor，形状 [1, state_dim]
            action = take_action(state,policy,device=device)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            # 可视化环境（确保你有图形界面支持）
            env.render()
        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}: Total Reward = {total_reward}, Steps = {steps}")
    env.close()

    # 绘制累计reward曲线
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, num_episodes + 1), episode_rewards, marker='o', linestyle='-')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Test Episodes Reward Curve on Pendulum-v1")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # 请确保模型文件路径正确
    test_model("./weights/grpo_pendulum_policy_update_final.pth", num_episodes=10)
