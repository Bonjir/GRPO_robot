import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import time
# 自定义回调函数，用于在平均奖励达到目标时停止训练
class RewardThresholdCallback(BaseCallback):
    def __init__(self, reward_threshold, verbose=0):
        super(RewardThresholdCallback, self).__init__(verbose)
        self.reward_threshold = reward_threshold
        self.count = 0

    def _on_step(self) -> bool:
        # 获取最近100个episode的平均奖励
        if len(self.locals['infos']) > 0 and 'episode' in self.locals['infos'][0]:
            episode_rewards = [info['episode']['r'] for info in self.locals['infos'] if 'episode' in info]
            if len(episode_rewards) > 0:
                mean_reward = sum(episode_rewards) / len(episode_rewards)
                if self.verbose > 0:
                    print(f"平均奖励: {mean_reward:.2f}")
                # 如果平均奖励达到或超过目标值，停止训练
                if mean_reward >= self.reward_threshold:
                    self.count += 1
                    print(f"平均奖励: {mean_reward:.2f}")
                    if self.count >= 5:
                        print(f"目标平均奖励 {self.reward_threshold} 达成，停止训练。")
                        return False
                    

        return True

# 创建 CartPole-v1 环境
env = gym.make("CartPole-v1")

# 初始化 PPO 模型，使用多层感知机策略
model = PPO("MlpPolicy", env, verbose=1)

# 创建回调函数实例，设定奖励阈值为 495
callback = RewardThresholdCallback(reward_threshold=495)

# 开始训练，训练过程中使用回调函数监控平均奖励
start = time.time()
model.learn(total_timesteps=100000, callback=callback)
print("used_time(s): ", time.time() - start)

# 保存训练好的模型
model.save("ppo_cartpole_v1")

# 关闭环境
env.close()



