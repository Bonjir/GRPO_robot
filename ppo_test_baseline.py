import gymnasium as gym
from stable_baselines3 import PPO

model = PPO.load("ppo_cartpole_v1")
env = gym.make('CartPole-v1',render_mode="human")
obs = env.reset()[0]
dones = False
while not dones:
    action, _states = model.predict(obs)
    obs, rewards, dones, info,_ = env.step(action)
    env.render()