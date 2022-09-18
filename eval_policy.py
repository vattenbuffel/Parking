import gym
from network import FeedForwardNN
import torch

env = gym.make('Pendulum-v0')
actor = FeedForwardNN(env.observation_space.shape[0], env.action_space.shape[0])

actor.load_state_dict(torch.load('ppo_actor.pth'))

while True:
    obs = env.reset()
    print("restart")
    done = False
    
    while not done:
        env.render()
        action = actor(obs).detach().numpy()
        obs, reward, done, _ = env.step(action)