from ParkingSimulator import ParkingSimulator
from network import FeedForwardNN
import torch

env = ParkingSimulator()
actor = FeedForwardNN(env.observation_space.shape[0], env.action_space.shape[0], is_actor=True)

actor.load_state_dict(torch.load('ppo_actor.pth'))

while True:
    obs = env.reset()
    print("restart")
    done = False
    
    while not done:
        env.render()
        action = actor(obs).detach().numpy()
        obs, reward, done, _ = env.step(action)