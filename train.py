from ppo import PPO
from ParkingSimulator import ParkingSimulator

model = PPO(ParkingSimulator())
model.learn(1_00_000)