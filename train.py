from ppo import PPO
from ParkingSimulator import ParkingSimulator

model = PPO(ParkingSimulator())
while True:
    model.learn(100_000)