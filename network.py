import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim, is_actor) -> None:
        super(FeedForwardNN, self).__init__()

        self.is_actor = is_actor

        # self.layer1 = nn.Linear(in_dim, 1024)
        # self.layer2 = nn.Linear(1024, 512)
        # self.layer3 = nn.Linear(512, 256)
        # self.layer4 = nn.Linear(256, 64)
        # self.layer5 = nn.Linear(64, out_dim)
        
        self.layer1 = nn.Linear(in_dim, out_dim)


    def forward(self, obs):
        # Convert obseration to tensor if it's a numpy array
        if isinstance(obs, np.ndarray) or isinstance(obs, np.float):
            obs = torch.tensor(obs, dtype=torch.float)

        ###
        # activation1 = F.relu(self.layer1(obs))
        # activation2 = F.relu(self.layer2(activation1))
        # output = self.layer3(activation2) #This works well for pendulum v0. 
        ###

        # activation1 = F.relu(self.layer1(obs))
        # activation2 = F.relu(self.layer2(activation1))
        # activation3 = F.relu(self.layer3(activation2))
        # activation4 = F.relu(self.layer4(activation3))
        # if self.is_actor:
        #     output = torch.tanh(self.layer5(activation4))
        # else:
        #     output = self.layer5(activation4)

        activation1 = F.relu(self.layer1(obs))
        if self.is_actor:
            output = torch.tanh(activation1)
        else:
            output = activation1

        return output

