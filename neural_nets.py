"""
    File contains the neural network architecture for the agent actor and critic
"""

import torch
from torch import nn, no_grad
import torch.nn.functional as F
import numpy as np

ACTOR_HL1 = 40 
ACTOR_HL2 = 30 
CRITIC_HL1 = 40 
CRITIC_HL2 = 30 

class Actor_NN(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Actor_NN, self).__init__()
        self.l1 = nn.Linear(state_dim, ACTOR_HL1)
        self.l1.bias.data = torch.nn.Parameter(torch.zeros(ACTOR_HL1))
        nn.init.xavier_uniform_(self.l1.weight.data)
        
        #self.bn1 = nn.LayerNorm(ACTOR_HL1)
        
        self.l2 = nn.Linear(ACTOR_HL1, ACTOR_HL2)
        self.l2.bias.data = torch.nn.Parameter(torch.zeros(ACTOR_HL2))
        nn.init.xavier_uniform_(self.l2.weight.data)
        
        #self.bn1 = nn.LayerNorm(ACTOR_HL2)
        
        self.l3 = nn.Linear(ACTOR_HL2, action_dim)
        self.l3.bias.data = torch.nn.Parameter(torch.zeros(action_dim))
        nn.init.xavier_uniform_(self.l3.weight.data)

        
    def forward(self, state):        
        x = self.l1(state)
        #x = self.bn1(x)
        x = F.relu(x)

        x = self.l2(x)
        #x = self.bn2(x)
        x = F.relu(x)

        x = self.l3(x)
        return x


class Critic_NN(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Critic_NN, self).__init__()
        
        self.l1 = nn.Linear(state_dim+action_dim, CRITIC_HL1)
        self.l1.bias.data = torch.nn.Parameter(torch.zeros(CRITIC_HL1))
        nn.init.xavier_uniform_(self.l1.weight.data)

        #self.bn1 = nn.LayerNorm(CRITIC_HL1)
        
        self.l2 = nn.Linear(CRITIC_HL1, CRITIC_HL2)
        self.l2.bias.data = torch.nn.Parameter(torch.zeros(CRITIC_HL2))
        nn.init.xavier_uniform_(self.l2.weight.data)

        #self.bn2 = nn.LayerNorm(CRITIC_HL2)

        self.l3 = nn.Linear(CRITIC_HL2, 1)
        self.l3.bias.data = torch.nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.l3.weight.data)


    def forward(self, state, action): 
        x = torch.cat((state, action), 1) 
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        output = self.l3(x)
        return output


if __name__ == "__main__":
    net = Actor_NN(5,1)
    print(net.l1(torch.tensor([0.,0.,0.,0.,0.])))