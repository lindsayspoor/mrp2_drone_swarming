import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class Critic(nn.Module):
    def __init__(self, alpha, input_dims, layer1_dims, layer2_dims, N, N_actions, filename, directory):
        super(Critic, self).__init__()

        self.file = os.path.join(directory, filename)

        self.layer1 = nn.Linear(input_dims + N * N_actions, layer1_dims) # critic takes the full state observation vector  
        self.layer2 = nn.Linear(layer1_dims, layer2_dims)
        self.out_layer = nn.Linear(layer2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')
        # self.double()
        self.to(self.device)

    def forward(self, state, action):

        x1 = F.relu(self.layer1(T.cat([state, action], dim=1)))
        x2 = F.relu(self.layer2(x1))
        q_val = self.out_layer(x2)

        return q_val
    

    def save(self):
        T.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(T.load(self.file))


class Actor(nn.Module):
    def __init__(self, alpha, input_dims, layer1_dims, layer2_dims, N_actions, filename, directory):
        super(Actor, self).__init__()

        self.file = os.path.join(directory, filename)

        self.layer1 = nn.Linear(input_dims, layer1_dims)
        self.layer2 = nn.Linear(layer1_dims, layer2_dims)
        self.out_layer = nn.Linear(layer2_dims, N_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')

        self.to(self.device)

    
    def forward(self, obs):
        # print(f"{obs=}")
        x1 = F.relu(self.layer1(obs))
        # print(f"{x1=}")
        x2 = F.relu(self.layer2(x1))
        output = T.softmax(self.out_layer(x2), dim=-1)
        print(f"{output.shape=}")
        return output
    

    
    def save(self):
        T.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(T.load(self.file))



