import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    """
    initialize the weights of a given layer

    """
    fan_in = layer.weight.data.size()[0]
    lim = np.sqrt(1.0/fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """
    Actor network
    that maps a state to an action.

    """

    def __init__(self, state_size, action_size, seed = 19):
        """
        Initialization.

        Params
        =====
        state_size (int): state size
        action_size (float): action size
        seed (int): random seed

        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed) # sets the random seed
        self.state_size = state_size # sets the state size
        self.action_size = action_size # sets the action size
        hidden_units = [128, 128] # assigns the numbers of the units in the hidden layers below

        # Linear layers
        self.fc1 = nn.Linear(self.state_size, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], self.action_size)
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_units[0])
        self.bn2 = nn.BatchNorm1d(hidden_units[1])


        self.reset_parameters() # initializees the weights

    def reset_parameters(self):

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Forwarding with relu and tanh

        """

        x  = F.relu(self.fc1(state))
        # x = self.bn1(x)
        x = F.relu(self.fc2(x))
        # x = self.bn2(x)
        x = F.tanh(self.fc3(x))

        return x


class Critic(nn.Module):
    """
    Cricic network
    that maps (state, action) to Q-value.

    """

    def __init__(self, state_size, action_size, seed = 8):
        """
        Initialization.

        Param
        =====
        state_size (int): state size
        action_size (int): action size
        seed (int): random seed

        """

        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed) # sets the random seed
        self.state_size = state_size # sets the state size
        self.action_size = action_size # sets the action size
        hidden_units = [128, 128] # assigns the numbers of nodes in the hidden layers

        # Linear layers
        self.fc1 = nn.Linear(self.state_size, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0] + self.action_size, hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], 1)
        # Batch normalization
        self.bn2 = nn.BatchNorm1d(hidden_units[1])


        self.reset_parameters() # initializes the weights

    def reset_parameters(self):
        """
        This method is for initializing the weights of the neural network.

        """

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Forwarding with cat and relu

        """

        xs = F.relu(self.fc1(state))
        x = torch.cat((xs, action), dim = 1)
        x = F.relu(self.fc2(x))
        # x = self.bn2(x)
        x = self.fc3(x)

        return x

