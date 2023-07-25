import numpy as np
import copy
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from model import Actor, Critic

# Hyper parameters
BUFFER_SIZE = 100000 # the size of the replay memory
BATCH_SIZE = 256 # batch sise
GAMMA = 0.99  # discount factor
TAU = 0.001 # parameter for soft update
LEARN_EVERY = 20 # time steps per which the learning occurs

LR_ACTOR = 0.0025  # learning rate for actor model
LR_CRITIC = 0.001 # learning rate for critic model
WEIGHT_DECAY_ACTOR = 0  #weight decay for the actor
WEIGHT_DECAY_CRITIC = 0 # weight decay for the critic

EPSILON = 1.0 # noise factor
EPSILON_DECAY = 0.999 # noise factor decay

# Select the device. GPU recommended.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """DDPG agent which interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed=123):
        """Initialize an agent.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.seed = random.seed(seed)
        self.action_size = action_size
        self.state_size = state_size

        print("Device used: ", device)

        # Actor networks (local+target)
        self.actor_local = Actor(self.state_size, self.action_size, seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = LR_ACTOR, weight_decay = WEIGHT_DECAY_ACTOR)

        # Critic network (local+target)
        self.critic_local = Critic(self.state_size, self.action_size, seed).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = LR_CRITIC, weight_decay = WEIGHT_DECAY_CRITIC)

        # Initial soft updates
        self.soft_update(self.critic_local, self.critic_target, 1.0)
        self.soft_update(self.actor_local, self.actor_target, 1.0)

        # Noise process
        self.noise = OUNoise(self.action_size, seed)
        self.epsilon = EPSILON

        # Replay buffer
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)

        # Time stamp for learning per LEARN_EVERY
        self.time_stamp = 0


    def step(self, states, actions, rewards, next_states, dones):
        """Save a tuple of experiences in replay buffer, and use random sample from buffer to learn"""
        # Save experience to the replay buffer
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        # Take the learning process 10 times per LEARN_EVERY
        self.time_stamp = (self.time_stamp + 1) % LEARN_EVERY
        if self.time_stamp == 0:
            if len(self.memory) > BATCH_SIZE:
                for i in range(10):
                    # recall randomly experiences from the memory and learn
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)
                    self.epsilon *= EPSILON_DECAY

    def act(self, state, add_noise=True):
        """Selects actions for given states and adds noise"""

        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample() * self.epsilon

        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()
        self.t_step  = 0

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        # unpackage given experiences
        states, actions, rewards, next_states, dones = experiences

        # Update critic
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Clip the gradient
        nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)
        self.critic_optimizer.step()

        # Update actor
        # Compute actor loss
        actions_pred = self.actor_local(states)
        # Note the minus sign
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        # Update epsilon and noise
        self.epsilon *= EPSILON_DECAY
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for local_param, target_param in zip(local_model.parameters(), target_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.1):
        """Initialize parameters and noise process"""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed=114):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.seed = random.seed(seed)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = deque(maxlen = self.buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


    def add(self, state, action, reward, next_state, done):
        """Add a new experience to replay buffer"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)


    def sample(self):
        """Random-sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k = self.batch_size)
        # change to the compatible form of tensors
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)


        return (states, actions, rewards, next_states, dones)


    def __len__(self):
        """Return the current size of the replay buffer memory"""
        return len(self.memory)