import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DuelingDQN(nn.Module):
    """Dueling network architecture for more stable value estimation."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        # Separate streams for state-value and advantages
        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.value(x)
        advantage = self.advantage(x)
        # Combine value and advantages into Q-values
        return value + advantage - advantage.mean(dim=1, keepdim=True)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.model = DuelingDQN(state_dim, action_dim)
        self.target = DuelingDQN(state_dim, action_dim)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 3)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            qvals = self.model(state)
        return qvals.argmax().item()

    def remember(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_, done = map(np.array, zip(*batch))
        s = torch.FloatTensor(s)
        a = torch.LongTensor(a)
        r = torch.FloatTensor(r)
        s_ = torch.FloatTensor(s_)
        done = torch.FloatTensor(done)

        q = self.model(s).gather(1, a.unsqueeze(1)).squeeze()
        next_actions = self.model(s_).argmax(1).unsqueeze(1)
        q_next = self.target(s_).gather(1, next_actions).squeeze()
        q_target = r + self.gamma * q_next * (1 - done)
        loss = nn.MSELoss()(q, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.target.load_state_dict(torch.load(path))
