import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, device):
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0

        # Pre-allocate on GPU
        self.states = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((max_size, 1), dtype=torch.float32, device=device)

    def add(self, state, action, reward, next_state, done):
        # Direct write — no python list append
        self.states[self.ptr] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr, 0] = reward
        self.next_states[self.ptr] = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[self.ptr, 0] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self.size