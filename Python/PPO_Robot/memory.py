import torch


class Memory:
    """
    Buffer for storing trajectory data.
    Stores states, actions, rewards, log probabilities, and terminal flags.
    """

    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        """Clear all stored data."""
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()

    def get_tensors(self, device):
        """
        Convert lists to tensors and return them.

        Args:
            device (torch.device): Device to move tensors to

        Returns:
            tuple: (states, actions, logprobs) as tensors
        """
        states = torch.stack(self.states).to(device).detach()
        actions = torch.stack(self.actions).to(device).detach()
        logprobs = torch.stack(self.logprobs).to(device).detach()

        return states, actions, logprobs