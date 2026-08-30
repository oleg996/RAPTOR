import numpy as np

class HistoryWrapper:
    def __init__(self, env, obs_dim, action_dim, history_len=3):
        self.env = env
        self.history_len = history_len
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.new_state_dim = (obs_dim * history_len) + (action_dim * (history_len))
        
        # Pre-allocate arrays
        self.obs_history = np.zeros((history_len, obs_dim), dtype=np.float32)
        self.action_history = np.zeros((history_len, action_dim), dtype=np.float32)

    def reset(self):
        obs, info = self.env.reset()
        
        # Broadcast the initial observation across all history frames
        self.obs_history[:] = obs
        
        # Keep actions at 0 (neutral / stationary)
        self.action_history[:] = 0
        
        return self._get_stacked_state(), info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Roll arrays back (shift index 1 to 0, 2 to 1, etc.)
        self.obs_history[:-1] = self.obs_history[1:]
        self.action_history[:-1] = self.action_history[1:]
        
        # Add new data
        self.obs_history[-1] = next_obs
        self.action_history[-1] = action
        
        return self._get_stacked_state(), reward, terminated, truncated, info

    def _get_stacked_state(self):
        # Flatten in one fast operation
        return np.concatenate([self.obs_history.flatten(), self.action_history.flatten()])