from collections import deque
import numpy as np

class HistoryWrapper:
    def __init__(self, env, obs_dim, action_dim, history_len=3):
        self.env = env
        self.history_len = history_len
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Create rolling buffers
        self.obs_history = deque(maxlen=history_len)
        self.action_history = deque(maxlen=history_len)
        
        # Calculate new state dimension: current obs + past obs + past actions
        self.new_state_dim = (obs_dim * history_len) + (action_dim * (history_len - 1))

    def reset(self):
        obs, info = self.env.reset()
        
        # Fill the buffers with the initial observation and zero actions
        for _ in range(self.history_len):
            self.obs_history.append(obs)
            
        for _ in range(self.history_len - 1):
            self.action_history.append(np.zeros(self.action_dim))
            
        return self._get_stacked_state(), info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update buffers
        self.obs_history.append(next_obs)
        self.action_history.append(action)
        
        return self._get_stacked_state(), reward, terminated, truncated, info

    def _get_stacked_state(self):
        # Flatten the deques into a single 1D numpy array
        stacked_obs = np.concatenate(self.obs_history)
        if len(self.action_history) > 0:
            stacked_actions = np.concatenate(self.action_history)
            return np.concatenate([stacked_obs, stacked_actions])
        return stacked_obs
        
    def close(self):
        self.env.close()