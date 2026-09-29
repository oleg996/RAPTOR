import numpy as np


class RunningMeanStd:
    """Welford-style running mean/variance with an optional freeze.

    The freeze exists because this normalizer is applied at *update* time, not
    at store time (the replay buffer holds raw observations). A normalizer that
    keeps drifting therefore silently re-scales every transition already in the
    buffer and makes the TD target for a fixed transition non-stationary, which
    is much worse for off-policy RL than for on-policy RL. Call ``freeze()``
    once the policy's observation distribution has settled.
    """

    def __init__(self, shape, max_count=1e5):
        self.n = 1e-4  # small epsilon, not zero
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.max_count = max_count
        self.frozen = False

    def update(self, x):
        if self.frozen:
            return

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.n + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.n
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.n * batch_count / tot_count
        new_var = M2 / tot_count

        # Update in-place to prevent thread race conditions
        np.copyto(self.mean, new_mean)
        np.copyto(self.var, new_var)
        self.n = min(tot_count, self.max_count)

    def freeze(self):
        self.frozen = True

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)
