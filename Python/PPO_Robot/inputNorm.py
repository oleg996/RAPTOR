import numpy as np

class RunningMeanStd:
    def __init__(self, shape, max_count=100000):
        self.n = 1e-4          # small epsilon, not zero
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.max_count = max_count

    def update(self, x):
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

        self.mean = new_mean
        self.var = new_var
        self.n = min(tot_count, self.max_count)  # stays adaptive

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)