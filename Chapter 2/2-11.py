# nonstationary case (as in 2.5)
# include e-greedy with alpha=0.1
# runs of 200,000 and metric as 100,000
import numpy as np

class NonstationaryBanditTask:
    """
    An instance of the nonstationary bandit task
    """
    def __init__(self, walk_mean=0., walk_std=0.01, n=10):
        self.q_values = np.zeros(n)

        self.n = 10
        self.walk_mean = walk_mean
        self.walk_std = walk_std

    def pull(self, action):

        action = action.item()
        assert isinstance(action, int)
        assert action < self.n

        reward = self.q_values[action]

        self.q_values += np.random.normal(loc=self.walk_mean, scale=self.walk_std, size=self.n)

        return reward

class BanditSolver:
    """
    A General Bandit solver
    """

    def __init__(self, n=10):
        self.n = n
        self.rewards = np.empty(0)

    
