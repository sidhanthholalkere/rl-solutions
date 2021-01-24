import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    """
    Simulates an n-armed bandit and attempts to solve it by using greedy
    action selection and incremental action value calculation
    """

    def __init__(self, n=10, q_mean=0, q_var=1, noise_mean=0, noise_var=1):
        """
        Initializes the n-armed bandit and solution
        """
        self.n = n

        self.q_values_real = np.random.normal(loc=q_mean, scale=np.sqrt(q_var), size=n)
        self.noise_mean = noise_mean
        self.noise_var = noise_var

        self.k = np.zeros(n)
        self.q_values_pred = np.zeros(n)
        
        self.cumulative_reward = 0
        self.rewards = []
        self.actions = []

    def pull(self, action):
        """
        Pulls the n-armed bandit when given an action
        """
        action = action.item() # convert from numpy int to standard int

        assert action < self.n
        assert isinstance(action, int)

        noise = np.random.normal(loc=self.noise_mean, scale=np.sqrt(self.noise_var))
        reward = self.q_values_real[action] + noise # return reward based on real q values with some noise

        return reward

    def act(self):
        """
        Greedily selects an action and pulls that lever
        """
        action = np.argmax(self.q_values_pred) # greedily select which action to take
        self.k[action] += 1 # increment the action count (k) by one
        reward = self.pull(action)

        # update the predicted value using alpha = 1/k
        self.q_values_pred[action] = self.q_values_pred[action] + (1 / self.k[action]) * (reward - self.q_values_pred[action])

        self.cumulative_reward += reward
        self.rewards.append(reward)
        self.actions.append(action)

def plot_rewards(rewards):
    """
    Plots the recieved rewards over time
    """
    x = np.arange(len(rewards))
    y = rewards
    plt.plot(x, y)
    plt.xlabel("timestep")
    plt.ylabel("reward")
    plt.show()

simulations = 1000
timesteps = 1000

avg_rewards = np.zeros(timesteps)

for _ in range(simulations):
    b = Bandit()
    for _ in range(timesteps):
        b.act()

    avg_rewards += np.array(b.rewards)

avg_rewards /= simulations

plot_rewards(avg_rewards)