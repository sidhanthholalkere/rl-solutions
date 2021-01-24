# Desc: 10-armed bandits where state value is equal and goes on random walks (mu=0, sigma=0.01)
# keep track of avg reward over time and %optimal action over time
# methods: incrementally computed sample avgs, action-value with different alpha
# hyperparams: alpha=0.1 for different alpha method, epsilon=0.1, steps=10000

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# We want to keep a bandit class so it's easy to repeat and modify the action methods
class BanditExperiment:
    """
    Performs the bandit experiment as described in exercise 2.5 in
    Sutton and Barto's Reinforcement Learning
    """

    def __init__(self, sample_avg=True, walk_mean=0, walk_stddev=0.01, arms=10, epsilon=0.1, alpha=0.1):
        self.arms = arms
        self.q_values_real = np.zeros(arms) # initialize q_* as all equal
        self.walk_mean = walk_mean
        self.walk_stddev = walk_stddev
        self.sample_avg = sample_avg
        self.epsilon = epsilon
        self.alpha = alpha

        self.q_values_pred = np.zeros(arms) # initialize Q as all 0s
        self.k = np.zeros(arms) # keep track of how many times we have selected each action

        self.rewards = np.empty(0) # keep track of what rewards we get
        self.optimal = np.empty(0) # keep track of whether we chose the optimal action or not

    def pull(self, action):
        """
        Pulls a lever as designated by action
        """
        action = int(action) # get the int from the numpy.int64

        assert action < self.arms
        assert isinstance(action, int)

        reward = self.q_values_real[action]
        self.q_values_real += np.random.normal(loc=self.walk_mean, scale=self.walk_stddev, size=self.arms) # take the true qs on a random walk

        return reward

    def act(self):
        """
        Chooses an action and updates the Q table based on the reward
        """
        # we randomly choose an action at probability epsilon, else greedily
        action = np.random.randint(0, self.arms) if np.random.rand() <= self.epsilon else np.argmax(self.q_values_pred)
        optimal = action == np.argmax(self.q_values_real)
        reward = self.pull(action)
        self.k[action] += 1

        alpha = 1 / self.k[action] if self.sample_avg else self.alpha # which method are we updating with?
        self.q_values_pred[action] = self.q_values_pred[action] + alpha * (reward - self.q_values_pred[action])

        # and now to keep track of optimal/reward
        self.rewards = np.append(self.rewards, reward)
        self.optimal = np.append(self.optimal, optimal)
    
def plot_rewards(rewards):
    """
    Plots the recieved rewards over time
    """
    x = np.arange(len(rewards))
    y = rewards
    plt.plot(x, y)
    plt.xlabel("time")
    plt.ylabel("avg reward")
    plt.show()

def plot_optimal(optimals):
    """
    Plots the rate of optimal decision making over time
    """
    x = np.arange(len(optimals))
    y = optimals
    plt.plot(x, y)
    plt.xlabel("time")
    plt.ylabel("percentage optimal actions")
    plt.show()

simulations = 100
timesteps = 10000

# Getting data for sample average
sample_avg_rewards = np.zeros(timesteps)
sample_avg_optimal = np.zeros(timesteps)

for _ in range(simulations):
    b = BanditExperiment(sample_avg=True)
    for _ in range(timesteps):
        b.act()

    #now add to rewards/optimals
    sample_avg_rewards += b.rewards
    sample_avg_optimal += b.optimal

sample_avg_rewards /= simulations
sample_avg_optimal /= simulations

# Getting data for exponentially weighted average
alpha_rewards = np.zeros(timesteps)
alpha_optimal = np.zeros(timesteps)

for _ in range(simulations):
    b = BanditExperiment(sample_avg=False)
    for _ in range(timesteps):
        b.act()

    #now add to rewards/optimals
    alpha_rewards += b.rewards
    alpha_optimal += b.optimal

alpha_rewards /= simulations
alpha_optimal /= simulations

# Now plot them
avg_reward_df = pd.DataFrame({"sample average method": sample_avg_rewards, "exponentially weighted average method": alpha_rewards})
sns.lineplot(data=avg_reward_df)
plt.show()

optimal_df = pd.DataFrame({"sample average method": sample_avg_optimal, "exponentially weighted average method": alpha_optimal})
sns.lineplot(data=optimal_df)
plt.show()