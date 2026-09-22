# import necessary libraries here
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(63)

# Defining bandit object
class Bandit(object):
    def __init__(self, arms=10):
        self.arms = arms
        self.q_star = np.random.normal(0, 1, self.arms)  # true action values

    def get_reward(self, action):
        return np.random.normal(self.q_star[action], 1)  # rewards sampled from N(q*(a), 1)

# Defining Multiarm bandit game using epsilon greedy approach
def BanditGame(bandit, timesteps, epsilon=0):
    arms = bandit.arms
    q_estimates = np.zeros(arms)
    action_counts = np.zeros(arms)
    rewards = np.zeros(timesteps)
    optimal_actions = np.zeros(timesteps)

    optimal_action = np.argmax(bandit.q_star)

    for t in range(timesteps):
        p = np.random.rand()
        if p < epsilon:
            action = np.random.choice(arms)
        else:
            action = np.argmax(q_estimates)

        reward = bandit.get_reward(action)
        rewards[t] = reward

        if action == optimal_action:
            optimal_actions[t] = 1

        action_counts[action] += 1
        q_estimates[action] += (reward - q_estimates[action]) / action_counts[action]

    return rewards, optimal_actions


# Simulating game with different parameters
def SimulateBanditGame(n_games, arms, timesteps, epsilons):
    avg_rewards = {epsilon: np.zeros(timesteps) for epsilon in epsilons}
    avg_optimal_actions = {epsilon: np.zeros(timesteps) for epsilon in epsilons}
    i = 0
    for _ in range(n_games):
        i += 1
        print(f"Simulating game {i}/{n_games}...")
        bandit = Bandit(arms)
        for epsilon in epsilons:
            rewards, optimal_actions = BanditGame(bandit, timesteps, epsilon)
            avg_rewards[epsilon] += rewards
            avg_optimal_actions[epsilon] += optimal_actions

    for epsilon in epsilons:
        avg_rewards[epsilon] /= n_games
        avg_optimal_actions[epsilon] /= n_games

    return avg_rewards, avg_optimal_actions


# Playing the Game
def PlayBanditGame(n_games, arms, timesteps, epsilons):
    avg_rewards, avg_optimal_actions = SimulateBanditGame(n_games, arms, timesteps, epsilons)
    steps = np.arange(timesteps)
    # Adjusting figure size for better clarity
    plt.figure(figsize=(10, 10))
    
    # Defining colors and line styles for a visually appealing distinction
    colors = {0: '#2E8B57', 0.01: '#FF4500', 0.1: '#1E90FF'}
    linestyles = {0: 'dotted', 0.01: 'dashed', 0.1: 'solid'}
    
    # Plotting Average Rewards
    plt.subplot(2, 1, 1)
    for epsilon, rewards in avg_rewards.items():
        label = 'Greedy' if epsilon == 0 else f'ε = {epsilon}'
        plt.plot(
            steps,
            rewards,
            label=label,
            color=colors[epsilon],
            linestyle=linestyles[epsilon],
            linewidth=2
        )
    plt.xlabel('Steps', fontsize=14, fontweight='bold', color='#333')
    plt.ylabel('Average Reward', fontsize=14, fontweight='bold', color='#333')
    plt.legend(fontsize=12)
    plt.title('Average Rewards vs. Steps', fontsize=16, fontweight='bold', color='#111')
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Plotting % Optimal Actions
    plt.subplot(2, 1, 2)
    for epsilon, optimal_action in avg_optimal_actions.items():
        label = 'Greedy' if epsilon == 0 else f'ε = {epsilon}'
        plt.plot(
            steps,
            optimal_action * 100,
            label=label,
            color=colors[epsilon],
            linestyle=linestyles[epsilon],
            linewidth=2
        )
    plt.xlabel('Steps', fontsize=14, fontweight='bold', color='#333')
    plt.ylabel('% Optimal Actions', fontsize=14, fontweight='bold', color='#333')
    plt.legend(fontsize=12)
    plt.title('% Optimal Actions vs. Steps', fontsize=16, fontweight='bold', color='#111')
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Tight layout and display
    plt.tight_layout(pad=3)
    # plt.show()
    plt.savefig('my_plot.png', bbox_inches='tight')

#GamePlay function call
epsilons = [0,0.01,0.1]
n_games = 2000
arms = 10
timesteps=1500

PlayBanditGame(n_games,arms,timesteps,epsilons)
