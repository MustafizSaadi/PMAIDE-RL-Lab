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

def UCBBanditGame(bandit, timesteps, c=2):
    arms = bandit.arms
    Q_estimates = np.zeros(arms)
    action_counts = np.zeros(arms)
    rewards = np.zeros(timesteps)
    optimal_actions = np.zeros(timesteps)

    optimal_action = np.argmax(bandit.q_star)

    for t in range(timesteps):
        if t < arms:
            action = t
        else:
            action = np.argmax(Q_estimates + c * np.sqrt(np.log(t + 1) / action_counts))

        reward = bandit.get_reward(action)
        rewards[t] = reward
        if action == optimal_action:
            optimal_actions[t] = 1
        action_counts[action] += 1
        Q_estimates[action] += (reward - Q_estimates[action]) / action_counts[action]

    return rewards, optimal_actions


# play the UCB epsilon game
def PlayUCBEpsilon(n_games, arms, timesteps, c, epsilon):
    avg_rewards = {epsilon: np.zeros(timesteps)}
    avg_rewards['UCB'] = np.zeros(timesteps)
    avg_optimal_actions = {epsilon: np.zeros(timesteps)}
    avg_optimal_actions['UCB'] = np.zeros(timesteps)
    i = 0
    for _ in range(n_games):
        i += 1
        print(f"Simulating game {i}/{n_games}...")
        bandit = Bandit(arms)
        rewards, optimal_actions = BanditGame(bandit, timesteps, epsilon)
        avg_rewards[epsilon] += rewards
        avg_optimal_actions[epsilon] += optimal_actions

        rewards_ucb, optimal_actions_ucb = UCBBanditGame(bandit, timesteps, c)
        avg_rewards['UCB'] += rewards_ucb
        avg_optimal_actions['UCB'] += optimal_actions_ucb

    for key in avg_rewards.keys():
        avg_rewards[key] /= n_games

    for key in avg_optimal_actions.keys():
        avg_optimal_actions[key] /= n_games

    steps = np.arange(timesteps)

    # Setting up the figure
    plt.figure(figsize=(10, 8))

    # Plotting Average Rewards
    plt.subplot(2, 1, 1)
    for key, rewards in avg_rewards.items():
        label = f'ε = {key}' if key != 'UCB' else 'UCB (c=2)'
        color = 'grey' if key != 'UCB' else '#1E90FF'
        linestyle = 'solid' if key == 'UCB' else 'dashed'
        plt.plot(
            steps,
            rewards,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2 if key == 'UCB' else 1.5
        )

    plt.xlabel('Steps', fontsize=14, fontweight='bold', color='#333')
    plt.ylabel('Average Reward', fontsize=14, fontweight='bold', color='#333')
    plt.legend(fontsize=12, loc='lower right')
    plt.title('Average Rewards vs. Steps', fontsize=16, fontweight='bold', color='#111')

    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # Plotting % Optimal Actions
    plt.subplot(2, 1, 2)
    for key, optimal_action in avg_optimal_actions.items():
        label = f'ε = {key}' if key != 'UCB' else 'UCB (c=2)'
        color = 'grey' if key != 'UCB' else '#1E90FF'
        linestyle = 'solid' if key == 'UCB' else 'dashed'
        plt.plot(
            steps,
            optimal_action * 100,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2
        )
    plt.xlabel('Steps', fontsize=14, fontweight='bold', color='#333')
    plt.ylabel('% Optimal Actions', fontsize=14, fontweight='bold', color='#333')
    plt.legend(fontsize=12)
    plt.title('% Optimal Actions vs. Steps', fontsize=16, fontweight='bold', color='#111')
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
    plt.tight_layout(pad=3)
    # plt.show()
    plt.savefig('ucb_epsilon_simulation.png', bbox_inches='tight')

# GamePlay
PlayUCBEpsilon(n_games=2000, arms=10, timesteps=1000, c=2, epsilon=0.1)