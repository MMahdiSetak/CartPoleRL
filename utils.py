import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt


def get_param(hparams, run):
    hparam = {}
    for key, value in hparams.items():
        n = len(value)
        hparam[key] = value[run % n]
        run //= n
    return hparam


def plot_learning_curve(rewards, window_size=100, runs=-1, hidden_size=-1, discount_factor=-1, normalize_reward=-1):
    rolling_average_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
    plt.plot(rewards, label='Raw Rewards')
    plt.plot(np.arange(window_size - 1, len(rewards)), rolling_average_rewards,
             label=f'Rolling Average ({window_size} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(
        f'Learning Curve\nRuns: {runs}, Hidden:{hidden_size}, DF: {discount_factor}, Normalize: {normalize_reward}')
    plt.legend()
    plt.show()


def human_render(policy):
    show_env = gym.make("CartPole-v1", render_mode="human")
    state, info = show_env.reset()
    render_reward = 0
    while True:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = policy(state_tensor)
        action_probs = action_probs.detach().numpy().flatten()[0]
        state, reward, terminated, truncated, _ = show_env.step(
            np.random.choice([0, 1], p=[1 - action_probs, action_probs]))
        render_reward += reward
        if terminated or truncated:
            show_env.close()
            break
    return render_reward
