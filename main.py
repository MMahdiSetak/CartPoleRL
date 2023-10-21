import gymnasium as gym
import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import utils
from REINFORCE.model import PolicyNetwork
from REINFORCE.policy import train_policy, collect_trajectory

env = gym.make('CartPole-v1')
output_size = env.action_space.n - 1
input_size = env.observation_space.shape[0]
total_episodes = 500
# Parameters
hparams = {
    'hidden_size': [2, 4, 8, 16, 32],
    'discount_factor': [0.5, 0.8, 0.9, 0.99, 1],
    'normalize': [False, True],
}
#####################
total_runs = np.prod([len(arr) for arr in hparams.values()])
total_runs *= 3  # for repeating the same hparams
all_rewards = []

# Render trained model
# policy = PolicyNetwork(input_size, 4, output_size)
# policy.load_state_dict(torch.load('assets/saved_models/model_4.pth'))
# policy.eval()
# utils.human_render(policy)
# exit()

for run in range(total_runs):
    hparam = utils.get_param(hparams, run)
    # Log in tensorboard
    writer = SummaryWriter(f"cartpole_tensorboard/lr:{run}", max_queue=1000000, flush_secs=30)
    policy = PolicyNetwork(input_size, hparam['hidden_size'], output_size)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)
    cumulative_rewards = 0
    rewards = []
    for episode in range(total_episodes):
        trajectory = collect_trajectory(env, policy)
        train_policy(trajectory, policy, optimizer, hparam['discount_factor'], hparam['normalize'])
        episode_reward = sum(x[2] for x in trajectory)
        if episode >= total_episodes - 100:  # storing sum of the last 100 episodes as score of the hparams
            cumulative_rewards += episode_reward
        rewards.append(episode_reward)
        writer.add_scalar("reward", episode_reward, episode)
        # if episode % 50 == 49:  # Visualize learning process
        #     render_reward = utils.human_render(policy)
        #     print(f'Episode {episode + 1}, Reward: {render_reward}')
    all_rewards.append(rewards)
    writer.add_hparams(hparam, {'final_result': cumulative_rewards})
    # torch.save(policy.state_dict(), f'assets/saved_models/model_{hparam["hidden_size"]}.pth')

# averaged_rewards = np.mean(all_rewards, axis=0)
# utils.plot_learning_curve(averaged_rewards, 20, runs=total_runs, hidden_size=hparams['hidden_size'][0],
#                           discount_factor=hparams['discount_factor'][0], normalize_reward=hparams['normalize'][0])
