import numpy as np
import torch
from torch.autograd import Variable


# Collect trajectory by running the policy
def collect_trajectory(env, policy):
    state, _ = env.reset()
    trajectory = []

    while True:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = policy(state_tensor)
        action_probs = action_probs.detach().numpy().flatten()[0]
        action = np.random.choice([0, 1], p=[1 - action_probs, action_probs])
        next_state, reward, terminated, truncated, _ = env.step(action)
        trajectory.append((state, action, reward))
        state = next_state
        if terminated or truncated:
            break

    return trajectory


# Train the policy network using the REINFORCE algorithm
def train_policy(trajectory, policy, optimizer, discount_factor, normalize_reward):
    states, actions, rewards = zip(*trajectory)

    # Compute discounted rewards
    discounted_rewards = [0] * len(rewards)
    cumulative_rewards = 0
    for i in reversed(range(len(rewards))):
        cumulative_rewards = rewards[i] + cumulative_rewards * discount_factor
        discounted_rewards[i] = cumulative_rewards

    # Normalize discounted rewards
    if normalize_reward:
        discounted_rewards = np.array(discounted_rewards)
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (
                np.max(discounted_rewards) - np.min(discounted_rewards))

    # Compute the policy gradient
    state_tensors = torch.FloatTensor(np.array(states))
    action_tensors = torch.LongTensor(actions)
    reward_tensors = torch.FloatTensor(discounted_rewards)

    probs = policy(Variable(state_tensors))
    probs = probs.flatten()
    probs = torch.stack((1 - probs, probs), dim=1)
    log_probs = torch.log(probs)
    selected_log_probs = reward_tensors * log_probs[np.arange(len(action_tensors)), action_tensors]
    loss = -selected_log_probs.sum()

    # Update the policy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
