import numpy as np
import gym

def random_shooting_mpc(start_state, model, reward_fn, horizon, n_samples, gamma):
    """
    Random shooting model predictive control for the Pendulum-v1 environment in OpenAI Gym.
    
    Args:
    - start_state (numpy array): the initial state of the environment
    - model (function): a function that takes a state and an action, and returns the next state
    - reward_fn (function): a function that takes a state and returns a reward
    - horizon (int): the number of timesteps in the MPC horizon
    - n_samples (int): the number of control sequences to sample
    - gamma (float): the discount factor
    
    Returns:
    - optimal_control (numpy array): the optimal control sequence for the MPC horizon
    """
    
    # Initialize the control sequences randomly
    control_sequences = np.random.uniform(low=-2.0, high=2.0, size=(n_samples, horizon))
    
    # Evaluate the control sequences
    rewards = np.zeros(n_samples)
    for i in range(n_samples):
        state = start_state
        for t in range(horizon):
            action = control_sequences[i, t]
            next_state = model(state, action)
            rewards[i] += reward_fn(next_state, action) * gamma**t
            state = next_state
    
    # Find the optimal control sequence
    optimal_index = np.argmax(rewards)
    optimal_control = control_sequences[optimal_index]
    
    return optimal_control


def pendulum_model(state, action):
    temp_env = gym.make('Pendulum-v1')
    temp_env.reset()
    temp_env.state = state
    next_state, _, _, _ = temp_env.step([action])
    temp_env.close()
    return next_state

def pendulum_reward(state, action):
    temp_env = gym.make('Pendulum-v1')
    temp_env.reset()
    temp_env.state = state
    cos_theta, sin_theta, theta_dot = temp_env.state
    temp_env.close()
    return -(theta_dot**2 + 0.1*cos_theta**2 + 0.001*(action**2))


env = gym.make('Pendulum-v1')
horizon = 10
n_samples = 100
gamma = 0.98

obs = env.reset()
rewards = []
for i in range(10000):
    action = random_shooting_mpc(obs, pendulum_model, pendulum_reward, horizon, n_samples, gamma)[0]
    next_state, reward, done, info = env.step([action])
    rewards.append(reward)
    if i % 1000 == 0:
        print("at step " + str(i))

# calculate statistics
rewards_mean = np.mean(rewards)
rewards_std = np.std(rewards)
rewards_max = np.max(rewards)
rewards_min = np.min(rewards)
rewards_median = np.median(rewards)
rewards_25 = np.percentile(rewards, 25)
rewards_75 = np.percentile(rewards, 75)

print("mean: " + str(rewards_mean))
print("std: " + str(rewards_std))
print("max: " + str(rewards_max))
print("min: " + str(rewards_min))
print("median: " + str(rewards_median))
print("25: " + str(rewards_25))
print("75: " + str(rewards_75))

env.close()

# cummulative_reward = 0
# for _ in range(1000):
#     obs = env.reset()
#     action = env.action_space.sample()
#     next_state, reward, done, info = env.step([action])
#     cummulative_reward += reward
#     if done:
#         # Reset the environment
#         env.reset()
#     if _ % 100 == 0:
#         print("2 Average reward over 100 episodes: {}".format(cummulative_reward/100) + " " + str(_))

# print("2 Average reward over 1000 episodes: {}".format(cummulative_reward/1000))
    

# env.close()
