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
            rewards[i] += reward_fn(next_state) * gamma**t
            state = next_state
    
    # Find the optimal control sequence
    optimal_index = np.argmax(rewards)
    optimal_control = control_sequences[optimal_index]
    
    return optimal_control


def pendulum_model(state, action):
    env = gym.make('Pendulum-v1')
    env.reset()
    env.state = state
    next_state, _, _, _ = env.step([action])
    env.close()
    return next_state

def pendulum_reward(state):
    env = gym.make('Pendulum-v1')
    env.reset()
    env.state = state
    cos_theta, sin_theta, theta_dot = env.state
    env.close()
    return -(theta_dot**2 + 0.1*sin_theta**2)


env = gym.make('Pendulum-v1')
start_state = env.reset()
horizon = 10
n_samples = 100
gamma = 0.98
optimal_control = random_shooting_mpc(start_state, pendulum_model, pendulum_reward, horizon, n_samples, gamma)

# Apply the first action in the optimal control sequence
action = optimal_control[0]
next_state, reward, done, info = env.step([action])


result = []
obs = env.reset()
cumulated_reward = 0
for i in range(100):
    all_rewards = 0
    for j in range(500):
        action = random_shooting_mpc(obs, pendulum_model, pendulum_reward, horizon, n_samples, gamma)[0]
        next_state, rewards, done, info = env.step([action])
        # env.render()
        all_rewards += rewards
    # reset the environment
    obs = env.reset()
    cumulated_reward += all_rewards
    print("All reward over 500 episodes: {}".format(all_rewards) + " " + str(i))
result.append(cumulated_reward/100)

print(result)


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
