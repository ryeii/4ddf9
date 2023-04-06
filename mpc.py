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


