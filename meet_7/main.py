import ale_py
import gymnasium as gym
import numpy as np

from replay_buffer import ReplayBuffer
from utils import build_model, preprocess
from training import train_dqn

def main():
    """
    Main function to train and test a DQN agent in the Freeway environment.
    """
    gym.register_envs(ale_py)

    env = gym.make("ALE/Freeway-v5", render_mode=None, max_episode_steps=500)
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n

    model = build_model(state_dim, action_dim)
    target_model = build_model(state_dim, action_dim)
    target_model.set_weights(model.get_weights())

    replay_buffer = ReplayBuffer(capacity=10000)

    train_dqn(env, model, target_model, replay_buffer, episodes=20, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, target_update_freq=10, fps=30)

    env = gym.make("ALE/Freeway-v5", render_mode="human")
    raw_state, _ = env.reset()
    state = preprocess(raw_state)
    total_reward = 0
    step_count = 0

    while True:
        action = np.argmax(model.predict(state[np.newaxis], verbose=0))
        raw_next_state, reward, terminated, truncated, _ = env.step(action)
        state = preprocess(raw_next_state)
        total_reward += reward
        step_count += 1

        if terminated or truncated:
            print(f"Test Completed: Total Steps: {step_count}, Total Reward: {total_reward}")
            break

    env.close()

if __name__ == "__main__":
    main()
