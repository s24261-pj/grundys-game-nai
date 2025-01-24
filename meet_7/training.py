import time
import numpy as np
from utils import preprocess, epsilon_greedy_policy

def train_dqn(env, model, target_model, replay_buffer, episodes, batch_size, gamma, epsilon, epsilon_min, epsilon_decay, target_update_freq, fps):
    """
    Train a DQN agent in the given environment.

    Args:
        env (gym.Env): Gymnasium environment.
        model (tf.keras.Model): Main Q-network.
        target_model (tf.keras.Model): Target Q-network.
        replay_buffer (ReplayBuffer): Experience replay buffer.
        episodes (int): Number of training episodes.
        batch_size (int): Size of each training batch.
        gamma (float): Discount factor for Q-learning.
        epsilon (float): Exploration rate for epsilon-greedy policy.
        epsilon_min (float): Minimum exploration rate.
        epsilon_decay (float): Decay rate for epsilon.
        target_update_freq (int): Frequency of target model updates.
        fps (int): Frames per second to control the speed of training.

    Returns:
        tf.keras.Model: Trained Q-network.
    """
    frame_duration = 1.0 / fps

    for episode in range(episodes):
        raw_state, _ = env.reset()
        state = preprocess(raw_state)
        total_reward = 0
        step_count = 0

        while True:
            start_time = time.time()

            action = epsilon_greedy_policy(state, epsilon, model, env.action_space.n)

            raw_next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = preprocess(raw_next_state)
            total_reward += reward
            step_count += 1

            replay_buffer.add((state, action, reward, next_state, terminated))

            state = next_state

            if terminated or truncated:
                print(f"Episode {episode + 1}/{episodes}, Total Steps: {step_count}, Total Reward: {total_reward}")
                break

            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = np.array(states)
                actions = np.array(actions)
                rewards = np.array(rewards)
                next_states = np.array(next_states)
                dones = np.array(dones)

                q_next = np.amax(target_model.predict(next_states, verbose=0), axis=1)
                q_targets = rewards + gamma * q_next * (1 - dones)

                q_values = model.predict(states, verbose=0)
                for i, action in enumerate(actions):
                    q_values[i, action] = q_targets[i]

                model.fit(states, q_values, epochs=1, verbose=0)
                print(f"Episode {episode + 1}, Step {step_count}: Training performed with batch size {batch_size}")

            time.sleep(max(0, frame_duration - (time.time() - start_time)))

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode % target_update_freq == 0:
            target_model.set_weights(model.get_weights())
            print(f"Episode {episode + 1}: Target model updated")

    return model
