import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def build_model(input_dim, output_dim):
    """
    Build and compile a deep Q-network model.

    Args:
        input_dim (int): Dimension of the input layer.
        output_dim (int): Number of possible actions (output layer).

    Returns:
        tf.keras.Model: Compiled DQN model.
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def preprocess(state):
    """
    Normalize and flatten the state.

    Args:
        state (np.ndarray): Raw state from the environment, typically an image.

    Returns:
        np.ndarray: Preprocessed state as a flattened array with values normalized to [0, 1].
    """
    state = state.astype('float32') / 255.0
    return state.flatten()

def epsilon_greedy_policy(state, epsilon, model, action_dim):
    """
    Select an action using the epsilon-greedy strategy.

    Args:
        state (np.ndarray): Current state as a flattened array.
        epsilon (float): Exploration rate. A higher value means more exploration.
        model (tf.keras.Model): Trained Q-network model for action selection.
        action_dim (int): Number of possible actions.

    Returns:
        int: Selected action index.
    """
    if np.random.rand() < epsilon:
        return np.random.randint(action_dim)
    q_values = model.predict(state[np.newaxis], verbose=0)
    return np.argmax(q_values[0])
