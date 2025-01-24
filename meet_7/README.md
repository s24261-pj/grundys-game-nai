# Deep Q-Network (DQN) Implementation for Atari Freeway

This project demonstrates an implementation of a Deep Q-Network (DQN) to train an agent to play the Atari game **Freeway** using `gymnasium` and `tensorflow`. The DQN agent employs reinforcement learning to determine optimal strategies for maximizing rewards in the game.

---

## Problem Statement
In Atari's Freeway, the objective is for the agent (a chicken) to cross a multi-lane highway filled with moving cars and reach the other side without getting hit. The challenge lies in identifying the optimal sequence of actions to achieve this goal efficiently.

---

## Features of This Implementation

1. **Custom DQN Architecture**:
   - Neural network with two hidden layers of 128 neurons each, using ReLU activation.
   - An output layer with linear activation to estimate Q-values for each possible action.

2. **Experience Replay Buffer**:
   - Stores past experiences and enables efficient training by sampling mini-batches.

3. **Epsilon-Greedy Policy**:
   - Balances exploration and exploitation for action selection.

4. **Target Network**:
   - Stabilizes training by periodically updating weights.

5. **Game Preprocessing**:
   - Normalizes and flattens game states to optimize training performance.

---

## Getting Started

### Prerequisites
Ensure the following are installed:
- Python 3.7 or later

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/s24261-pj/grundys-game-nai.git
   cd grundys-game-nai
   ```

2. Navigate to the Meet 7 directory:
   ```bash
   cd meet_7
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## File Structure
- `main.py`: The main script for training and testing the DQN agent.
- `replay_buffer.py`: Handles the replay buffer logic.
- `training.py`: Contains the core training loop and DQN algorithm.
- `utils.py`: Helper functions for preprocessing and logging.
- `requirements.txt`: Lists all dependencies required to run the project.

---

## Usage

### 1. Train the Agent
Run the following command to train the agent on the Freeway environment:
```bash
python main.py
```
This trains the agent for the specified number of episodes and outputs training logs for each episode.

### 2. Test the Agent
Once training is complete, the agent's performance can be evaluated. During testing, the agent plays the game using the learned policy and the total reward is displayed.

### Adjustable Hyperparameters
The following hyperparameters can be modified in the script to fine-tune performance:
- `episodes`: Number of training episodes.
- `batch_size`: Size of mini-batches for training.
- `gamma`: Discount factor for future rewards.
- `epsilon`: Initial exploration probability.
- `epsilon_min`: Minimum exploration probability.
- `epsilon_decay`: Decay rate for epsilon.
- `target_update_freq`: Frequency of target network updates.

### Example Output
```
Episode 1/10, Steps: 123, Reward: 1.0
Episode 2/10, Steps: 150, Reward: 2.0
...
Episode 10/10, Steps: 170, Reward: 4.0
Test Completed: Steps: 200, Reward: 5.0
```

---

## Authors
This project was created by:
- **s24260**
- **s24261**

---

## Media
