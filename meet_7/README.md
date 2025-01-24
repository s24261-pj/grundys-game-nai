# Deep Q-Network (DQN) Implementation for Atari Freeway

This project demonstrates a basic implementation of a Deep Q-Network (DQN) to train an agent to play the Atari game **Freeway** using the `gymnasium` and `tensorflow` libraries. The DQN agent uses reinforcement learning to learn optimal strategies for maximizing rewards in the game.

## Problem Statement
In Atari's Freeway, the objective is for the agent (a chicken) to cross a multi-lane highway filled with moving cars and reach the other side without getting hit. The challenge is to determine the optimal sequence of actions to achieve this goal efficiently.

## Features of This Implementation
1. **Custom DQN Architecture**:
   - A neural network with two hidden layers of 128 neurons each, using ReLU activation.
   - Output layer with linear activation to estimate Q-values for each possible action.

2. **Experience Replay Buffer**:
   - A replay buffer is implemented to store past experiences, enabling efficient training by sampling mini-batches.

3. **Epsilon-Greedy Policy**:
   - Balances exploration and exploitation to choose actions.

4. **Target Network**:
   - A separate target network is used for stability in training by periodically updating weights.

5. **Game Preprocessing**:
   - States are normalized and flattened to optimize training.

## Getting Started
### Prerequisites
Before running the code, ensure you have the following installed:

- Python 3.7 or later

Clone the repository:
```bash
    git clone https://github.com/s24261-pj/grundys-game-nai.git
    cd grundys-game-nai
```

Navigate to the Meet 6 directory:
```bash
cd meet_7
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

### File Structure
- `dqn_freeway.py`: The main Python script for training and testing the DQN agent.
- `requirements.txt`: A file containing all the dependencies required to run the project.

## Usage

### 1. Train the Agent
Run the script to train the agent on the Freeway environment:
```bash
python main.py
```
This trains the agent for 10 episodes and outputs training logs for each episode.

### 2. Test the Agent
After training, the agent is tested in the same environment to evaluate its performance. During the test phase, the agent plays the game using the learned policy and its total reward is displayed.

### Adjustable Hyperparameters
The following hyperparameters can be modified in the script:

- `episodes`: Number of training episodes.
- `batch_size`: Mini-batch size for training.
- `gamma`: Discount factor for future rewards.
- `epsilon`: Initial exploration probability.
- `epsilon_min`: Minimum exploration probability.
- `epsilon_decay`: Decay rate for epsilon.
- `target_update_freq`: Frequency for updating the target network.

### Example Output
```text
Episode 1/10, Steps: 123, Reward: 1.0
Episode 2/10, Steps: 150, Reward: 2.0
...
Episode 10/10, Steps: 170, Reward: 4.0
Test Completed: Steps: 200, Reward: 5.0
```

## Authors
This project was created by:
- s24260
- s24261

## Media
