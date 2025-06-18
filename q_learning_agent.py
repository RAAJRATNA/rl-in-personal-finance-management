import numpy as np
import pickle
from collections import deque
import random
from finance_env import FinanceEnv

class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def add(self, experience, error=None):
        priority = self.max_priority if error is None else min(error + 1e-5, self.max_priority)
        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size):
        total_priority = sum(p ** self.alpha for p in self.priorities)
        probs = [p ** self.alpha / total_priority for p in self.priorities]
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = [(len(self.buffer) * probs[i]) ** (-self.beta) for i in indices]
        weights = np.array(weights) / max(weights)
        
        experiences = [self.buffer[i] for i in indices]
        return experiences, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = min(error + 1e-5, self.max_priority)
            self.max_priority = max(self.max_priority, self.priorities[idx])

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.2, episodes=1000):
        self.env = env
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate
        self.episodes = episodes
        self.q_table = {}
        
        # Enhanced learning parameters
        self.learning_rate_decay = 0.995
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.min_alpha = 0.01
        
        # Experience replay
        self.replay_buffer = PrioritizedReplayBuffer()
        self.batch_size = 32
        self.replay_start_size = 1000
        
        # Double Q-learning
        self.q_table_target = {}
        self.target_update_frequency = 10
        self.steps = 0

    def get_state_key(self, state):
        # Discretize state space for better generalization
        return tuple((state * 100).astype(int))

    def choose_action(self, state):
        key = self.get_state_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.env.action_space.n)
            self.q_table_target[key] = np.zeros(self.env.action_space.n)

        # Epsilon-greedy strategy with decay
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[key])

    def learn(self, state, action, reward, next_state, done):
        key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)

        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.env.action_space.n)
            self.q_table_target[key] = np.zeros(self.env.action_space.n)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.env.action_space.n)
            self.q_table_target[next_key] = np.zeros(self.env.action_space.n)

        # Store experience in replay buffer
        experience = (state, action, reward, next_state, done)
        self.replay_buffer.add(experience)

        # Update Q-table using experience replay
        if len(self.replay_buffer.buffer) >= self.replay_start_size:
            self._update_from_replay()

        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_frequency == 0:
            self.q_table_target = self.q_table.copy()

        # Update learning parameters
        self.alpha = max(self.min_alpha, self.alpha * self.learning_rate_decay)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _update_from_replay(self):
        # Sample from replay buffer
        experiences, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        # Calculate TD errors for each experience
        td_errors = []
        for exp, weight in zip(experiences, weights):
            state, action, reward, next_state, done = exp
            key = self.get_state_key(state)
            next_key = self.get_state_key(next_state)
            
            # Double Q-learning update
            if done:
                target = reward
            else:
                next_action = np.argmax(self.q_table[next_key])
                target = reward + self.gamma * self.q_table_target[next_key][next_action]
            
            current_q = self.q_table[key][action]
            td_error = abs(target - current_q)
            td_errors.append(td_error)
            
            # Update Q-value with importance sampling weight
            self.q_table[key][action] += self.alpha * weight * (target - current_q)
        
        # Update priorities in replay buffer
        self.replay_buffer.update_priorities(indices, td_errors)

    def train(self):
        best_reward = float('-inf')
        rewards_history = []
        episode_rewards = []

        for episode in range(self.episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.learn(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

            rewards_history.append(episode_reward)
            episode_rewards.append(episode_reward)
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                with open("model/best_q_table.pkl", "wb") as f:
                    pickle.dump(self.q_table, f)

            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                print(f"Episode {episode + 1}/{self.episodes}")
                print(f"Average Reward (last 100): {avg_reward:.2f}")
                print(f"Epsilon: {self.epsilon:.3f}, Alpha: {self.alpha:.3f}")
                print(f"Replay Buffer Size: {len(self.replay_buffer.buffer)}")

        # Save final model
        with open("model/q_table.pkl", "wb") as f:
            pickle.dump(self.q_table, f)
        print("✅ Training complete. Q-table saved.")
        
        return rewards_history

    def get_recommendation(self, state):
        """Get the best action for a given state"""
        key = self.get_state_key(state)
        if key not in self.q_table:
            return np.random.choice(self.env.action_space.n)
        return np.argmax(self.q_table[key])
