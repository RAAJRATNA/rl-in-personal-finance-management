import os
from finance_env import FinanceEnv
from q_learning_agent import QLearningAgent
import numpy as np
import matplotlib.pyplot as plt

def plot_training_progress(rewards_history, save_path="training_progress.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, label='Episode Reward')
    plt.plot(np.convolve(rewards_history, np.ones(100)/100, mode='valid'), 
             label='100-Episode Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def main():
    """Train the RL agent for financial recommendations"""
    print("🚀 Starting RL agent training...")
    
    # Create model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)
    
    # Create environment
    env = FinanceEnv()
    
    # Create agent with enhanced parameters
    agent = QLearningAgent(
        env=env,
        alpha=0.1,      # Learning rate
        gamma=0.95,     # Discount factor (increased for better long-term planning)
        epsilon=0.3,    # Initial exploration rate
        episodes=2000   # Increased number of episodes
    )
    
    # Train the agent
    print("📊 Training agent...")
    rewards_history = agent.train()
    
    # Plot training progress
    os.makedirs("plots", exist_ok=True)
    plot_training_progress(rewards_history, "plots/training_progress.png")
    
    print("✅ Training complete!")
    print(f"Model saved in 'model/best_q_table.pkl'")

if __name__ == "__main__":
    main() 